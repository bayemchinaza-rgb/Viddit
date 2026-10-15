/*
  Viddit: browser-based instance segmentation with YOLO26n-seg (ONNX)
  and ONNX Runtime Web.
*/

const MODEL_URL = './yolo26n-seg.onnx';
const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.35;
const NMS_IOU_THRESHOLD = 0.45;
const MAX_DETECTIONS = 30;
const MASK_THRESHOLD = 0.5;
const MAX_IMAGE_SIDE = 1920;
const ORT_WASM_PATH =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';

const CLASS_NAMES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
  'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
  'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
  'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
  'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
  'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
  'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
  'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
  'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

/* ---------------------------------------------------------------
   MODEL LOADING
   --------------------------------------------------------------- */

let modelPromise = null;

function getModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      if (typeof ort === 'undefined') {
        throw new Error(
          'ONNX Runtime could not load. Check your internet connection and refresh.'
        );
      }

      ort.env.wasm.wasmPaths = ORT_WASM_PATH;
      ort.env.wasm.numThreads = 1;
      ort.env.logLevel = 'error';

      const session = await ort.InferenceSession.create(
        MODEL_URL,
        { executionProviders: ['wasm'] }
      );

      return {
        session,
        inputName: session.inputNames[0],
        outputNames: session.outputNames
      };
    })();

    modelPromise.catch(() => {
      modelPromise = null;
    });
  }

  return modelPromise;
}

/* ---------------------------------------------------------------
   IMAGE PREPROCESSING
   --------------------------------------------------------------- */

const prepCanvas = document.createElement('canvas');
prepCanvas.width = INPUT_SIZE;
prepCanvas.height = INPUT_SIZE;

const prepCtx = prepCanvas.getContext('2d', {
  willReadFrequently: true
});

const inputBuffer = new Float32Array(
  3 * INPUT_SIZE * INPUT_SIZE
);

function sourceSize(src) {
  if (src instanceof HTMLVideoElement) {
    return [src.videoWidth, src.videoHeight];
  }

  return [
    src.naturalWidth || src.width,
    src.naturalHeight || src.height
  ];
}

function preprocess(src) {
  const [sw, sh] = sourceSize(src);

  if (!sw || !sh) {
    throw new Error('The input image or video frame is not ready.');
  }

  const scale = Math.min(
    INPUT_SIZE / sw,
    INPUT_SIZE / sh
  );

  const nw = Math.round(sw * scale);
  const nh = Math.round(sh * scale);

  const dx = Math.floor((INPUT_SIZE - nw) / 2);
  const dy = Math.floor((INPUT_SIZE - nh) / 2);

  prepCtx.fillStyle = 'rgb(114,114,114)';
  prepCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);

  prepCtx.drawImage(
    src, 0, 0, sw, sh,
    dx, dy, nw, nh
  );

  const { data } = prepCtx.getImageData(
    0, 0, INPUT_SIZE, INPUT_SIZE
  );

  const plane = INPUT_SIZE * INPUT_SIZE;

  for (let i = 0, j = 0; i < plane; i++, j += 4) {
    inputBuffer[i] = data[j] / 255;
    inputBuffer[i + plane] = data[j + 1] / 255;
    inputBuffer[i + 2 * plane] = data[j + 2] / 255;
  }

  return { scale, dx, dy, sw, sh };
}

/* ---------------------------------------------------------------
   SEGMENTATION OUTPUT DECODING
   --------------------------------------------------------------- */

const clamp = (v, lo, hi) =>
  Math.min(Math.max(v, lo), hi);

const sigmoid = x => 1 / (1 + Math.exp(-x));

function boxIoU(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);

  const intersection =
    Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

  const areaA =
    Math.max(0, a.x2 - a.x1) *
    Math.max(0, a.y2 - a.y1);

  const areaB =
    Math.max(0, b.x2 - b.x1) *
    Math.max(0, b.y2 - b.y1);

  return intersection /
    Math.max(areaA + areaB - intersection, 1e-6);
}

function classAwareNms(candidates) {
  candidates.sort((a, b) => b.conf - a.conf);

  if (candidates.length > 1500) {
    candidates.length = 1500;
  }

  const kept = [];

  for (const candidate of candidates) {
    const overlaps = kept.some(
      k =>
        k.cls === candidate.cls &&
        boxIoU(k, candidate) > NMS_IOU_THRESHOLD
    );

    if (overlaps) continue;

    kept.push(candidate);

    if (kept.length >= MAX_DETECTIONS) break;
  }

  return kept;
}

function toInputPixels(value) {
  return Math.abs(value) <= 2
    ? value * INPUT_SIZE
    : value;
}

function buildMask(
  coeffs,
  protoData,
  maskDim,
  maskH,
  maskW,
  box
) {
  const mask = new Uint8Array(maskH * maskW);

  const x0 = clamp(
    Math.floor(box.x1 / INPUT_SIZE * maskW),
    0,
    maskW - 1
  );

  const y0 = clamp(
    Math.floor(box.y1 / INPUT_SIZE * maskH),
    0,
    maskH - 1
  );

  const x1 = clamp(
    Math.ceil(box.x2 / INPUT_SIZE * maskW),
    0,
    maskW
  );

  const y1 = clamp(
    Math.ceil(box.y2 / INPUT_SIZE * maskH),
    0,
    maskH
  );

  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const pixel = y * maskW + x;
      let logit = 0;

      for (let m = 0; m < maskDim; m++) {
        logit +=
          coeffs[m] *
          protoData[m * maskH * maskW + pixel];
      }

      if (sigmoid(logit) >= MASK_THRESHOLD) {
        mask[pixel] = 1;
      }
    }
  }

  return mask;
}

function postprocess(results, outputNames, meta) {
  const outputs = outputNames.map(name => ({
    name,
    tensor: results[name]
  }));

  const protoEntry = outputs.find(
    o => o.tensor.dims.length === 4
  );

  const predEntry = outputs.find(
    o => o.tensor.dims.length === 3
  );

  if (!protoEntry || !predEntry) {
    const shapes = outputs.map(
      o => `${o.name}: [${o.tensor.dims.join(', ')}]`
    ).join('; ');

    throw new Error(
      `Unexpected segmentation model outputs (${shapes}). ` +
      'Check the ONNX export format and output shapes.'
    );
  }

  const proto = protoEntry.tensor;
  const protoDims = proto.dims;

  const maskDim = protoDims[1];
  const maskH = protoDims[2];
  const maskW = protoDims[3];
  const protoData = proto.data;

  const pred = predEntry.tensor;
  const dims = pred.dims;
  const data = pred.data;

  const candidates = [];

  /*
    End-to-end exports may return rows like:
    [x1, y1, x2, y2, confidence, class, mask coefficients...]
  */
  const rowMajorEndToEnd =
    dims[2] >= 6 + maskDim &&
    dims[2] <= 512 &&
    dims[1] <= 1000;

  if (rowMajorEndToEnd) {
    const rows = dims[1];
    const stride = dims[2];

    for (let i = 0; i < rows; i++) {
      const offset = i * stride;
      const conf = data[offset + 4];

      if (
        !Number.isFinite(conf) ||
        conf < CONFIDENCE_THRESHOLD
      ) {
        continue;
      }

      const cls = Math.round(data[offset + 5]);

      const x1 = toInputPixels(data[offset]);
      const y1 = toInputPixels(data[offset + 1]);
      const x2 = toInputPixels(data[offset + 2]);
      const y2 = toInputPixels(data[offset + 3]);

      if (
        x2 <= x1 ||
        y2 <= y1 ||
        cls < 0 ||
        cls >= CLASS_NAMES.length
      ) {
        continue;
      }

      const coeffs = Array.from(
        data.slice(
          offset + 6,
          offset + 6 + maskDim
        )
      );

      candidates.push({
        x1, y1, x2, y2, conf, cls, coeffs
      });
    }
  } else {
    /*
      Standard raw output:
      [1, 4 + class count + mask coefficients, anchors]

      Some exports may use:
      [1, anchors, channels]
    */
    const channelFirst = dims[1] < dims[2];

    const channels = channelFirst ? dims[1] : dims[2];
    const anchors = channelFirst ? dims[2] : dims[1];

    const numClasses = channels - 4 - maskDim;

    if (
      numClasses < 1 ||
      numClasses > CLASS_NAMES.length
    ) {
      throw new Error(
        `Could not interpret segmentation output shape ` +
        `[${dims.join(', ')}] with ${maskDim} mask channels.`
      );
    }

    const value = (channel, anchor) =>
      channelFirst
        ? data[channel * anchors + anchor]
        : data[anchor * channels + channel];

    for (let i = 0; i < anchors; i++) {
      let conf = -Infinity;
      let cls = -1;

      for (let c = 0; c < numClasses; c++) {
        const score = value(4 + c, i);

        if (score > conf) {
          conf = score;
          cls = c;
        }
      }

      if (
        !Number.isFinite(conf) ||
        conf < CONFIDENCE_THRESHOLD
      ) {
        continue;
      }

      const cx = toInputPixels(value(0, i));
      const cy = toInputPixels(value(1, i));
      const bw = toInputPixels(value(2, i));
      const bh = toInputPixels(value(3, i));

      const x1 = cx - bw / 2;
      const y1 = cy - bh / 2;
      const x2 = cx + bw / 2;
      const y2 = cy + bh / 2;

      if (x2 <= x1 || y2 <= y1) continue;

      const coeffs = new Array(maskDim);

      for (let m = 0; m < maskDim; m++) {
        coeffs[m] = value(
          4 + numClasses + m,
          i
        );
      }

      candidates.push({
        x1, y1, x2, y2, conf, cls, coeffs
      });
    }
  }

  const selected = classAwareNms(candidates);

  return selected.map(d => {
    const x1 = clamp(
      (d.x1 - meta.dx) / meta.scale,
      0,
      meta.sw
    );

    const y1 = clamp(
      (d.y1 - meta.dy) / meta.scale,
      0,
      meta.sh
    );

    const x2 = clamp(
      (d.x2 - meta.dx) / meta.scale,
      0,
      meta.sw
    );

    const y2 = clamp(
      (d.y2 - meta.dy) / meta.scale,
      0,
      meta.sh
    );

    return {
      x: x1,
      y: y1,
      w: Math.max(0, x2 - x1),
      h: Math.max(0, y2 - y1),
      conf: d.conf,
      cls: d.cls,
      label: CLASS_NAMES[d.cls] || `object ${d.cls}`,

      mask: buildMask(
        d.coeffs,
        protoData,
        maskDim,
        maskH,
        maskW,
        d
      ),

      scale: meta.scale,
      metaDx: meta.dx / meta.scale,
      metaDy: meta.dy / meta.scale,
      maskW,
      maskH
    };
  }).filter(d => d.w >= 2 && d.h >= 2);
}

/* ---------------------------------------------------------------
   INFERENCE
   --------------------------------------------------------------- */

let inferenceQueue = Promise.resolve();

async function predictWithModel(source) {
  const {
    session,
    inputName,
    outputNames
  } = await getModel();

  const startedAt = performance.now();
  const meta = preprocess(source);

  const tensor = new ort.Tensor(
    'float32',
    inputBuffer,
    [1, 3, INPUT_SIZE, INPUT_SIZE]
  );

  const results = await session.run({
    [inputName]: tensor
  });

  const detections = postprocess(
    results,
    outputNames,
    meta
  );

  return {
    detections,
    ms: performance.now() - startedAt
  };
}

/* ---------------------------------------------------------------
   SEGMENTATION RENDERING
   --------------------------------------------------------------- */

const colorFor = cls =>
  `hsl(${(cls * 47) % 360} 85% 58%)`;

function colorRgb(cls) {
  const h = ((cls * 47) % 360) / 60;
  const c = 0.85;
  const x = c * (1 - Math.abs(h % 2 - 1));
  const m = 0.15;

  let r = 0, g = 0, b = 0;

  if (h < 1) [r, g, b] = [c, x, 0];
  else if (h < 2) [r, g, b] = [x, c, 0];
  else if (h < 3) [r, g, b] = [0, c, x];
  else if (h < 4) [r, g, b] = [0, x, c];
  else if (h < 5) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];

  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255)
  ];
}

const maskCanvas = document.createElement('canvas');
const maskCtx = maskCanvas.getContext('2d', {
  willReadFrequently: true
});

function drawDetections(ctx, detections, w, h) {
  const fs = Math.max(
    12,
    Math.round(Math.min(w, h) / 36)
  );

  ctx.font = `700 ${fs}px Inter, system-ui, sans-serif`;
  ctx.textBaseline = 'top';

  detections.forEach(d => {
    const [r, g, b] = colorRgb(d.cls);
    const mw = d.maskW;
    const mh = d.maskH;

    if (
      maskCanvas.width !== mw ||
      maskCanvas.height !== mh
    ) {
      maskCanvas.width = mw;
      maskCanvas.height = mh;
    }

    const pixels = maskCtx.createImageData(mw, mh);

    for (let i = 0; i < d.mask.length; i++) {
      if (!d.mask[i]) continue;

      const p = i * 4;

      pixels.data[p] = r;
      pixels.data[p + 1] = g;
      pixels.data[p + 2] = b;
      pixels.data[p + 3] = 112;
    }

    maskCtx.putImageData(pixels, 0, 0);

    /*
      Map the prototype mask back from the letterboxed model input.
      The mask itself is rendered as a transparent colored overlay.
    */
    ctx.drawImage(
      maskCanvas,
      -d.metaDx,
      -d.metaDy,
      INPUT_SIZE / d.scale,
      INPUT_SIZE / d.scale
    );

    const text = `${d.label} ${Math.round(d.conf * 100)}%`;
    const tw = ctx.measureText(text).width + fs * 0.6;
    const th = fs * 1.35;

    const tx = clamp(d.x, 0, Math.max(0, w - tw));
    const ty = d.y - th >= 0 ? d.y - th : d.y;

    ctx.fillStyle = colorFor(d.cls);
    ctx.fillRect(tx, ty, tw, th);

    ctx.fillStyle = '#06121a';
    ctx.fillText(text, tx + fs * 0.3, ty + fs * 0.17);
  });
}

function summarize(detections) {
  if (!detections.length) {
    return 'No objects found.';
  }

  const counts = {};

  detections.forEach(d => {
    counts[d.label] = (counts[d.label] || 0) + 1;
  });

  const parts = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => `${count} ${name}`);

  return `${detections.length} object` +
    `${detections.length > 1 ? 's' : ''}: ${parts.join(', ')}`;
}

function fitStage(stage, vw, vh) {
  if (!stage || !vw || !vh) return;

  stage.style.aspectRatio = `${vw} / ${vh}`;
  stage.style.width = `min(100%, calc(72vh * ${vw / vh}))`;
  stage.style.marginInline = 'auto';
}

function sizeCanvas(canvas, w, h) {
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

/* ---------------------------------------------------------------
   PAGE WIRING
   --------------------------------------------------------------- */

function init() {
  const $ = id => document.getElementById(id);
  const setText = (el, msg) => {
    if (el) el.textContent = msg;
  };

  const errMsg = e =>
    e && e.message ? e.message : 'Something went wrong.';

  const modelNote = () =>
    modelPromise
      ? ''
      : 'Loading the model (first time only)... ';

  /* ---------------- IMAGE ---------------- */

  const imageInput = $('imageInput');
  const captureInput = $('captureImageInput');
  const previewContainer = $('previewContainer');
  const resultCanvas = $('resultCanvas');
  const imagePreview = $('imagePreview');
  const detectButton = $('detectButton');
  const resetButton = $('resetButton');
  const fileNameEl = $('fileName');
  const detectionStatus = $('detectionStatus');

  const srcCanvas = document.createElement('canvas');
  let imageBusy = false;

  async function loadImageFile(file) {
    if (!file) return;

    const url = URL.createObjectURL(file);

    try {
      const img = new Image();
      img.src = url;
      await img.decode();

      const k = Math.min(
        1,
        MAX_IMAGE_SIDE /
          Math.max(img.naturalWidth, img.naturalHeight)
      );

      const w = Math.round(img.naturalWidth * k);
      const h = Math.round(img.naturalHeight * k);

      srcCanvas.width = w;
      srcCanvas.height = h;

      srcCanvas.getContext('2d').drawImage(img, 0, 0, w, h);

      resultCanvas.width = w;
      resultCanvas.height = h;

      resultCanvas.getContext('2d').drawImage(
        srcCanvas,
        0,
        0
      );

      imagePreview.hidden = true;
      resultCanvas.hidden = false;
      previewContainer.hidden = false;

      fileNameEl.textContent = file.name || 'Camera picture';
      detectButton.disabled = false;
      resetButton.disabled = false;

      setText(
        detectionStatus,
        'Image ready. Press "Segment objects".'
      );
    } catch (e) {
      console.error(e);
      setText(
        detectionStatus,
        'Could not open that image. Try a JPG or PNG.'
      );
      previewContainer.hidden = false;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  [imageInput, captureInput].forEach(input => {
    if (!input) return;

    input.addEventListener('change', async ev => {
      await loadImageFile(
        ev.target.files && ev.target.files[0]
      );

      ev.target.value = '';
    });
  });

  detectButton.addEventListener('click', async () => {
    if (imageBusy) return;

    imageBusy = true;
    detectButton.disabled = true;

    setText(
      detectionStatus,
      modelNote() + 'Running segmentation...'
    );

    try {
      const { detections, ms } =
        await predictWithModel(srcCanvas);

      const ctx = resultCanvas.getContext('2d');

      ctx.drawImage(srcCanvas, 0, 0);
      drawDetections(
        ctx,
        detections,
        resultCanvas.width,
        resultCanvas.height
      );

      setText(
        detectionStatus,
        `${summarize(detections)} (${Math.round(ms)} ms)`
      );
    } catch (e) {
      console.error(e);
      setText(
        detectionStatus,
        'Segmentation failed: ' + errMsg(e)
      );
    } finally {
      imageBusy = false;
      detectButton.disabled = false;
    }
  });

  resetButton.addEventListener('click', () => {
    resultCanvas.getContext('2d').clearRect(
      0, 0, resultCanvas.width, resultCanvas.height
    );

    resultCanvas.hidden = true;
    imagePreview.hidden = true;
    previewContainer.hidden = true;

    detectButton.disabled = true;
    resetButton.disabled = true;

    fileNameEl.textContent = '';
    setText(detectionStatus, 'Ready.');
  });

  /* ---------------- LIVE CAMERA ---------------- */

  const camStartBtn = $('cameraStartButton');
  const camStopBtn = $('cameraStopButton');
  const camShotBtn = $('cameraCaptureButton');

  const camVideo = $('cameraPreview');
  const camCanvas = $('cameraResultCanvas');
  const camStage = $('cameraStage');

  const camWorkspace = $('cameraWorkspace');
  const camStartPanel = $('cameraStartPanel');

  const camPermission = $('cameraPermissionStatus');
  const camStatus = $('cameraDetectionStatus');

  let camStream = null;
  let camRunning = false;
  let camBusy = false;
  let camRaf = 0;
  let camLastDone = 0;
  let camFps = 0;
  let camLast = [];

  async function startCamera() {
    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {
      setText(
        camPermission,
        'This browser cannot use the camera. Open the site over HTTPS.'
      );
      return;
    }

    camStartBtn.disabled = true;
    setText(camPermission, 'Requesting camera...');

    try {
      camStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });
    } catch (e) {
      console.error(e);

      setText(
        camPermission,
        e && e.name === 'NotAllowedError'
          ? 'Camera permission was denied. Allow it in browser settings.'
          : 'No camera available: ' + errMsg(e)
      );

      camStartBtn.disabled = false;
      return;
    }

    camVideo.srcObject = camStream;
    camVideo.hidden = false;
    camCanvas.hidden = false;
    camWorkspace.hidden = false;
    camStartPanel.hidden = true;

    camStopBtn.disabled = false;
    camStartBtn.disabled = false;

    const fit = () => fitStage(
      camStage,
      camVideo.videoWidth,
      camVideo.videoHeight
    );

    camVideo.onloadedmetadata = fit;

    try {
      await camVideo.play();
    } catch (e) {
      // The camera video is muted; autoplay may be permitted.
    }

    fit();

    setText(
      camStatus,
      modelNote() + 'Starting live segmentation...'
    );

    camRunning = true;
    camLastDone = 0;
    camFps = 0;

    try {
      await getModel();
    } catch (e) {
      setText(
        camStatus,
        'Could not load the model: ' + errMsg(e)
      );
      camRunning = false;
      return;
    }

    camRaf = requestAnimationFrame(camTick);
  }

  async function camTick() {
    if (!camRunning) return;

    if (
      !camBusy &&
      camVideo.readyState >= 2 &&
      camVideo.videoWidth
    ) {
      camBusy = true;

      try {
        const { detections, ms } =
          await predictWithModel(camVideo);

        if (!camRunning) return;

        const w = camVideo.videoWidth;
        const h = camVideo.videoHeight;

        sizeCanvas(camCanvas, w, h);

        const ctx = camCanvas.getContext('2d');
        ctx.clearRect(0, 0, w, h);

        drawDetections(ctx, detections, w, h);
        camLast = detections;

        const now = performance.now();

        if (camLastDone) {
          const f = 1000 / (now - camLastDone);
          camFps = camFps
            ? camFps * 0.8 + f * 0.2
            : f;
        }

        camLastDone = now;

        setText(
          camStatus,
          `${summarize(detections)} · ${Math.round(ms)} ms · ${camFps.toFixed(1)} fps`
        );
      } catch (e) {
        console.error(e);

        setText(
          camStatus,
          'Segmentation stopped: ' + errMsg(e)
        );

        camRunning = false;
      } finally {
        camBusy = false;
      }
    }

    if (camRunning) {
      camRaf = requestAnimationFrame(camTick);
    }
  }

  function stopCamera() {
    camRunning = false;
    cancelAnimationFrame(camRaf);

    if (camStream) {
      camStream.getTracks().forEach(track => track.stop());
      camStream = null;
    }

    camVideo.srcObject = null;
    camVideo.hidden = true;

    camCanvas.getContext('2d').clearRect(
      0, 0, camCanvas.width, camCanvas.height
    );

    camCanvas.hidden = true;
    camWorkspace.hidden = true;
    camStartPanel.hidden = false;
    camStopBtn.disabled = true;
    camLast = [];

    setText(camPermission, 'Camera off.');
  }

  function captureFromCamera() {
    if (!camStream || camVideo.readyState < 2) {
      setText(camStatus, 'Camera not ready yet.');
      return;
    }

    const w = camVideo.videoWidth;
    const h = camVideo.videoHeight;

    const shot = document.createElement('canvas');
    shot.width = w;
    shot.height = h;

    const ctx = shot.getContext('2d');
    ctx.drawImage(camVideo, 0, 0, w, h);

    drawDetections(ctx, camLast, w, h);

    shot.toBlob(blob => {
      if (!blob) {
        setText(camStatus, 'Could not save the picture.');
        return;
      }

      const a = document.createElement('a');

      a.href = URL.createObjectURL(blob);
      a.download =
        'viddit-' +
        new Date().toISOString().replace(/[:.]/g, '-') +
        '.png';

      document.body.appendChild(a);
      a.click();
      a.remove();

      setTimeout(() => URL.revokeObjectURL(a.href), 1500);

      setText(camStatus, 'Picture saved to your downloads.');
    }, 'image/png');
  }

  camStartBtn.addEventListener('click', startCamera);
  camStopBtn.addEventListener('click', stopCamera);
  camShotBtn.addEventListener('click', captureFromCamera);

  /* ---------------- VIDEO FILE ---------------- */

  const vidInput = $('videoInput');
  const vidWorkspace = $('videoWorkspace');
  const vidName = $('videoFileName');

  const vidVideo = $('videoPreview');
  const vidCanvas = $('videoResultCanvas');
  const vidStage = $('videoStage');

  const vidStatus = $('videoDetectionStatus');

  const vidStartBtn = $('videoDetectButton');
  const vidStopBtn = $('videoStopButton');
  const vidResetBtn = $('videoResetButton');

  let vidURL = null;
  let vidRunning = false;
  let vidBusy = false;
  let vidRaf = 0;
  let vidLastTime = -1;
  let vidLastDone = 0;
  let vidFps = 0;

  const clearVidCanvas = () => {
    vidCanvas.getContext('2d').clearRect(
      0, 0, vidCanvas.width, vidCanvas.height
    );
  };

  vidInput.addEventListener('change', ev => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;

    stopVideoDetection();

    if (vidURL) URL.revokeObjectURL(vidURL);

    vidURL = URL.createObjectURL(file);
    vidVideo.src = vidURL;
    vidName.textContent = file.name;

    vidWorkspace.hidden = false;
    vidCanvas.hidden = false;

    vidStartBtn.disabled = true;
    vidResetBtn.disabled = false;
    vidStopBtn.disabled = true;

    setText(vidStatus, 'Loading video...');
    ev.target.value = '';
  });

  vidVideo.addEventListener('loadedmetadata', () => {
    fitStage(
      vidStage,
      vidVideo.videoWidth,
      vidVideo.videoHeight
    );

    vidStartBtn.disabled = false;

    setText(
      vidStatus,
      'Video ready. Press "Start segmentation".'
    );
  });

  vidVideo.addEventListener('error', () => {
    if (vidVideo.getAttribute('src')) {
      setText(
        vidStatus,
        'This browser cannot play that video format. Try MP4 (H.264) or WebM.'
      );
    }
  });

  vidVideo.addEventListener('seeking', clearVidCanvas);

  vidVideo.addEventListener('ended', () => {
    if (vidRunning) {
      stopVideoDetection();
      setText(vidStatus, 'Video finished.');
    }
  });

  async function startVideoDetection() {
    if (vidRunning) return;

    vidRunning = true;
    vidLastTime = -1;
    vidLastDone = 0;
    vidFps = 0;

    vidStartBtn.disabled = true;
    vidStopBtn.disabled = false;

    setText(
      vidStatus,
      modelNote() + 'Starting segmentation...'
    );

    try {
      await getModel();
    } catch (e) {
      setText(
        vidStatus,
        'Could not load the model: ' + errMsg(e)
      );

      stopVideoDetection();
      return;
    }

    if (!vidRunning) return;

    if (vidVideo.ended) {
      vidVideo.currentTime = 0;
    }

    try {
      await vidVideo.play();
    } catch (e) {
      // The user can press play manually if playback is blocked.
    }

    vidRaf = requestAnimationFrame(vidTick);
  }

  async function vidTick() {
    if (!vidRunning) return;

    if (
      !vidBusy &&
      vidVideo.readyState >= 2 &&
      vidVideo.currentTime !== vidLastTime
    ) {
      vidBusy = true;
      vidLastTime = vidVideo.currentTime;

      try {
        const { detections, ms } =
          await predictWithModel(vidVideo);

        if (!vidRunning) return;

        const w = vidVideo.videoWidth;
        const h = vidVideo.videoHeight;

        sizeCanvas(vidCanvas, w, h);

        const ctx = vidCanvas.getContext('2d');
        ctx.clearRect(0, 0, w, h);

        drawDetections(ctx, detections, w, h);

        const now = performance.now();

        if (vidLastDone) {
          const f = 1000 / (now - vidLastDone);
          vidFps = vidFps
            ? vidFps * 0.8 + f * 0.2
            : f;
        }

        vidLastDone = now;

        // Slow playback when inference is slower than normal video playback.
        vidVideo.playbackRate = Math.min(
          1,
          Math.max(0.1, 300 / Math.max(ms, 1))
        );

        setText(
          vidStatus,
          `${summarize(detections)} · ${Math.round(ms)} ms · ${vidFps.toFixed(1)} fps`
        );
      } catch (e) {
        console.error(e);

        setText(
          vidStatus,
          'Segmentation stopped: ' + errMsg(e)
        );

        stopVideoDetection();
        return;
      } finally {
        vidBusy = false;
      }
    }

    if (vidRunning) {
      vidRaf = requestAnimationFrame(vidTick);
    }
  }

  function stopVideoDetection() {
    vidRunning = false;
    cancelAnimationFrame(vidRaf);

    vidVideo.playbackRate = 1;
    clearVidCanvas();

    vidStopBtn.disabled = true;

    vidStartBtn.disabled = !(
      vidVideo.getAttribute('src') &&
      vidVideo.readyState >= 1
    );
  }

  vidStartBtn.addEventListener(
    'click',
    startVideoDetection
  );

  vidStopBtn.addEventListener('click', () => {
    stopVideoDetection();
    setText(vidStatus, 'Segmentation stopped.');
  });

  vidResetBtn.addEventListener('click', () => {
    stopVideoDetection();

    vidVideo.pause();
    vidVideo.removeAttribute('src');
    vidVideo.load();

    if (vidURL) {
      URL.revokeObjectURL(vidURL);
      vidURL = null;
    }

    vidWorkspace.hidden = true;
    vidCanvas.hidden = true;
    vidName.textContent = '';

    vidStartBtn.disabled = true;
    vidResetBtn.disabled = true;

    setText(vidStatus, 'Video ready.');
  });

  /* ---------------- WORKSPACE CLEANUP ---------------- */

  document.addEventListener(
    'viddit:workspace-change',
    ev => {
      const workspace = ev.detail && ev.detail.workspace;

      if (workspace !== 'camera' && camStream) {
        stopCamera();
      }

      if (workspace !== 'video' && vidRunning) {
        stopVideoDetection();
        vidVideo.pause();
        setText(vidStatus, 'Segmentation stopped.');
      }
    }
  );
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
