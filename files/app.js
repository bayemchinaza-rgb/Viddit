/*
  Viddit: browser object detection with YOLO26n (ONNX) + ONNX Runtime Web.

  How one detection works:
    1. PREPARE  Resize the picture to fit 640x640 (keeping its shape, grey bars fill the gaps),
                then turn the pixels into numbers the model understands.
    2. RUN      The model returns 300 candidate boxes, each as 6 numbers:
                x1, y1, x2, y2, confidence, class.
                YOLO26 removes duplicate boxes by itself, so no extra clean-up step is needed.
    3. MAP BACK Keep boxes above the confidence threshold and convert them from the
                640x640 space back to the original picture's pixels.
*/

const MODEL_URL = './yolo26n.onnx';
const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.35;   // raise for fewer, surer boxes; lower to see more
const MAX_IMAGE_SIDE = 1920;         // very large photos are shrunk to this before processing
const ORT_WASM_PATH = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';

const CLASS_NAMES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush'
];

/* ------------------------------------------------------------------
   MODEL: loaded once, the first time it is needed
   ------------------------------------------------------------------ */
let modelPromise = null;

function getModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      if (typeof ort === 'undefined') {
        throw new Error('ONNX Runtime could not load. Check your internet connection and refresh.');
      }
      ort.env.wasm.wasmPaths = ORT_WASM_PATH;
      ort.env.wasm.numThreads = 1;      // GitHub Pages cannot enable multi-threading
      ort.env.logLevel = 'error';
      const session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
      return { session, inputName: session.inputNames[0], outputName: session.outputNames[0] };
    })();
    modelPromise.catch(() => { modelPromise = null; });   // allow a retry after a failure
  }
  return modelPromise;
}

/* ------------------------------------------------------------------
   STEP 1: PREPARE the picture
   ------------------------------------------------------------------ */
const prepCanvas = document.createElement('canvas');
prepCanvas.width = prepCanvas.height = INPUT_SIZE;
const prepCtx = prepCanvas.getContext('2d', { willReadFrequently: true });
const inputBuffer = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

function sourceSize(src) {
  if (src instanceof HTMLVideoElement) return [src.videoWidth, src.videoHeight];
  return [src.naturalWidth || src.width, src.naturalHeight || src.height];
}

function preprocess(src) {
  const [sw, sh] = sourceSize(src);
  const scale = Math.min(INPUT_SIZE / sw, INPUT_SIZE / sh);
  const nw = Math.round(sw * scale), nh = Math.round(sh * scale);
  const dx = Math.floor((INPUT_SIZE - nw) / 2), dy = Math.floor((INPUT_SIZE - nh) / 2);

  prepCtx.fillStyle = 'rgb(114,114,114)';
  prepCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  prepCtx.drawImage(src, 0, 0, sw, sh, dx, dy, nw, nh);

  // Pixels arrive as R,G,B,A,R,G,B,A... The model wants all reds, then all greens, then all blues, scaled 0 to 1.
  const { data } = prepCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const plane = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0, j = 0; i < plane; i++, j += 4) {
    inputBuffer[i] = data[j] / 255;
    inputBuffer[i + plane] = data[j + 1] / 255;
    inputBuffer[i + 2 * plane] = data[j + 2] / 255;
  }
  return { scale, dx, dy, sw, sh };
}

/* ------------------------------------------------------------------
   STEP 3: MAP BACK to the original picture
   ------------------------------------------------------------------ */
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

function postprocess(out, meta) {
  const detections = [];
  for (let i = 0; i + 5 < out.length; i += 6) {
    const conf = out[i + 4];
    if (conf < CONFIDENCE_THRESHOLD) continue;
    const x1 = clamp((out[i] - meta.dx) / meta.scale, 0, meta.sw);
    const y1 = clamp((out[i + 1] - meta.dy) / meta.scale, 0, meta.sh);
    const x2 = clamp((out[i + 2] - meta.dx) / meta.scale, 0, meta.sw);
    const y2 = clamp((out[i + 3] - meta.dy) / meta.scale, 0, meta.sh);
    if (x2 - x1 < 2 || y2 - y1 < 2) continue;
    const cls = Math.round(out[i + 5]);
    detections.push({ x: x1, y: y1, w: x2 - x1, h: y2 - y1, conf, cls, label: CLASS_NAMES[cls] || 'object ' + cls });
  }
  return detections;
}

/* ------------------------------------------------------------------
   STEP 2: RUN. Detections go through one queue so two clicks never run at once.
   ------------------------------------------------------------------ */
let inferenceQueue = Promise.resolve();

function predictWithModel(source) {
  const task = inferenceQueue.then(async () => {
    const { session, inputName, outputName } = await getModel();
    const startedAt = performance.now();
    const meta = preprocess(source);
    const tensor = new ort.Tensor('float32', inputBuffer, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const results = await session.run({ [inputName]: tensor });
    const detections = postprocess(results[outputName].data, meta);
    return { detections, ms: performance.now() - startedAt };
  });
  inferenceQueue = task.then(() => undefined, () => undefined);
  return task;
}

/* ------------------------------------------------------------------
   DRAWING
   ------------------------------------------------------------------ */
const colorFor = cls => `hsl(${(cls * 47) % 360} 85% 58%)`;

function drawDetections(ctx, detections, w, h) {
  const lw = Math.max(2, Math.round(Math.min(w, h) / 320));
  const fs = Math.max(12, Math.round(Math.min(w, h) / 36));
  ctx.lineWidth = lw;
  ctx.font = `700 ${fs}px Inter, system-ui, sans-serif`;
  ctx.textBaseline = 'top';
  detections.forEach(d => {
    const color = colorFor(d.cls);
    ctx.strokeStyle = color;
    ctx.strokeRect(d.x, d.y, d.w, d.h);
    const text = `${d.label} ${Math.round(d.conf * 100)}%`;
    const tw = ctx.measureText(text).width + fs * 0.6, th = fs * 1.35;
    const ty = d.y - th >= 0 ? d.y - th : d.y;
    ctx.fillStyle = color;
    ctx.fillRect(d.x - lw / 2, ty, tw, th);
    ctx.fillStyle = '#06121a';
    ctx.fillText(text, d.x + fs * 0.3, ty + fs * 0.17);
  });
}

function summarize(detections) {
  if (!detections.length) return 'No objects found.';
  const counts = {};
  detections.forEach(d => { counts[d.label] = (counts[d.label] || 0) + 1; });
  const parts = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${v} ${k}`);
  return `${detections.length} object${detections.length > 1 ? 's' : ''}: ${parts.join(', ')}`;
}

// Size a video box to the real video shape so the boxes line up exactly (no letterbox gaps).
function fitStage(stage, vw, vh) {
  if (!stage || !vw || !vh) return;
  stage.style.aspectRatio = `${vw} / ${vh}`;
  stage.style.width = `min(100%, calc(72vh * ${vw / vh}))`;
  stage.style.marginInline = 'auto';
}

function sizeCanvas(canvas, w, h) {
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
}

/* ------------------------------------------------------------------
   PAGE WIRING
   ------------------------------------------------------------------ */
function init() {
  const $ = id => document.getElementById(id);
  const setText = (el, msg) => { if (el) el.textContent = msg; };
  const errMsg = e => (e && e.message) ? e.message : 'Something went wrong.';
  const modelNote = () => (modelPromise ? '' : 'Loading the model (about 10 MB, first time only)... ');

  /* ---------------- IMAGE ---------------- */
  const imageInput = $('imageInput'), captureInput = $('captureImageInput');
  const previewContainer = $('previewContainer'), resultCanvas = $('resultCanvas'), imagePreview = $('imagePreview');
  const detectButton = $('detectButton'), resetButton = $('resetButton');
  const fileNameEl = $('fileName'), detectionStatus = $('detectionStatus');
  const srcCanvas = document.createElement('canvas');     // untouched copy of the photo
  let imageBusy = false;

  async function loadImageFile(file) {
    if (!file) return;
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      const k = Math.min(1, MAX_IMAGE_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
      const w = Math.round(img.naturalWidth * k), h = Math.round(img.naturalHeight * k);
      srcCanvas.width = w; srcCanvas.height = h;
      srcCanvas.getContext('2d').drawImage(img, 0, 0, w, h);
      resultCanvas.width = w; resultCanvas.height = h;
      resultCanvas.getContext('2d').drawImage(srcCanvas, 0, 0);
      imagePreview.hidden = true;            // the canvas shows the photo, so the extra <img> stays hidden
      resultCanvas.hidden = false;
      previewContainer.hidden = false;
      fileNameEl.textContent = file.name || 'Camera picture';
      detectButton.disabled = false; resetButton.disabled = false;
      setText(detectionStatus, 'Image ready. Press "Detect objects".');
    } catch (e) {
      console.error(e);
      setText(detectionStatus, 'Could not open that image. Try a JPG or PNG.');
      previewContainer.hidden = false;
    } finally { URL.revokeObjectURL(url); }
  }
  [imageInput, captureInput].forEach(input => input && input.addEventListener('change', async ev => {
    await loadImageFile(ev.target.files && ev.target.files[0]);
    ev.target.value = '';                    // allows choosing the same file again
  }));

  detectButton.addEventListener('click', async () => {
    if (imageBusy) return;
    imageBusy = true; detectButton.disabled = true;
    setText(detectionStatus, modelNote() + 'Running detection...');
    try {
      const { detections, ms } = await predictWithModel(srcCanvas);
      const ctx = resultCanvas.getContext('2d');
      ctx.drawImage(srcCanvas, 0, 0);        // redraw the clean photo first, so boxes never pile up
      drawDetections(ctx, detections, resultCanvas.width, resultCanvas.height);
      setText(detectionStatus, `${summarize(detections)} (${Math.round(ms)} ms)`);
    } catch (e) {
      console.error(e);
      setText(detectionStatus, 'Detection failed: ' + errMsg(e));
    } finally { imageBusy = false; detectButton.disabled = false; }
  });

  resetButton.addEventListener('click', () => {
    resultCanvas.getContext('2d').clearRect(0, 0, resultCanvas.width, resultCanvas.height);
    resultCanvas.hidden = true; imagePreview.hidden = true; previewContainer.hidden = true;
    detectButton.disabled = true; resetButton.disabled = true;
    fileNameEl.textContent = '';
    setText(detectionStatus, 'Ready.');
  });

  /* ---------------- LIVE CAMERA ---------------- */
  const camStartBtn = $('cameraStartButton'), camStopBtn = $('cameraStopButton'), camShotBtn = $('cameraCaptureButton');
  const camVideo = $('cameraPreview'), camCanvas = $('cameraResultCanvas'), camStage = $('cameraStage');
  const camWorkspace = $('cameraWorkspace'), camStartPanel = $('cameraStartPanel');
  const camPermission = $('cameraPermissionStatus'), camStatus = $('cameraDetectionStatus');
  let camStream = null, camRunning = false, camBusy = false, camRaf = 0, camLastDone = 0, camFps = 0, camLast = [];

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setText(camPermission, 'This browser cannot use the camera. Open the site over https.');
      return;
    }
    camStartBtn.disabled = true;
    setText(camPermission, 'Requesting camera...');
    try {
      camStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false
      });
    } catch (e) {
      console.error(e);
      setText(camPermission, e && e.name === 'NotAllowedError' ? 'Camera permission was denied. Allow it in your browser settings.' : 'No camera available: ' + errMsg(e));
      camStartBtn.disabled = false;
      return;
    }
    camVideo.srcObject = camStream;
    camVideo.hidden = false; camCanvas.hidden = false;
    camWorkspace.hidden = false; camStartPanel.hidden = true;
    camStopBtn.disabled = false; camStartBtn.disabled = false;
    const fit = () => fitStage(camStage, camVideo.videoWidth, camVideo.videoHeight);
    camVideo.onloadedmetadata = fit;
    try { await camVideo.play(); } catch (e) { /* autoplay is allowed for muted video */ }
    fit();
    setText(camStatus, modelNote() + 'Starting live detection...');
    camRunning = true; camLastDone = 0; camFps = 0;
    try { await getModel(); }
    catch (e) { setText(camStatus, 'Could not load the model: ' + errMsg(e)); camRunning = false; return; }
    camRaf = requestAnimationFrame(camTick);
  }

  async function camTick() {
    if (!camRunning) return;
    if (!camBusy && camVideo.readyState >= 2 && camVideo.videoWidth) {
      camBusy = true;
      try {
        const { detections, ms } = await predictWithModel(camVideo);
        if (!camRunning) return;
        const w = camVideo.videoWidth, h = camVideo.videoHeight;
        sizeCanvas(camCanvas, w, h);
        const ctx = camCanvas.getContext('2d');
        ctx.clearRect(0, 0, w, h);
        drawDetections(ctx, detections, w, h);
        camLast = detections;
        const now = performance.now();
        if (camLastDone) { const f = 1000 / (now - camLastDone); camFps = camFps ? camFps * 0.8 + f * 0.2 : f; }
        camLastDone = now;
        setText(camStatus, `${summarize(detections)} · ${Math.round(ms)} ms · ${camFps.toFixed(1)} fps`);
      } catch (e) {
        console.error(e);
        setText(camStatus, 'Detection stopped: ' + errMsg(e));
        camRunning = false;
      } finally { camBusy = false; }
    }
    if (camRunning) camRaf = requestAnimationFrame(camTick);
  }

  function stopCamera() {
    camRunning = false; cancelAnimationFrame(camRaf);
    if (camStream) { camStream.getTracks().forEach(t => t.stop()); camStream = null; }
    camVideo.srcObject = null; camVideo.hidden = true;
    camCanvas.getContext('2d').clearRect(0, 0, camCanvas.width, camCanvas.height); camCanvas.hidden = true;
    camWorkspace.hidden = true; camStartPanel.hidden = false;
    camStopBtn.disabled = true; camLast = [];
    setText(camPermission, 'Camera off.');
  }

  // "Take picture": saves the current frame with its detection boxes as a PNG
  function captureFromCamera() {
    if (!camStream || camVideo.readyState < 2) { setText(camStatus, 'Camera not ready yet.'); return; }
    const w = camVideo.videoWidth, h = camVideo.videoHeight;
    const shot = document.createElement('canvas');
    shot.width = w; shot.height = h;
    const ctx = shot.getContext('2d');
    ctx.drawImage(camVideo, 0, 0, w, h);
    drawDetections(ctx, camLast, w, h);
    shot.toBlob(blob => {
      if (!blob) { setText(camStatus, 'Could not save the picture.'); return; }
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'viddit-' + new Date().toISOString().replace(/[:.]/g, '-') + '.png';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 1500);
      setText(camStatus, 'Picture saved to your downloads.');
    }, 'image/png');
  }
  camStartBtn.addEventListener('click', startCamera);
  camStopBtn.addEventListener('click', stopCamera);
  camShotBtn.addEventListener('click', captureFromCamera);

  /* ---------------- VIDEO FILE ---------------- */
  const vidInput = $('videoInput'), vidWorkspace = $('videoWorkspace'), vidName = $('videoFileName');
  const vidVideo = $('videoPreview'), vidCanvas = $('videoResultCanvas'), vidStage = $('videoStage');
  const vidStatus = $('videoDetectionStatus');
  const vidStartBtn = $('videoDetectButton'), vidStopBtn = $('videoStopButton'), vidResetBtn = $('videoResetButton');
  let vidURL = null, vidRunning = false, vidBusy = false, vidRaf = 0, vidLastTime = -1, vidLastDone = 0, vidFps = 0;

  const clearVidCanvas = () => vidCanvas.getContext('2d').clearRect(0, 0, vidCanvas.width, vidCanvas.height);

  vidInput.addEventListener('change', ev => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    stopVideoDetection();
    if (vidURL) URL.revokeObjectURL(vidURL);
    vidURL = URL.createObjectURL(file);
    vidVideo.src = vidURL;
    vidName.textContent = file.name;
    vidWorkspace.hidden = false; vidCanvas.hidden = false;
    vidStartBtn.disabled = true; vidResetBtn.disabled = false; vidStopBtn.disabled = true;
    setText(vidStatus, 'Loading video...');
    ev.target.value = '';
  });
  vidVideo.addEventListener('loadedmetadata', () => {
    fitStage(vidStage, vidVideo.videoWidth, vidVideo.videoHeight);
    vidStartBtn.disabled = false;
    setText(vidStatus, 'Video ready. Press "Start detection".');
  });
  vidVideo.addEventListener('error', () => {
    if (vidVideo.getAttribute('src')) setText(vidStatus, 'This browser cannot play that video format. Try MP4 (H.264) or WebM.');
  });
  vidVideo.addEventListener('seeking', clearVidCanvas);
  vidVideo.addEventListener('ended', () => { if (vidRunning) { stopVideoDetection(); setText(vidStatus, 'Video finished.'); } });

  async function startVideoDetection() {
    if (vidRunning) return;
    vidRunning = true; vidLastTime = -1; vidLastDone = 0; vidFps = 0;
    vidStartBtn.disabled = true; vidStopBtn.disabled = false;
    setText(vidStatus, modelNote() + 'Starting detection...');
    try { await getModel(); }
    catch (e) { setText(vidStatus, 'Could not load the model: ' + errMsg(e)); stopVideoDetection(); return; }
    if (!vidRunning) return;
    if (vidVideo.ended) vidVideo.currentTime = 0;
    try { await vidVideo.play(); } catch (e) { /* the user can press play */ }
    vidRaf = requestAnimationFrame(vidTick);
  }

  async function vidTick() {
    if (!vidRunning) return;
    if (!vidBusy && vidVideo.readyState >= 2 && vidVideo.currentTime !== vidLastTime) {
      vidBusy = true; vidLastTime = vidVideo.currentTime;
      try {
        const { detections, ms } = await predictWithModel(vidVideo);
        if (!vidRunning) return;
        const w = vidVideo.videoWidth, h = vidVideo.videoHeight;
        sizeCanvas(vidCanvas, w, h);
        const ctx = vidCanvas.getContext('2d');
        ctx.clearRect(0, 0, w, h);
        drawDetections(ctx, detections, w, h);
        const now = performance.now();
        if (vidLastDone) { const f = 1000 / (now - vidLastDone); vidFps = vidFps ? vidFps * 0.8 + f * 0.2 : f; }
        vidLastDone = now;
        // The model is slower than normal playback, so slow the video down to keep the boxes in step with it
        vidVideo.playbackRate = Math.min(1, Math.max(0.1, 300 / Math.max(ms, 1)));
        setText(vidStatus, `${summarize(detections)} · ${Math.round(ms)} ms · ${vidFps.toFixed(1)} fps`);
      } catch (e) {
        console.error(e);
        setText(vidStatus, 'Detection stopped: ' + errMsg(e));
        stopVideoDetection();
        return;
      } finally { vidBusy = false; }
    }
    if (vidRunning) vidRaf = requestAnimationFrame(vidTick);
  }

  function stopVideoDetection() {
    vidRunning = false; cancelAnimationFrame(vidRaf);
    vidVideo.playbackRate = 1;
    clearVidCanvas();
    vidStopBtn.disabled = true;
    vidStartBtn.disabled = !(vidVideo.getAttribute('src') && vidVideo.readyState >= 1);
  }

  vidStartBtn.addEventListener('click', startVideoDetection);
  vidStopBtn.addEventListener('click', () => { stopVideoDetection(); setText(vidStatus, 'Detection stopped.'); });
  vidResetBtn.addEventListener('click', () => {
    stopVideoDetection();
    vidVideo.pause(); vidVideo.removeAttribute('src'); vidVideo.load();
    if (vidURL) { URL.revokeObjectURL(vidURL); vidURL = null; }
    vidWorkspace.hidden = true; vidCanvas.hidden = true; vidName.textContent = '';
    vidStartBtn.disabled = true; vidResetBtn.disabled = true;
    setText(vidStatus, 'Video ready.');
  });

  // Leaving a workspace frees the camera and stops the work
  document.addEventListener('viddit:workspace-change', ev => {
    const w = ev.detail && ev.detail.workspace;
    if (w !== 'camera' && camStream) stopCamera();
    if (w !== 'video' && vidRunning) { stopVideoDetection(); vidVideo.pause(); setText(vidStatus, 'Detection stopped.'); }
  });
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
