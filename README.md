# Viddit
A browser-based computer vision platform for object detection, motion detection, and object tracking

## What works now
- **Image**: choose a photo or take one, then detect objects.
- **Video**: choose a video file and detect objects frame by frame (playback slows to match the model's speed).
- **Live camera**: real-time detection with a frames-per-second readout. "Take picture" saves the current frame with its boxes.

Everything runs in the browser with the YOLO26n model (80 everyday object classes, such as person, car, dog, cup, laptop, phone). No pictures are uploaded anywhere.

## Files
`index.html`, `style.css`, `app.js`, `yolo26n.onnx` (all in the same folder, repo root).

## Running it
Open it through GitHub Pages or a local server. Double-clicking `index.html` will not work, because browsers block the model file and the camera on `file://` pages.

Local server: `python -m http.server 8000`, then open `http://localhost:8000`.

## Settings
Top of `app.js`: `CONFIDENCE_THRESHOLD` (default 0.35). Raise it for fewer, surer boxes; lower it to see more.

## Notes
- The model runs on the CPU through ONNX Runtime Web (single thread), so speed depends on the device.
- Videos must be in a format the browser can play (MP4 H.264 or WebM).

## Next
Confidence slider, motion detection, object tracking.
