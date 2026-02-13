import asyncio
import json
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

app = FastAPI()

# Load a fast general-purpose detector (COCO 80 classes)
# Options: yolov8n.pt (fastest), yolov8s.pt (better), yolov8m.pt (heavier)
model = YOLO("yolov8n.pt")

# Optional: if you have GPU + CUDA, Ultralytics will use it automatically.
# model.to("cuda")  # Only if your setup supports it.


def decode_jpeg_bytes_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes into an OpenCV BGR image."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image (cv2.imdecode returned None).")
    return img


def yolo_infer(img_bgr: np.ndarray) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Run YOLO inference and return detections in normalized coords (0..1).
    Each detection: label, conf, x, y, w, h where x,y is top-left.
    """
    h, w = img_bgr.shape[:2]

    # Ultralytics expects RGB for best correctness, but works with BGR too.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Reduce overhead: single-image inference, no saving, no verbose.
    results = model.predict(img_rgb, imgsz=640, conf=0.25, iou=0.45, verbose=False)

    detections: List[Dict[str, Any]] = []
    r = results[0]
    if r.boxes is None:
        return w, h, detections

    boxes = r.boxes
    names = model.names  # class_id -> name

    for b in boxes:
        # xyxy in pixels
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf = float(b.conf[0].item()) if b.conf is not None else 0.0
        cls_id = int(b.cls[0].item()) if b.cls is not None else -1
        label = names.get(cls_id, str(cls_id))

        # Clamp + normalize
        x1 = max(0.0, min(x1, w - 1))
        y1 = max(0.0, min(y1, h - 1))
        x2 = max(0.0, min(x2, w - 1))
        y2 = max(0.0, min(y2, h - 1))

        nx = x1 / w
        ny = y1 / h
        nw = (x2 - x1) / w
        nh = (y2 - y1) / h

        detections.append(
            {
                "label": label,
                "conf": round(conf, 4),
                "x": round(nx, 6),
                "y": round(ny, 6),
                "w": round(nw, 6),
                "h": round(nh, 6),
            }
        )

    return w, h, detections


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WS client connected")

    try:
        while True:
            # Receive JPEG bytes from client
            msg = await websocket.receive()

            if "bytes" in msg and msg["bytes"] is not None:
                jpeg_bytes = msg["bytes"]
            elif "text" in msg and msg["text"] is not None:
                # (Optional fallback) if you ever decide to send base64 text
                # You can implement base64 decode here if needed.
                await websocket.send_text(json.dumps({"error": "Text frames not supported in this MVP."}))
                continue
            else:
                continue

            t0 = time.time()

            try:
                img = decode_jpeg_bytes_to_bgr(jpeg_bytes)
                width, height, detections = yolo_infer(img)

                payload = {
                    "ts": time.time(),
                    "width": width,
                    "height": height,
                    "ms": int((time.time() - t0) * 1000),
                    "detections": detections,
                }

                await websocket.send_text(json.dumps(payload))
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))

            # tiny yield so one client can't starve event loop
            await asyncio.sleep(0)

    except WebSocketDisconnect:
        print("WS client disconnected")
    except Exception as e:
        print("WS error:", e)
