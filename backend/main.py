import asyncio
import json
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

app = FastAPI()

# Load a fast general-purpose detector (COCO 80 classes)
# Options: yolov8n.pt (fastest), yolov8s.pt (better), yolov8m.pt (heavier)
model = YOLO("yolov8n.pt")
model_lock = threading.Lock()

# Lightweight face detector for face-priority targeting.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Optional: if you have GPU + CUDA, Ultralytics will use it automatically.
# model.to("cuda")  # Only if your setup supports it.

HORIZONTAL_FOV_DEG = 68.0
MAX_TRACK_MISSES = 12
TRACK_IOU_THRESHOLD = 0.30

OBJECT_META: Dict[str, Dict[str, Any]] = {
    "face": {"category": "biometric", "uses": ["identity", "attention", "access control"]},
    "person": {"category": "human", "uses": ["safety monitoring", "interaction", "tracking"]},
    "car": {"category": "vehicle", "uses": ["navigation", "traffic analysis", "fleet monitoring"]},
    "truck": {"category": "vehicle", "uses": ["logistics", "fleet monitoring", "risk detection"]},
    "bus": {"category": "vehicle", "uses": ["transit monitoring", "capacity analysis"]},
    "bicycle": {"category": "vehicle", "uses": ["micromobility", "safety monitoring"]},
    "motorcycle": {"category": "vehicle", "uses": ["traffic analysis", "safety monitoring"]},
    "dog": {"category": "animal", "uses": ["pet monitoring", "wildlife screening"]},
    "cat": {"category": "animal", "uses": ["pet monitoring", "wildlife screening"]},
    "bottle": {"category": "object", "uses": ["inventory", "quality checks"]},
    "cell phone": {"category": "device", "uses": ["device detection", "policy enforcement"]},
    "laptop": {"category": "device", "uses": ["asset monitoring", "workspace analytics"]},
}

REAL_WORLD_WIDTH_M: Dict[str, float] = {
    "face": 0.16,
    "person": 0.45,
    "car": 1.80,
    "truck": 2.50,
    "bus": 2.55,
    "bicycle": 0.55,
    "motorcycle": 0.75,
    "dog": 0.30,
    "cat": 0.22,
    "bottle": 0.07,
    "cell phone": 0.07,
    "laptop": 0.33,
}


def decode_jpeg_bytes_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes into an OpenCV BGR image."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image (cv2.imdecode returned None).")
    return img


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0
    return inter_area / union


def focal_length_px(image_width_px: int) -> float:
    half_fov_rad = math.radians(HORIZONTAL_FOV_DEG / 2.0)
    return (image_width_px / 2.0) / math.tan(half_fov_rad)


def estimate_distance_m(pixel_width: float, image_width: int, label: str) -> Optional[float]:
    if pixel_width <= 1.0:
        return None

    real_width = REAL_WORLD_WIDTH_M.get(label, 0.50)
    fpx = focal_length_px(image_width)
    distance = (real_width * fpx) / pixel_width
    return round(float(distance), 2)


def estimate_angle_deg(cx_px: float, image_width: int) -> float:
    fpx = focal_length_px(image_width)
    offset = cx_px - (image_width / 2.0)
    angle = math.degrees(math.atan2(offset, fpx))
    return round(float(angle), 2)


def raw_yolo_detections(img_bgr: np.ndarray) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Run YOLO inference and return detections in pixel-space xyxy."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with model_lock:
        # Keep all classes active and use a slightly lower threshold so non-human
        # objects still surface in typical webcam scenes.
        results = model.predict(img_rgb, imgsz=960, conf=0.20, iou=0.45, verbose=False)

    detections: List[Dict[str, Any]] = []
    r = results[0]
    if r.boxes is None:
        return w, h, detections

    names = model.names

    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf = float(b.conf[0].item()) if b.conf is not None else 0.0
        cls_id = int(b.cls[0].item()) if b.cls is not None else -1
        label = names.get(cls_id, str(cls_id))

        x1 = max(0.0, min(x1, w - 1))
        y1 = max(0.0, min(y1, h - 1))
        x2 = max(0.0, min(x2, w - 1))
        y2 = max(0.0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "label": label,
                "type": "object",
                "conf": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    return w, h, detections


def raw_face_detections(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Run Haar cascade face detection and return pixel-space xyxy boxes."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(36, 36),
    )

    detections: List[Dict[str, Any]] = []
    for x, y, w, h in faces:
        detections.append(
            {
                "label": "face",
                "type": "face",
                "conf": 0.95,
                "x1": float(x),
                "y1": float(y),
                "x2": float(x + w),
                "y2": float(y + h),
            }
        )

    return detections


class SimpleTracker:
    def __init__(self) -> None:
        self.next_id = 1
        self.tracks: List[Dict[str, Any]] = []

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for track in self.tracks:
            track["misses"] += 1

        for det in detections:
            det_bbox = (det["x1"], det["y1"], det["x2"], det["y2"])
            best_track: Optional[Dict[str, Any]] = None
            best_score = 0.0

            for track in self.tracks:
                score = iou_xyxy(det_bbox, track["bbox"])
                if det["label"] != track["label"]:
                    score *= 0.7
                if score > best_score:
                    best_score = score
                    best_track = track

            if best_track and best_score >= TRACK_IOU_THRESHOLD:
                best_track["bbox"] = det_bbox
                best_track["label"] = det["label"]
                best_track["type"] = det["type"]
                best_track["misses"] = 0
                det["track_id"] = best_track["id"]
            else:
                new_track = {
                    "id": self.next_id,
                    "bbox": det_bbox,
                    "label": det["label"],
                    "type": det["type"],
                    "misses": 0,
                }
                self.tracks.append(new_track)
                det["track_id"] = self.next_id
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["misses"] <= MAX_TRACK_MISSES]
        return detections


def enrich_and_normalize(
    detections: List[Dict[str, Any]], image_width: int, image_height: int
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        cx = x1 + bbox_w / 2.0

        distance_m = estimate_distance_m(bbox_w, image_width, det["label"])
        angle_deg = estimate_angle_deg(cx, image_width)

        base_priority = 3.0 if det["label"] == "face" else 2.0 if det["label"] == "person" else 1.0
        priority = base_priority + min(1.0, det["conf"])

        meta = OBJECT_META.get(det["label"], {"category": "object", "uses": ["scene awareness"]})

        out.append(
            {
                "track_id": det.get("track_id", -1),
                "label": det["label"],
                "type": det["type"],
                "category": meta["category"],
                "uses": meta["uses"],
                "conf": round(float(det["conf"]), 4),
                "priority": round(priority, 3),
                "distance_m": distance_m,
                "angle_deg": angle_deg,
                "x": round(x1 / image_width, 6),
                "y": round(y1 / image_height, 6),
                "w": round(bbox_w / image_width, 6),
                "h": round(bbox_h / image_height, 6),
            }
        )

    out.sort(key=lambda d: (d["priority"], d["conf"]), reverse=True)
    return out


def process_jpeg_frame(jpeg_bytes: bytes, tracker: SimpleTracker) -> Dict[str, Any]:
    t0 = time.time()
    img = decode_jpeg_bytes_to_bgr(jpeg_bytes)

    width, height, yolo_dets = raw_yolo_detections(img)
    face_dets = raw_face_detections(img)

    tracked = tracker.update(yolo_dets + face_dets)
    detections = enrich_and_normalize(tracked, width, height)

    primary = detections[0]["track_id"] if detections else None

    return {
        "ts": time.time(),
        "width": width,
        "height": height,
        "ms": int((time.time() - t0) * 1000),
        "primary_target_id": primary,
        "detections": detections,
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WS client connected")
    tracker = SimpleTracker()

    try:
        while True:
            msg = await websocket.receive()

            if "bytes" in msg and msg["bytes"] is not None:
                jpeg_bytes = msg["bytes"]
            elif "text" in msg and msg["text"] is not None:
                await websocket.send_text(
                    json.dumps({"error": "Text frames not supported in this MVP."})
                )
                continue
            else:
                continue

            try:
                payload = await asyncio.to_thread(process_jpeg_frame, jpeg_bytes, tracker)
                await websocket.send_text(json.dumps(payload))
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))

            await asyncio.sleep(0)

    except WebSocketDisconnect:
        print("WS client disconnected")
    except Exception as e:
        print("WS error:", e)
