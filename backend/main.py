import asyncio
import json
import math
import os
import threading
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

app = FastAPI()

cors_origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "")
cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
if not cors_origins:
    cors_origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FLORENCE_MODEL_ID = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-large")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Loading {FLORENCE_MODEL_ID} on {DEVICE} (dtype={TORCH_DTYPE})...")
florence_model = (
    AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=TORCH_DTYPE,
        attn_implementation="eager",
    )
    .to(DEVICE)
    .eval()
)
florence_processor = AutoProcessor.from_pretrained(
    FLORENCE_MODEL_ID,
    trust_remote_code=True,
)
model_lock = threading.Lock()
print("Florence-2 ready.")

HORIZONTAL_FOV_DEG = 68.0
MAX_TRACK_MISSES = 12
TRACK_IOU_THRESHOLD = 0.30
DEDUP_IOU_THRESHOLD = 0.55

OD_PROMPT = "<OD>"
GROUNDING_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"
GROUNDING_TEXT = "a human face"

# Florence-2 emits free-form natural-language labels. Collapse the common
# variants down to the canonical names that OBJECT_META / REAL_WORLD_WIDTH_M
# and (critically) the frontend branch on.
LABEL_NORMALIZATION: Dict[str, str] = {
    "human face": "face",
    "face": "face",
    "head": "face",
    "human": "person",
    "person": "person",
    "people": "person",
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "child": "person",
    "pedestrian": "person",
    "automobile": "car",
    "vehicle": "car",
    "car": "car",
    "sedan": "car",
    "suv": "car",
    "truck": "truck",
    "pickup truck": "truck",
    "bus": "bus",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "scooter": "motorcycle",
    "dog": "dog",
    "puppy": "dog",
    "cat": "cat",
    "kitten": "cat",
    "bottle": "bottle",
    "water bottle": "bottle",
    "cell phone": "cell phone",
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "phone": "cell phone",
    "smartphone": "cell phone",
    "laptop": "laptop",
    "notebook computer": "laptop",
}

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


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


def decode_jpeg_bytes_to_pil(jpeg_bytes: bytes) -> Image.Image:
    """Decode JPEG bytes into a PIL RGB image."""
    img = Image.open(BytesIO(jpeg_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
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


def normalize_label(raw: str) -> Tuple[str, str]:
    """Map a Florence-2 free-form label to (canonical_label, detection_type)."""
    key = (raw or "").strip().lower()
    label = LABEL_NORMALIZATION.get(key, key or "object")
    det_type = "face" if label == "face" else "object"
    return label, det_type


def _run_florence_task(
    pil_image: Image.Image,
    task_prompt: str,
    text_input: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run one Florence-2 inference. Returns parsed result or None on failure."""
    prompt = task_prompt if text_input is None else f"{task_prompt}{text_input}"
    try:
        with model_lock:
            inputs = florence_processor(
                text=prompt, images=pil_image, return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(DEVICE)
            pixel_values = inputs["pixel_values"].to(DEVICE, dtype=TORCH_DTYPE)

            with torch.inference_mode():
                generated_ids = florence_model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )

            generated_text = florence_processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

        return florence_processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(pil_image.width, pil_image.height),
        )
    except Exception as exc:
        import traceback
        print(f"Florence-2 inference failed for {task_prompt!r}: {exc}", flush=True)
        traceback.print_exc()
        return None


def _clip_bbox(
    bbox: List[float], width: int, height: int
) -> Optional[Tuple[float, float, float, float]]:
    if len(bbox) < 4:
        return None
    x1, y1, x2, y2 = bbox[:4]
    x1 = max(0.0, min(float(x1), width - 1))
    y1 = max(0.0, min(float(y1), height - 1))
    x2 = max(0.0, min(float(x2), width - 1))
    y2 = max(0.0, min(float(y2), height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _suppress_duplicates(
    dets: List[Dict[str, Any]], iou_thresh: float = DEDUP_IOU_THRESHOLD
) -> List[Dict[str, Any]]:
    """Drop overlapping same-label boxes, keeping the higher-confidence one."""
    keep: List[Dict[str, Any]] = []
    for det in sorted(dets, key=lambda d: d["conf"], reverse=True):
        bbox = (det["x1"], det["y1"], det["x2"], det["y2"])
        duplicate = False
        for other in keep:
            if other["label"] != det["label"]:
                continue
            other_bbox = (other["x1"], other["y1"], other["x2"], other["y2"])
            if iou_xyxy(bbox, other_bbox) > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            keep.append(det)
    return keep


def florence_detect_all(
    pil_image: Image.Image,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Unified detection: <OD> for general objects + <CAPTION_TO_PHRASE_GROUNDING> for faces."""
    width, height = pil_image.width, pil_image.height
    detections: List[Dict[str, Any]] = []

    od_result = _run_florence_task(pil_image, OD_PROMPT)
    if od_result and OD_PROMPT in od_result:
        block = od_result[OD_PROMPT] or {}
        boxes = block.get("bboxes", []) or []
        labels = block.get("labels", []) or []
        print(f"[florence OD raw] {len(labels)} boxes, labels={labels}", flush=True)
        for raw_label, bbox in zip(labels, boxes):
            clipped = _clip_bbox(bbox, width, height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            label, det_type = normalize_label(str(raw_label))
            detections.append(
                {
                    "label": label,
                    "type": det_type,
                    "conf": 0.85,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    grounding_result = _run_florence_task(
        pil_image, GROUNDING_PROMPT, text_input=GROUNDING_TEXT
    )
    if grounding_result and GROUNDING_PROMPT in grounding_result:
        block = grounding_result[GROUNDING_PROMPT] or {}
        boxes = block.get("bboxes", []) or []
        grd_labels = block.get("labels", []) or []
        print(f"[florence grounding raw] {len(boxes)} boxes, labels={grd_labels}", flush=True)
        for bbox in boxes:
            clipped = _clip_bbox(bbox, width, height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            detections.append(
                {
                    "label": "face",
                    "type": "face",
                    "conf": 0.90,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    detections = _suppress_duplicates(detections)
    return width, height, detections


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
    img = decode_jpeg_bytes_to_pil(jpeg_bytes)

    width, height, raw_dets = florence_detect_all(img)

    tracked = tracker.update(raw_dets)
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
    print("WS connected:", websocket.client)
    tracker = SimpleTracker()

    try:
        while True:
            msg = await websocket.receive()

            if "bytes" in msg and msg["bytes"] is not None:
                jpeg_bytes = msg["bytes"]
                print("WS frame bytes:", len(jpeg_bytes))
            elif "text" in msg and msg["text"] is not None:
                await websocket.send_text(
                    json.dumps({"error": "Text frames not supported in this MVP."})
                )
                continue
            else:
                continue

            try:
                t0 = time.time()
                print("Inference start")
                payload = await asyncio.to_thread(process_jpeg_frame, jpeg_bytes, tracker)
                elapsed_ms = int((time.time() - t0) * 1000)
                print("Inference done ms:", elapsed_ms)
                await websocket.send_text(json.dumps(payload))
                print("Sent detections:", len(payload.get("detections", [])))
            except Exception as e:
                print("Frame processing error:", str(e))
                try:
                    await websocket.send_text(json.dumps({"error": str(e)}))
                except Exception as send_err:
                    print("Failed to forward error to client:", send_err)

            await asyncio.sleep(0)

    except WebSocketDisconnect:
        print("WS disconnected:", websocket.client)
    except Exception as e:
        print("WS error:", e)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
