"""Drive a single JPEG through ws://localhost:8000/ws and print the response.

Usage: python smoke_ws.py [image_url_or_path]

Defaults to a COCO val image with a person + TV.
"""

import asyncio
import json
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image
import websockets

DEFAULT_URLS = [
    # COCO val2017 — person + tv
    "http://images.cocodataset.org/val2017/000000000139.jpg",
    # Florence-2 model card sample (a car)
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
]


def load_image_bytes(arg: str | None) -> bytes:
    if arg and Path(arg).exists():
        return Path(arg).read_bytes()
    candidates = [arg] if arg else DEFAULT_URLS
    last_err = None
    for url in candidates:
        try:
            print(f"[client] fetching {url}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "smoke-test/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            img = Image.open(BytesIO(data))
            img.load()
            buf = BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except Exception as e:
            last_err = e
            print(f"[client] fetch failed: {e}", flush=True)
    raise RuntimeError(f"All image sources failed; last error: {last_err}")


async def main(arg: str | None) -> None:
    jpeg = load_image_bytes(arg)
    print(f"[client] image bytes: {len(jpeg)}", flush=True)

    async with websockets.connect("ws://127.0.0.1:8000/ws", max_size=None) as ws:
        print("[client] connected", flush=True)
        await ws.send(jpeg)
        print("[client] sent frame, waiting for reply...", flush=True)
        t0 = time.time()
        reply = await asyncio.wait_for(ws.recv(), timeout=180)
        elapsed = time.time() - t0
        print(f"[client] reply in {elapsed:.1f}s", flush=True)

        try:
            payload = json.loads(reply)
        except Exception:
            print("[client] non-JSON reply:", reply[:500], flush=True)
            return

        if "error" in payload:
            print(f"[client] server error: {payload['error']}", flush=True)
            return

        dets = payload.get("detections", [])
        print(
            f"[client] frame {payload.get('width')}x{payload.get('height')} "
            f"ms={payload.get('ms')} count={len(dets)}",
            flush=True,
        )
        for d in dets:
            print(
                f"  #{d['track_id']:>3}  {d['label']:<15} type={d['type']:<7} "
                f"conf={d['conf']}  bbox=({d['x']:.3f},{d['y']:.3f},{d['w']:.3f},{d['h']:.3f})  "
                f"dist={d['distance_m']}m angle={d['angle_deg']}",
                flush=True,
            )


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(arg))
