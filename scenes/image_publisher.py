"""
image_publisher.py

Captures the Isaac Sim viewport and publishes JPEG frames via ZMQ PUB.

Primary usage — inject into a running Isaac Sim GUI via the VSCode extension TCP socket:
    python3 isaacsim_client.py "$(cat image_publisher.py)"

Standalone usage (GUI mode, not headless):
    ~/isaacsim/python.sh ~/isaac_projects/scenes/image_publisher.py

Subscriber: connect a ZMQ SUB socket to tcp://127.0.0.1:5556
Frame format: multipart [b"frame", <JPEG bytes>]
Call take_snapshot() from inside Isaac Sim, or use take_snapshot.py externally.

Implementation note:
    rep.AnnotatorRegistry annotators attached to the viewport render product
    only yield data after rep.orchestrator.step_async() triggers the replicator
    pipeline. An async loop drives this at ~TARGET_FPS.
"""

import asyncio
import os
import sys
import threading

import numpy as np

# ── config ─────────────────────────────────────────────────────────────────────
ZMQ_PORT     = 5556
JPEG_QUALITY = 85
TARGET_FPS   = 10          # capture rate; reduce if GPU load is a concern
SNAPSHOT_DIR = os.path.expanduser("~/isaac_projects/scenes/snapshots")
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ── detect execution context ───────────────────────────────────────────────────
_injected = "omni.usd" in sys.modules

if not _injected:
    from isaacsim import SimulationApp
    _app = SimulationApp({"headless": False})
# ───────────────────────────────────────────────────────────────────────────────

import omni.kit.app
import omni.replicator.core as rep
from omni.kit.viewport.utility import get_active_viewport

# ── ZMQ publisher ──────────────────────────────────────────────────────────────
import zmq as _zmq

_zmq_ctx    = _zmq.Context.instance()
_zmq_socket = _zmq_ctx.socket(_zmq.PUB)
_zmq_socket.setsockopt(_zmq.SNDHWM, 2)      # drop stale frames under backpressure
_zmq_socket.bind(f"tcp://127.0.0.1:{ZMQ_PORT}")
# ───────────────────────────────────────────────────────────────────────────────

# ── attach annotator to the EXISTING viewport render product ───────────────────
# Use the live viewport render product (not a new rep.create.render_product)
# so frames reflect what the user sees in the GUI.
_viewport = get_active_viewport()
_rp_path  = _viewport.render_product_path

_rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb_ann.attach([_rp_path])
# ───────────────────────────────────────────────────────────────────────────────

_latest_jpeg  = None
_latest_lock  = threading.Lock()
_publisher_running = False


def _encode_jpeg(rgba: np.ndarray) -> bytes:
    rgb = rgba[:, :, :3]
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1],
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            return bytes(buf)
    except ImportError:
        pass
    from PIL import Image
    import io
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


async def _capture_loop():
    """Async loop: step the replicator pipeline, grab the annotator data, publish."""
    global _publisher_running
    _publisher_running = True
    interval = 1.0 / TARGET_FPS
    print(f"[image_publisher] Capture loop started at {TARGET_FPS} FPS")
    while _publisher_running:
        await rep.orchestrator.step_async(rt_subframes=2, pause_timeline=False)
        data = _rgb_ann.get_data()
        if data is not None and data.size > 0:
            jpeg = _encode_jpeg(data)
            with _latest_lock:
                global _latest_jpeg
                _latest_jpeg = jpeg
            _zmq_socket.send_multipart([b"frame", jpeg])
        await asyncio.sleep(interval)


def stop_publisher():
    """Call this from Isaac Sim (via TCP) to stop the capture loop."""
    global _publisher_running
    _publisher_running = False
    print("[image_publisher] Stopped.")


def take_snapshot(filename: str = None) -> str:
    """
    Save the most-recently published frame to SNAPSHOT_DIR.
    Returns the full path of the saved file.
    """
    with _latest_lock:
        jpeg = _latest_jpeg
    if jpeg is None:
        raise RuntimeError("No frame captured yet — wait a moment and retry.")
    if filename is None:
        import time
        filename = f"snapshot_{int(time.time())}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    with open(path, "wb") as f:
        f.write(jpeg)
    print(f"[image_publisher] Snapshot saved: {path}")
    return path


# ── start the capture loop on Isaac Sim's asyncio event loop ───────────────────
asyncio.ensure_future(_capture_loop())
# ───────────────────────────────────────────────────────────────────────────────

print(f"[image_publisher] Started — ZMQ PUB on tcp://127.0.0.1:{ZMQ_PORT}")
print(f"[image_publisher] Viewport render product: {_rp_path}")
print(f"[image_publisher] Target FPS: {TARGET_FPS}  |  Snapshots dir: {SNAPSHOT_DIR}")
print(f"[image_publisher] stop_publisher() to stop  |  take_snapshot() to save a frame")

# ── standalone loop ────────────────────────────────────────────────────────────
if not _injected:
    from isaacsim.core.api import World
    _world = World()
    _world.scene.add_default_ground_plane()
    _world.reset()

    _step = 0
    while _app.is_running():
        _world.step(render=True)
        _step += 1
        if _step % 300 == 0:
            print(f"[image_publisher] step={_step}")

    _app.close()
