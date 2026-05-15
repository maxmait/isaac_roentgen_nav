"""
take_snapshot.py

Subscribes to the ZMQ frame stream published by image_publisher.py (running
inside Isaac Sim) and saves one frame to disk.

Usage:
    python3 take_snapshot.py [output_filename.jpg]

Requires: pyzmq  (already available in Isaac Sim Python; also installable via pip)
"""

import os
import sys
import time

import zmq

SNAPSHOT_DIR = os.path.expanduser("~/isaac_projects/scenes/snapshots")
ZMQ_HOST     = "tcp://127.0.0.1:5556"
TIMEOUT_MS   = 5000   # wait up to 5 s for a frame

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

filename = sys.argv[1] if len(sys.argv) > 1 else f"snapshot_{int(time.time())}.jpg"
out_path = os.path.join(SNAPSHOT_DIR, filename)

ctx  = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)
sock.setsockopt(zmq.SUBSCRIBE, b"frame")
sock.connect(ZMQ_HOST)

print(f"Waiting for frame from {ZMQ_HOST} …")
try:
    topic, jpeg = sock.recv_multipart()
    with open(out_path, "wb") as f:
        f.write(jpeg)
    print(f"Saved: {out_path}  ({len(jpeg):,} bytes)")
except zmq.Again:
    print("Timeout — no frame received.")
    print("Make sure image_publisher.py has been injected into Isaac Sim.")
    sys.exit(1)
finally:
    sock.close()
    ctx.term()
