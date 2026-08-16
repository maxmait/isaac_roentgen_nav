#!/usr/bin/env python3
"""Publish the recovered fluoroscopy-registration result as a ROS 2 TF tree + RViz markers.

Reads output/robot_to_anatomy.json (produced by compute_robot_to_anatomy.py /
register_oneshot.py) and broadcasts, rooted at the **anatomy** frame:

    anatomy ──► carm        (inverse of recovered T_anatomy_in_carm)
    anatomy ──► tool        (recovered T_robot_in_anatomy — the EE / TCP)

plus a MarkerArray on /fluoro_markers:
    * the CT anatomy surface mesh (output/spine_mesh.obj, in metres, at the
      anatomy origin),
    * the tool as an oriented shaft cylinder + tip sphere (in the tool frame),
    * the C-arm source + beam axis toward the isocenter,
    * a text panel with the registration error + clinical inside/outside check.

The JSON is re-read every tick, so re-running the registration updates RViz live
— the natural hook for a future real-time tracking loop.

Run (ROS 2 Humble must be sourced):
    source /opt/ros/humble/setup.bash
    python3 ros/fluoro_tf_bridge.py
or use ros/run_rviz.sh to launch this node together with rviz2.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

OUT = Path(os.path.expanduser("~/isaac_projects/output"))
RESULT_JSON = OUT / "robot_to_anatomy.json"
MESH_OBJ = OUT / "spine_mesh.obj"

ANATOMY = "anatomy"
CARM = "carm"
TOOL = "tool"

SID_M = 0.510   # source-to-isocenter (fluorosim geometry)


# ─── rigid-transform helpers (json quats are wxyz; ROS quats are xyzw) ─────────
def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    return np.array([
        [1 - s * (y * y + z * z),     s * (x * y - z * w),     s * (x * z + y * w)],
        [    s * (x * y + z * w), 1 - s * (x * x + z * z),     s * (y * z - x * w)],
        [    s * (x * z - y * w),     s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quat_xyzw(R: np.ndarray) -> list[float]:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def to_matrix4(t_mm, q_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_matrix(q_wxyz)
    T[:3, 3] = np.asarray(t_mm, dtype=np.float64) / 1000.0   # mm → m
    return T


def matrix4_to_t_q(T: np.ndarray):
    return T[:3, 3].tolist(), matrix_to_quat_xyzw(T[:3, :3])


class FluoroTfBridge(Node):
    def __init__(self):
        super().__init__("fluoro_tf_bridge")
        self.bc = TransformBroadcaster(self)
        self.markers_pub = self.create_publisher(MarkerArray, "fluoro_markers", 1)
        self.timer = self.create_timer(0.2, self.tick)   # 5 Hz
        self._warned = False
        self.get_logger().info(f"Reading {RESULT_JSON}")

    # ── TF ────────────────────────────────────────────────────────────────
    def _tf(self, parent: str, child: str, t_m, q_xyzw, stamp) -> TransformStamped:
        m = TransformStamped()
        m.header.stamp = stamp
        m.header.frame_id = parent
        m.child_frame_id = child
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = t_m
        (m.transform.rotation.x, m.transform.rotation.y,
         m.transform.rotation.z, m.transform.rotation.w) = q_xyzw
        return m

    def tick(self):
        if not RESULT_JSON.exists():
            if not self._warned:
                self.get_logger().warn(f"{RESULT_JSON} not found — run the "
                                       "registration first. Retrying...")
                self._warned = True
            return
        try:
            res = json.loads(RESULT_JSON.read_text())
            rec = res["recovered"]
        except (json.JSONDecodeError, KeyError):
            return  # mid-write or malformed; try again next tick

        stamp = self.get_clock().now().to_msg()

        # anatomy → tool  (recovered T_robot_in_anatomy)
        tra = rec["T_robot_in_anatomy"]
        T_tool = to_matrix4(tra["t_mm"], tra["R_quat_wxyz"])
        t_tool, q_tool = matrix4_to_t_q(T_tool)

        # anatomy → carm  = inverse of recovered T_anatomy_in_carm
        tac = rec["T_anatomy_in_carm"]
        T_anat_in_carm = to_matrix4(tac["t_mm"], tac["R_quat_wxyz"])
        T_carm_in_anat = np.linalg.inv(T_anat_in_carm)
        t_carm, q_carm = matrix4_to_t_q(T_carm_in_anat)

        self.bc.sendTransform([
            self._tf(ANATOMY, TOOL, t_tool, q_tool, stamp),
            self._tf(ANATOMY, CARM, t_carm, q_carm, stamp),
        ])
        self.markers_pub.publish(self._markers(res, t_carm, stamp))

    # ── Markers ───────────────────────────────────────────────────────────
    def _markers(self, res: dict, carm_pos_m, stamp) -> MarkerArray:
        arr = MarkerArray()
        idx = 0

        def base(frame: str, mtype: int) -> Marker:
            nonlocal idx
            mk = Marker()
            mk.header.frame_id = frame
            mk.header.stamp = stamp
            mk.ns = "fluoro"
            mk.id = idx
            idx += 1
            mk.type = mtype
            mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0
            return mk

        # CT anatomy surface mesh (metres, at the anatomy origin)
        if MESH_OBJ.exists():
            mesh = base(ANATOMY, Marker.MESH_RESOURCE)
            mesh.mesh_resource = f"file://{MESH_OBJ}"
            mesh.mesh_use_embedded_materials = False
            mesh.scale.x = mesh.scale.y = mesh.scale.z = 1.0
            mesh.color = ColorRGBA(r=0.9, g=0.9, b=0.85, a=0.65)
            arr.markers.append(mesh)

        # Tool shaft (cylinder along the tool frame's local z, tip at origin)
        shaft = base(TOOL, Marker.CYLINDER)
        shaft.scale.x = shaft.scale.y = 0.010   # 10 mm diameter
        shaft.scale.z = 0.100                    # 100 mm long
        shaft.pose.position.z = -0.050           # extend from tip (0) to −100 mm
        shaft.color = ColorRGBA(r=0.0, g=0.85, b=1.0, a=0.9)
        arr.markers.append(shaft)

        # Tool tip (sphere at the TCP)
        tip = base(TOOL, Marker.SPHERE)
        tip.scale.x = tip.scale.y = tip.scale.z = 0.008
        tip.color = ColorRGBA(r=0.2, g=0.9, b=0.2, a=1.0)
        arr.markers.append(tip)

        # C-arm source (sphere in the carm frame) + beam axis to the isocenter
        src = base(CARM, Marker.SPHERE)
        src.scale.x = src.scale.y = src.scale.z = 0.03
        src.color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.9)
        arr.markers.append(src)

        beam = base(ANATOMY, Marker.LINE_STRIP)
        beam.scale.x = 0.003
        beam.color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.6)
        from geometry_msgs.msg import Point
        beam.points = [Point(x=carm_pos_m[0], y=carm_pos_m[1], z=carm_pos_m[2]),
                       Point(x=0.0, y=0.0, z=0.0)]
        arr.markers.append(beam)

        # Text panel: registration error + clinical check
        txt = base(ANATOMY, Marker.TEXT_VIEW_FACING)
        txt.pose.position.z = 0.09
        txt.scale.z = 0.012
        txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
        err = res.get("errors", {}).get("T_robot_in_anatomy_t_err_norm_mm")
        clin = res.get("clinical", {}).get("interpretation", "")
        err_s = f"reg ‖err‖ = {err:.3f} mm\n" if err is not None else ""
        txt.text = err_s + clin
        arr.markers.append(txt)
        return arr


def main():
    rclpy.init()
    node = FluoroTfBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
