#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import signal
import threading
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

from openarm_compl_ctrl.msg import PIDDebug, CartesianPIDDebug


class ROSSpinThread(QtCore.QThread):
    """QThread wrapper for ROS spinning to avoid timer conflicts."""
    def __init__(self, node: Node) -> None:
        super().__init__()
        self.node = node
        self._running = True

    def run(self) -> None:
        while self._running and rclpy.ok():
            try:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            except Exception as exc:
                self.node.get_logger().error(f"Spin exception: {exc!r}")
                break

    def stop(self) -> None:
        self._running = False
        self.wait()


class DebugVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("debug_visualizer")

        # ---------------- FLAGS ----------------
        self.ENABLE_JOINTS: bool = True
        self.ENABLE_EE: bool = True
        self.ENABLE_PID: bool = True

        self.lock = threading.Lock()

        # ---------------- SUBSCRIPTIONS ----------------
        self.create_subscription(
            JointState, "/debug_js", self.joint_state_callback, qos_profile_sensor_data
        )
        # Updated: Cartesian PID debug (6 PIDDebug fields)
        self.create_subscription(
            CartesianPIDDebug, "/cart_pid_debug", self.pid_callback, qos_profile_sensor_data
        )

        # ---------------- STORAGE ----------------
        self.start_time_js: Optional[Time] = None
        self.start_time_ee: Optional[Time] = None
        self.start_time_pid: Optional[Time] = None

        # Joint
        self.num_joints: int = 7
        self.time_js: List[float] = []
        self.q: List[List[float]] = [[] for _ in range(self.num_joints)]
        self.effort: List[List[float]] = [[] for _ in range(self.num_joints)]

        # EE vel
        self.time_ee: List[float] = []
        self.ee_vel: List[List[float]] = [[] for _ in range(6)]

        # Cartesian PID: 6 axes, 5 components
        # Axes: linear xyz + angular rpy (yaw)
        self.pid_axes = ["x", "y", "z", "roll", "pitch", "yaw"]
        self.pid_axis_labels = ["lin x", "lin y", "lin z", "ang r", "ang p", "ang y"]
        self.pid_components = ["p", "i", "d", "error", "control_sig"]

        self.time_pid: List[float] = []
        self.pid_enabled: Dict[str, bool] = {ax: True for ax in self.pid_axes}
        self.pid_data: Dict[str, Dict[str, List[float]]] = {
            ax: {comp: [] for comp in self.pid_components} for ax in self.pid_axes
        }

        self._spin_thread: Optional[ROSSpinThread] = None
        self._update_timer: Optional[QtCore.QTimer] = None

        self.setup_ui()

    def _elapsed(self, start: Optional[Time]) -> tuple[float, Time]:
        now = self.get_clock().now()
        if start is None:
            start = now
        t = (now - start).nanoseconds / 1e9
        return float(t), start

    # ---------------- CALLBACKS ----------------

    def joint_state_callback(self, msg: JointState) -> None:
        with self.lock:
            t, self.start_time_js = self._elapsed(self.start_time_js)
            self.time_js.append(t)

            n = min(self.num_joints, len(msg.position))
            for i in range(n):
                self.q[i].append(float(msg.position[i]))
            for i in range(n, self.num_joints):
                self.q[i].append(0.0)

            n_effort = min(self.num_joints, len(msg.effort))
            for i in range(n_effort):
                self.effort[i].append(float(msg.effort[i]))
            for i in range(n_effort, self.num_joints):
                self.effort[i].append(0.0)

    @staticmethod
    def _append_pid(dst: Dict[str, List[float]], src: PIDDebug) -> None:
        dst["p"].append(float(src.p))
        dst["i"].append(float(src.i))
        dst["d"].append(float(src.d))
        dst["error"].append(float(src.error))
        dst["control_sig"].append(float(src.control_sig))

    @staticmethod
    def _get_cart_axis(msg: CartesianPIDDebug, name: str) -> PIDDebug:
        # Prefer canonical field names: x, y, z, r, p, yaw
        if hasattr(msg, name):
            return getattr(msg, name)
        # Fallbacks if you used a different yaw name
        if name == "yaw":
            for alt in ("yawv", "yy", "y_"):
                if hasattr(msg, alt):
                    return getattr(msg, alt)
        raise AttributeError(
            f"CartesianPIDDebug has no field '{name}'. "
            f"Use fields: x,y,z,r,p,yaw (recommended)."
        )

    def pid_callback(self, msg: CartesianPIDDebug) -> None:
        with self.lock:
            t, self.start_time_pid = self._elapsed(self.start_time_pid)
            self.time_pid.append(t)

            for ax in self.pid_axes:
                pid = self._get_cart_axis(msg, ax)
                self._append_pid(self.pid_data[ax], pid)

    # ---------------- UI ----------------

    def setup_ui(self) -> None:
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("OpenArm Debug Visualizer")
        self.window.resize(1200, 1000)

        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.plots: List[pg.PlotWidget] = []
        self.curves: List[List[pg.PlotDataItem]] = []

        # Plot indices (so we can update reliably)
        self.plot_idx_joint_pos: Optional[int] = None
        self.plot_idx_pid: Dict[str, int] = {}  # comp -> plot index
        self.plot_idx_joint_eff: Optional[int] = None

        # ---------------- JOINT POSITIONS ----------------
        if self.ENABLE_JOINTS:
            plot = pg.PlotWidget(title="Joint Positions")
            plot.setLabel("left", "Position (rad)")
            plot.setLabel("bottom", "Time (s)")
            plot.addLegend()
            plot.showGrid(x=True, y=True)
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.enableAutoRange(x=True, y=True)

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
            curves = []
            for i in range(self.num_joints):
                pen = pg.mkPen(color=colors[i % len(colors)], width=2)
                curves.append(plot.plot([], [], pen=pen, name=f"J{i + 1}"))

            self.plot_idx_joint_pos = len(self.plots)
            self.plots.append(plot)
            self.curves.append(curves)
            layout.addWidget(plot)

        # ---------------- PID GROUP: CHECKBOXES + 5 PLOTS ----------------
        if self.ENABLE_PID:
            pid_box = QtWidgets.QGroupBox("Cartesian PID Debug (6 axes)")
            pid_layout = QtWidgets.QVBoxLayout(pid_box)

            # Axis enable/disable controls (affects all 5 PID plots)
            controls = QtWidgets.QHBoxLayout()
            controls.addWidget(QtWidgets.QLabel("Show axes:"))

            self.pid_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
            for ax, label in zip(self.pid_axes, self.pid_axis_labels):
                cb = QtWidgets.QCheckBox(label)
                cb.setChecked(True)
                cb.stateChanged.connect(lambda state, a=ax: self._toggle_pid_axis(a, state))
                self.pid_checkboxes[ax] = cb
                controls.addWidget(cb)

            controls.addStretch()
            pid_layout.addLayout(controls)

            # 5 plots: P/I/D/Error/Signal, each with 6 curves
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

            titles = [
                ("p", "PID: P (6 axes)"),
                ("i", "PID: I (6 axes)"),
                ("d", "PID: D (6 axes)"),
                ("error", "PID: Error (6 axes)"),
                ("control_sig", "PID: Signal (6 axes)"),
            ]

            for comp, title in titles:
                plot = pg.PlotWidget(title=title)
                plot.setLabel("left", "Value")
                plot.setLabel("bottom", "Time (s)")
                plot.addLegend()
                plot.showGrid(x=True, y=True)
                plot.getAxis("left").enableAutoSIPrefix(False)
                plot.enableAutoRange(x=True, y=True)

                curves = []
                for i, (ax, label) in enumerate(zip(self.pid_axes, self.pid_axis_labels)):
                    pen = pg.mkPen(color=colors[i % len(colors)], width=2)
                    curves.append(plot.plot([], [], pen=pen, name=label))

                self.plot_idx_pid[comp] = len(self.plots)
                self.plots.append(plot)
                self.curves.append(curves)
                pid_layout.addWidget(plot)

            layout.addWidget(pid_box)

        # ---------------- JOINT EFFORT ----------------
        if self.ENABLE_JOINTS:
            plot = pg.PlotWidget(title="Joint Effort (Torque)")
            plot.setLabel("left", "Effort (Nm)")
            plot.setLabel("bottom", "Time (s)")
            plot.addLegend()
            plot.showGrid(x=True, y=True)
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.enableAutoRange(x=True, y=True)

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
            curves = []
            for i in range(self.num_joints):
                pen = pg.mkPen(color=colors[i % len(colors)], width=2)
                curves.append(plot.plot([], [], pen=pen, name=f"J{i + 1}"))

            self.plot_idx_joint_eff = len(self.plots)
            self.plots.append(plot)
            self.curves.append(curves)
            layout.addWidget(plot)

        # ---------------- BUTTONS ----------------
        btn_layout = QtWidgets.QHBoxLayout()

        self.theme_btn = QtWidgets.QPushButton("Theme: Light")
        self.theme_btn.clicked.connect(self.toggle_theme)
        btn_layout.addWidget(self.theme_btn)

        export_csv_btn = QtWidgets.QPushButton("Export CSV")
        export_csv_btn.clicked.connect(self.export_csv)
        btn_layout.addWidget(export_csv_btn)

        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_plots)
        btn_layout.addWidget(reset_btn)

        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self.save_figure)
        btn_layout.addWidget(save_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Theme state
        self.is_dark_theme = False
        self.apply_theme()

        # Update timer
        self._update_timer = QtCore.QTimer()
        self._update_timer.timeout.connect(self.update_plots)
        self._update_timer.start(50)

    def _toggle_pid_axis(self, axis: str, state: int) -> None:
        self.pid_enabled[axis] = state == QtCore.Qt.Checked
        # Apply visibility immediately across all PID plots
        for comp, pidx in self.plot_idx_pid.items():
            axis_i = self.pid_axes.index(axis)
            self.curves[pidx][axis_i].setVisible(self.pid_enabled[axis])

    # ---------------- PLOTTING ----------------

    def update_plots(self) -> None:
        with self.lock:
            # Joint positions
            if self.ENABLE_JOINTS and self.plot_idx_joint_pos is not None:
                if len(self.time_js) >= 2:
                    t = np.asarray(self.time_js, dtype=np.float64)
                    for i in range(self.num_joints):
                        y = np.asarray(self.q[i], dtype=np.float64)
                        n = min(len(t), len(y))
                        if n > 0:
                            self.curves[self.plot_idx_joint_pos][i].setData(t[:n], y[:n])

            # PID: 5 plots * 6 curves
            if self.ENABLE_PID and len(self.time_pid) >= 2:
                t = np.asarray(self.time_pid, dtype=np.float64)

                for comp, pidx in self.plot_idx_pid.items():
                    for axis_i, ax in enumerate(self.pid_axes):
                        visible = self.pid_enabled.get(ax, True)
                        self.curves[pidx][axis_i].setVisible(visible)
                        if not visible:
                            continue
                        y = np.asarray(self.pid_data[ax][comp], dtype=np.float64)
                        n = min(len(t), len(y))
                        if n > 0:
                            self.curves[pidx][axis_i].setData(t[:n], y[:n])

            # Joint effort
            if self.ENABLE_JOINTS and self.plot_idx_joint_eff is not None:
                if len(self.time_js) >= 2:
                    t = np.asarray(self.time_js, dtype=np.float64)
                    for i in range(self.num_joints):
                        y = np.asarray(self.effort[i], dtype=np.float64)
                        n = min(len(t), len(y))
                        if n > 0:
                            self.curves[self.plot_idx_joint_eff][i].setData(t[:n], y[:n])

    # ---------------- UTILITIES ----------------

    def reset_plots(self) -> None:
        with self.lock:
            self.start_time_js = None
            self.start_time_ee = None
            self.start_time_pid = None

            self.time_js.clear()
            self.time_ee.clear()
            self.time_pid.clear()

            for q in self.q:
                q.clear()
            for e in self.effort:
                e.clear()
            for v in self.ee_vel:
                v.clear()

            for ax in self.pid_axes:
                for comp in self.pid_components:
                    self.pid_data[ax][comp].clear()

        for curves in self.curves:
            for curve in curves:
                curve.setData([], [])

    def save_figure(self) -> None:
        filename = datetime.now().strftime("debug_%Y%m%d_%H%M%S.png")
        path = os.path.abspath(filename)
        pixmap = self.window.grab()
        pixmap.save(path)
        self.get_logger().info(f"Saved: {path}")

    def export_csv(self) -> None:
        filename = datetime.now().strftime("debug_%Y%m%d_%H%M%S.csv")
        path = os.path.abspath(filename)

        with self.lock:
            with open(path, "w") as f:
                header = ["time"]

                header.extend([f"q{i + 1}" for i in range(self.num_joints)])
                header.extend([f"eff{i + 1}" for i in range(self.num_joints)])

                # PID: all components for all axes
                for comp in self.pid_components:
                    for ax in self.pid_axes:
                        header.append(f"pid_{comp}_{ax}")

                f.write(",".join(header) + "\n")

                min_len = min(len(self.time_js), len(self.time_ee), len(self.time_pid))

                for i in range(min_len):
                    row = [str(self.time_js[i])]
                    row.extend([str(self.q[j][i]) for j in range(self.num_joints)])
                    row.extend([str(self.effort[j][i]) for j in range(self.num_joints)])

                    for comp in self.pid_components:
                        for ax in self.pid_axes:
                            row.append(str(self.pid_data[ax][comp][i]))

                    f.write(",".join(row) + "\n")

        self.get_logger().info(f"Exported CSV: {path}")

    def apply_theme(self) -> None:
        if self.is_dark_theme:
            self.theme_btn.setText("Theme: Dark")
            bg_color = "k"
            text_color = "w"
        else:
            self.theme_btn.setText("Theme: Light")
            bg_color = "w"
            text_color = "k"

        for plot in self.plots:
            plot.setBackground(bg_color)
            axis_bottom = plot.getAxis("bottom")
            axis_left = plot.getAxis("left")
            axis_bottom.setPen(text_color)
            axis_bottom.setTextPen(text_color)
            axis_left.setPen(text_color)
            axis_left.setTextPen(text_color)
            plot.showGrid(x=True, y=True, alpha=0.3)

    def toggle_theme(self) -> None:
        self.is_dark_theme = not self.is_dark_theme
        self.apply_theme()

    def show(self) -> None:
        self.window.show()

    def start_spin_in_background(self) -> None:
        if self._spin_thread is not None:
            return
        self._spin_thread = ROSSpinThread(self)
        self._spin_thread.start()


def main(args=None) -> None:
    rclpy.init(args=args)

    app = QtWidgets.QApplication(sys.argv)

    node = DebugVisualizer()
    node.show()
    node.start_spin_in_background()

    def signal_handler(signum, frame):
        app.quit()

    signal.signal(signal.SIGINT, signal_handler)

    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    try:
        sys.exit(app.exec())
    finally:
        if node._spin_thread is not None:
            node._spin_thread.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
