#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import signal
import threading
from datetime import datetime
from typing import List, Optional

import numpy as np

# Use PySide6 as the Qt backend
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

from openarm_compl_ctrl.msg import PIDDebug


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

        # Downsampling: max points to display per plot (0 = no downsampling)
        self.max_display_points: int = 0

        # Thread safety
        self.lock = threading.Lock()

        # ---------------- SUBSCRIPTIONS ----------------
        self.create_subscription(
            JointState, "/debug_js", self.joint_state_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Twist, "/debug_ee_vel", self.ee_vel_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            PIDDebug,
            "/pid_debug_left_ee_ang_vel_z",
            self.pid_callback,
            qos_profile_sensor_data,
        )

        # ---------------- STORAGE ----------------
        # Separate time bases
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

        # PID
        self.time_pid: List[float] = []
        self.pid_p: List[float] = []
        self.pid_i: List[float] = []
        self.pid_d: List[float] = []
        self.pid_error: List[float] = []
        self.pid_control: List[float] = []

        self._spin_thread: Optional[ROSSpinThread] = None
        self._update_timer: Optional[QtCore.QTimer] = None

        self.setup_ui()

    def _elapsed(self, start: Optional[Time]) -> tuple[float, Time]:
        now = self.get_clock().now()
        if start is None:
            start = now
        t = (now - start).nanoseconds / 1e9
        return float(t), start

    def joint_state_callback(self, msg: JointState) -> None:
        with self.lock:
            t, self.start_time_js = self._elapsed(self.start_time_js)
            self.time_js.append(t)
            n = min(self.num_joints, len(msg.position))
            for i in range(n):
                self.q[i].append(float(msg.position[i]))
            for i in range(n, self.num_joints):
                self.q[i].append(0.0)
            # Store effort values
            n_effort = min(self.num_joints, len(msg.effort))
            for i in range(n_effort):
                self.effort[i].append(float(msg.effort[i]))
            for i in range(n_effort, self.num_joints):
                self.effort[i].append(0.0)

    def ee_vel_callback(self, msg: Twist) -> None:
        with self.lock:
            t, self.start_time_ee = self._elapsed(self.start_time_ee)
            self.time_ee.append(t)
            vals = [
                float(msg.linear.x),
                float(msg.linear.y),
                float(msg.linear.z),
                float(msg.angular.x),
                float(msg.angular.y),
                float(msg.angular.z),
            ]
            for i in range(6):
                self.ee_vel[i].append(vals[i])

    def pid_callback(self, msg: PIDDebug) -> None:
        with self.lock:
            t, self.start_time_pid = self._elapsed(self.start_time_pid)
            self.time_pid.append(t)
            self.pid_p.append(float(msg.p))
            self.pid_i.append(float(msg.i))
            self.pid_d.append(float(msg.d))
            self.pid_error.append(float(msg.error))
            self.pid_control.append(float(msg.control_sig))

    def setup_ui(self) -> None:
        # Create main window
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("OpenArm Debug Visualizer")
        self.window.resize(1200, 800)

        # Central widget with layout
        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Create plots
        self.plots: List[pg.PlotWidget] = []
        self.curves: List[List[pg.PlotDataItem]] = []

        if self.ENABLE_JOINTS:
            plot = pg.PlotWidget(title="Joint Positions")
            plot.setLabel("left", "Position (rad)")
            plot.setLabel("bottom", "Time (s)")
            plot.addLegend()
            plot.showGrid(x=True, y=True)
            plot.getAxis("left").enableAutoSIPrefix(False)
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
            ]
            curves = []
            for i in range(self.num_joints):
                pen = pg.mkPen(color=colors[i % len(colors)], width=2)
                curve = plot.plot([], [], pen=pen, name=f"J{i + 1}")
                curves.append(curve)
            self.plots.append(plot)
            self.curves.append(curves)
            layout.addWidget(plot)

        if self.ENABLE_EE:
            plot = pg.PlotWidget(title="End-Effector Velocity")
            plot.setLabel("left", "Vel (m/s or rad/s)")
            plot.setLabel("bottom", "Time (s)")
            plot.addLegend()
            plot.showGrid(x=True, y=True)
            # Disable SI prefix auto-scaling (e.g., x0.001) on Y axis
            plot.getAxis("left").enableAutoSIPrefix(False)
            labels = ["vx", "vy", "vz", "wx", "wy", "wz"]
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            curves = []
            for i in range(6):
                pen = pg.mkPen(color=colors[i % len(colors)], width=2)
                curve = plot.plot([], [], pen=pen, name=labels[i])
                curves.append(curve)
            self.plots.append(plot)
            self.curves.append(curves)
            layout.addWidget(plot)

        if self.ENABLE_PID:
            # Separate plots for each PID component to avoid scaling issues
            pid_labels = [
                ("P Term", "p", "#1f77b4"),
                ("I Term", "i", "#ff7f0e"),
                ("D Term", "d", "#2ca02c"),
                ("Error", "error", "#d62728"),
                ("Control Signal", "control", "#9467bd"),
            ]
            for title, _, color in pid_labels:
                plot = pg.PlotWidget(title=f"PID: {title}")
                plot.setLabel("left", "Value")
                plot.setLabel("bottom", "Time (s)")
                plot.showGrid(x=True, y=True)
                plot.getAxis("left").enableAutoSIPrefix(False)
                pen = pg.mkPen(color=color, width=2)
                curve = plot.plot([], [], pen=pen)
                self.plots.append(plot)
                self.curves.append([curve])
                layout.addWidget(plot)

        # Joint Effort plot (last)
        if self.ENABLE_JOINTS:
            plot = pg.PlotWidget(title="Joint Effort (Torque)")
            plot.setLabel("left", "Effort (Nm)")
            plot.setLabel("bottom", "Time (s)")
            plot.addLegend()
            plot.showGrid(x=True, y=True)
            plot.getAxis("left").enableAutoSIPrefix(False)
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
            ]
            effort_curves = []
            for i in range(self.num_joints):
                pen = pg.mkPen(color=colors[i % len(colors)], width=2)
                curve = plot.plot([], [], pen=pen, name=f"J{i + 1}")
                effort_curves.append(curve)
            self.plots.append(plot)
            self.curves.append(effort_curves)
            layout.addWidget(plot)

        # Buttons
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

        # Theme state - default to light
        self.is_dark_theme = False
        self.apply_theme()

        # Setup update timer (50ms = 20Hz refresh)
        self._update_timer = QtCore.QTimer()
        self._update_timer.timeout.connect(self.update_plots)
        self._update_timer.start(50)

    def downsample(self, data: np.ndarray) -> np.ndarray:
        """Downsample data if it exceeds max_display_points."""
        if self.max_display_points <= 0 or len(data) <= self.max_display_points:
            return data
        # Select evenly spaced indices
        indices = np.linspace(0, len(data) - 1, self.max_display_points, dtype=int)
        return data[indices]

    def update_plots(self) -> None:
        with self.lock:
            plot_idx = 0

            if self.ENABLE_JOINTS and plot_idx < len(self.plots):
                if len(self.time_js) >= 2:
                    t = np.array(self.time_js, dtype=np.float64)
                    t_plot = self.downsample(t)
                    for i in range(self.num_joints):
                        y = np.array(self.q[i], dtype=np.float64)
                        min_len = min(len(t), len(y))
                        if min_len > 0:
                            y_plot = self.downsample(y[:min_len])
                            self.curves[plot_idx][i].setData(t_plot, y_plot)
                    # Auto-range with some padding
                    self.plots[plot_idx].setXRange(t[0], t[-1], padding=0.01)
                plot_idx += 1

            if self.ENABLE_EE and plot_idx < len(self.plots):
                if len(self.time_ee) >= 2:
                    t = np.array(self.time_ee, dtype=np.float64)
                    t_plot = self.downsample(t)
                    for i in range(6):
                        y = np.array(self.ee_vel[i], dtype=np.float64)
                        min_len = min(len(t), len(y))
                        if min_len > 0:
                            y_plot = self.downsample(y[:min_len])
                            self.curves[plot_idx][i].setData(t_plot, y_plot)
                    self.plots[plot_idx].setXRange(t[0], t[-1], padding=0.01)
                    # Auto-scale Y axis to fit data
                    self.plots[plot_idx].autoRange()
                plot_idx += 1

            if self.ENABLE_PID and plot_idx < len(self.plots):
                if len(self.time_pid) >= 2:
                    t = np.array(self.time_pid, dtype=np.float64)
                    t_plot = self.downsample(t)
                    pid_data_series = [
                        self.pid_p,
                        self.pid_i,
                        self.pid_d,
                        self.pid_error,
                        self.pid_control,
                    ]
                    for i, data in enumerate(pid_data_series):
                        if plot_idx + i < len(self.plots):
                            y = np.array(data, dtype=np.float64)
                            min_len = min(len(t), len(y))
                            if min_len > 0:
                                y_plot = self.downsample(y[:min_len])
                                self.curves[plot_idx + i][0].setData(t_plot, y_plot)
                            self.plots[plot_idx + i].setXRange(
                                t[0], t[-1], padding=0.01
                            )
                    # Advance plot_idx past all PID plots
                    plot_idx += 5

            # Joint Effort plot (last)
            if self.ENABLE_JOINTS and plot_idx < len(self.plots):
                if len(self.time_js) >= 2:
                    t = np.array(self.time_js, dtype=np.float64)
                    t_plot = self.downsample(t)
                    for i in range(self.num_joints):
                        y = np.array(self.effort[i], dtype=np.float64)
                        min_len = min(len(t), len(y))
                        if min_len > 0:
                            y_plot = self.downsample(y[:min_len])
                            self.curves[plot_idx][i].setData(t_plot, y_plot)
                    self.plots[plot_idx].setXRange(t[0], t[-1], padding=0.01)

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

            self.pid_p.clear()
            self.pid_i.clear()
            self.pid_d.clear()
            self.pid_error.clear()
            self.pid_control.clear()

        # Clear curves
        for curves in self.curves:
            for curve in curves:
                curve.setData([], [])

    def save_figure(self) -> None:
        """Save the entire window as PNG."""
        filename = datetime.now().strftime("debug_%Y%m%d_%H%M%S.png")
        path = os.path.abspath(filename)
        # Capture the entire window using Qt
        pixmap = self.window.grab()
        pixmap.save(path)
        self.get_logger().info(f"Saved: {path}")

    def export_csv(self) -> None:
        """Export all data to CSV file."""
        filename = datetime.now().strftime("debug_%Y%m%d_%H%M%S.csv")
        path = os.path.abspath(filename)

        with self.lock:
            with open(path, "w") as f:
                # Write header
                header = ["time"]
                header.extend([f"q{i + 1}" for i in range(self.num_joints)])
                header.extend(["vx", "vy", "vz", "wx", "wy", "wz"])
                header.extend(["pid_p", "pid_i", "pid_d", "pid_error", "pid_control"])
                f.write(",".join(header) + "\n")

                # Find the minimum length across all data series
                min_len = min(
                    len(self.time_js),
                    len(self.time_ee),
                    len(self.time_pid),
                )

                # Write data rows
                for i in range(min_len):
                    row = [str(self.time_js[i])]
                    row.extend([str(self.q[j][i]) for j in range(self.num_joints)])
                    row.extend([str(self.ee_vel[j][i]) for j in range(6)])
                    row.extend(
                        [
                            str(self.pid_p[i]),
                            str(self.pid_i[i]),
                            str(self.pid_d[i]),
                            str(self.pid_error[i]),
                            str(self.pid_control[i]),
                        ]
                    )
                    f.write(",".join(row) + "\n")

        self.get_logger().info(f"Exported CSV: {path}")

    def apply_theme(self) -> None:
        """Apply current theme to plots."""
        if self.is_dark_theme:
            self.theme_btn.setText("Theme: Dark")
            bg_color = "k"  # black
            text_color = "w"  # white
        else:
            self.theme_btn.setText("Theme: Light")
            bg_color = "w"  # white
            text_color = "k"  # black

        # Update all plots
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
        """Toggle between dark and light themes."""
        self.is_dark_theme = not self.is_dark_theme
        self.apply_theme()

    def show(self) -> None:
        self.window.show()

    def start_spin_in_background(self) -> None:
        """Start ROS spin in a QThread to avoid timer conflicts."""
        if self._spin_thread is not None:
            return

        self._spin_thread = ROSSpinThread(self)
        self._spin_thread.start()


def main(args=None) -> None:
    rclpy.init(args=args)

    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)

    node = DebugVisualizer()
    node.show()
    node.start_spin_in_background()

    # Handle Ctrl+C (SIGINT) gracefully
    def signal_handler(signum, frame):
        print("\nCtrl+C pressed, shutting down...")
        app.quit()

    signal.signal(signal.SIGINT, signal_handler)

    # Use a timer to allow Python signal handlers to run during Qt event loop
    timer = QtCore.QTimer()
    timer.start(500)  # 500ms interval
    timer.timeout.connect(lambda: None)  # Dummy connection to process events

    # Run Qt event loop
    try:
        sys.exit(app.exec())
    finally:
        # Stop the ROS spin thread first
        if node._spin_thread is not None:
            node._spin_thread.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
