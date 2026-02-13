#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from collections import deque
import numpy as np
import threading


class DebugVisualizer(Node):

    def __init__(self):
        super().__init__('debug_visualizer')

        # Thread safety
        self.lock = threading.Lock()

        # ---------------- SUBSCRIBERS ----------------
        self.create_subscription(JointState,
                                 '/debug_js',
                                 self.joint_state_callback,
                                 10)

        self.create_subscription(Twist,
                                 '/debug_ee_vel',
                                 self.ee_vel_callback,
                                 10)

        self.create_subscription(Twist,
                                 '/debug_ee_vel_error',
                                 self.ee_error_callback,
                                 10)

        self.create_subscription(Twist,
                                 '/debug_ee_vel_cntrl_sig',
                                 self.ee_ctrl_callback,
                                 10)

        # ---------------- DATA STORAGE ----------------
        self.max_points = 1000
        self.start_time = None

        # Joint state
        self.time_js = deque(maxlen=self.max_points)
        self.num_joints = 7
        self.q = [deque(maxlen=self.max_points) for _ in range(self.num_joints)]
        self.qd = [deque(maxlen=self.max_points) for _ in range(self.num_joints)]
        self.eff = [deque(maxlen=self.max_points) for _ in range(self.num_joints)]

        # EE velocity
        self.time_ee_vel = deque(maxlen=self.max_points)
        self.ee_vel = [deque(maxlen=self.max_points) for _ in range(6)]

        # EE error
        self.time_ee_error = deque(maxlen=self.max_points)
        self.ee_error = [deque(maxlen=self.max_points) for _ in range(6)]

        # EE control
        self.time_ee_ctrl = deque(maxlen=self.max_points)
        self.ee_ctrl = [deque(maxlen=self.max_points) for _ in range(6)]

        self.joint_names = [f'J{i+1}' for i in range(self.num_joints)]
        self.twist_labels = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']

        self.setup_plots()

        # ---------------- RESET BUTTON ----------------
        reset_ax = self.fig.add_axes([0.90, 0.94, 0.08, 0.04])
        self.reset_button = Button(reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset_plots)

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    def get_elapsed_time(self):
        if self.start_time is None:
            self.start_time = self.get_clock().now()
        now = self.get_clock().now()
        return (now - self.start_time).nanoseconds / 1e9

    # ---------------------------------------------------------
    # Callbacks (thread safe)
    # ---------------------------------------------------------

    def joint_state_callback(self, msg):
        with self.lock:
            t = self.get_elapsed_time()
            self.time_js.append(t)

            n = min(self.num_joints,
                    len(msg.position),
                    len(msg.velocity),
                    len(msg.effort))

            for i in range(n):
                self.q[i].append(msg.position[i])
                self.qd[i].append(msg.velocity[i])
                self.eff[i].append(msg.effort[i])

            for i in range(n, self.num_joints):
                self.q[i].append(0.0)
                self.qd[i].append(0.0)
                self.eff[i].append(0.0)

    def ee_vel_callback(self, msg):
        with self.lock:
            t = self.get_elapsed_time()
            self.time_ee_vel.append(t)

            twist = [
                msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z
            ]
            for i in range(6):
                self.ee_vel[i].append(twist[i])

    def ee_error_callback(self, msg):
        with self.lock:
            t = self.get_elapsed_time()
            self.time_ee_error.append(t)

            twist = [
                msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z
            ]
            for i in range(6):
                self.ee_error[i].append(twist[i])

    def ee_ctrl_callback(self, msg):
        with self.lock:
            t = self.get_elapsed_time()
            self.time_ee_ctrl.append(t)

            twist = [
                msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z
            ]
            for i in range(6):
                self.ee_ctrl[i].append(twist[i])

    # ---------------------------------------------------------
    # Plot Setup
    # ---------------------------------------------------------

    def setup_plots(self):
        self.fig, self.axes = plt.subplots(6, 1, figsize=(12, 16))
        self.fig.suptitle("OpenArm Debug Visualizer")

        # Joint Position
        self.q_lines = []
        for i in range(self.num_joints):
            line, = self.axes[0].plot([], [], label=self.joint_names[i])
            self.q_lines.append(line)
        self.axes[0].set_ylabel("q (rad)")
        self.axes[0].legend(ncol=7, fontsize=8)
        self.axes[0].grid(True)

        # Joint Velocity
        self.qd_lines = []
        for i in range(self.num_joints):
            line, = self.axes[1].plot([], [], label=self.joint_names[i])
            self.qd_lines.append(line)
        self.axes[1].set_ylabel("qd (rad/s)")
        self.axes[1].legend(ncol=7, fontsize=8)
        self.axes[1].grid(True)

        # Joint Effort
        self.eff_lines = []
        for i in range(self.num_joints):
            line, = self.axes[2].plot([], [], label=self.joint_names[i])
            self.eff_lines.append(line)
        self.axes[2].set_ylabel("Effort (Nm)")
        self.axes[2].legend(ncol=7, fontsize=8)
        self.axes[2].grid(True)

        # EE Velocity
        self.vel_lines = []
        for i in range(6):
            line, = self.axes[3].plot([], [], label=self.twist_labels[i])
            self.vel_lines.append(line)
        self.axes[3].set_ylabel("EE Velocity")
        self.axes[3].legend(ncol=6, fontsize=8)
        self.axes[3].grid(True)

        # EE Error
        self.err_lines = []
        for i in range(6):
            line, = self.axes[4].plot([], [], label=self.twist_labels[i])
            self.err_lines.append(line)
        self.axes[4].set_ylabel("EE Velocity Error")
        self.axes[4].legend(ncol=6, fontsize=8)
        self.axes[4].grid(True)

        # EE Control
        self.ctrl_lines = []
        for i in range(6):
            line, = self.axes[5].plot([], [], label=self.twist_labels[i])
            self.ctrl_lines.append(line)
        self.axes[5].set_ylabel("EE Control Signal")
        self.axes[5].set_xlabel("Time (s)")
        self.axes[5].legend(ncol=6, fontsize=8)
        self.axes[5].grid(True)

        plt.tight_layout()

    # ---------------------------------------------------------
    # Plot Update (thread safe snapshot)
    # ---------------------------------------------------------

    def update_plots(self, _):

        with self.lock:
            time_js = np.array(self.time_js)
            q = [np.array(self.q[i]) for i in range(self.num_joints)]
            qd = [np.array(self.qd[i]) for i in range(self.num_joints)]
            eff = [np.array(self.eff[i]) for i in range(self.num_joints)]

            time_vel = np.array(self.time_ee_vel)
            ee_vel = [np.array(self.ee_vel[i]) for i in range(6)]

            time_err = np.array(self.time_ee_error)
            ee_err = [np.array(self.ee_error[i]) for i in range(6)]

            time_ctrl = np.array(self.time_ee_ctrl)
            ee_ctrl = [np.array(self.ee_ctrl[i]) for i in range(6)]

        if len(time_js) > 1:
            for i in range(self.num_joints):
                self.q_lines[i].set_data(time_js, q[i])
                self.qd_lines[i].set_data(time_js, qd[i])
                self.eff_lines[i].set_data(time_js, eff[i])
            for ax in self.axes[0:3]:
                ax.set_xlim(time_js[0], time_js[-1])
                ax.relim()
                ax.autoscale_view()

        if len(time_vel) > 1:
            for i in range(6):
                self.vel_lines[i].set_data(time_vel, ee_vel[i])
            self.axes[3].set_xlim(time_vel[0], time_vel[-1])
            self.axes[3].relim()
            self.axes[3].autoscale_view()

        if len(time_err) > 1:
            for i in range(6):
                self.err_lines[i].set_data(time_err, ee_err[i])
            self.axes[4].set_xlim(time_err[0], time_err[-1])
            self.axes[4].relim()
            self.axes[4].autoscale_view()

        if len(time_ctrl) > 1:
            for i in range(6):
                self.ctrl_lines[i].set_data(time_ctrl, ee_ctrl[i])
            self.axes[5].set_xlim(time_ctrl[0], time_ctrl[-1])
            self.axes[5].relim()
            self.axes[5].autoscale_view()

        return (self.q_lines + self.qd_lines +
                self.eff_lines + self.vel_lines +
                self.err_lines + self.ctrl_lines)


    def reset_plots(self, event):
        with self.lock:
            self.start_time = None

            # Clear all stored data
            self.time_js.clear()
            self.time_ee_vel.clear()
            self.time_ee_error.clear()
            self.time_ee_ctrl.clear()

            for i in range(self.num_joints):
                self.q[i].clear()
                self.qd[i].clear()
                self.eff[i].clear()

            for i in range(6):
                self.ee_vel[i].clear()
                self.ee_error[i].clear()
                self.ee_ctrl[i].clear()

        # Clear visual lines immediately
        for line in (self.q_lines + self.qd_lines +
                     self.eff_lines + self.vel_lines +
                     self.err_lines + self.ctrl_lines):
            line.set_data([], [])

        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw_idle()


def main(args=None):
    rclpy.init(args=args)

    node = DebugVisualizer()

    ani = animation.FuncAnimation(
        node.fig,
        node.update_plots,
        interval=50,
        blit=False,
        cache_frame_data=False
    )

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )
    spin_thread.start()

    plt.show()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

