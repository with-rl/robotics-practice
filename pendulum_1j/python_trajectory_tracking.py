# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import platform
import numpy as np
from scipy.integrate import odeint
import matplotlib
import matplotlib.pyplot as plt

if platform.system() == "Darwin":
    matplotlib.use("TkAgg")


class PythonPendulum1J:
    def __init__(self):
        self.m = 1
        self.I = 0.1
        self.c = 0.5
        self.l = 1
        self.g = 9.81

    def get_tau(self, Kp, Kd, theta, theta_d, theta_ref, theta_d_ref, theta_dd_ref):
        M = 1.0 * self.I + 1.0 * self.c**2 * self.m
        C = 0
        G = self.c * self.g * self.m * np.cos(theta)

        tau = (
            M * (theta_dd_ref - Kp * (theta - theta_ref) - Kd * (theta_d - theta_d_ref))
            + C
            + G
        )
        tau = np.clip(tau, -10, 10)
        return M, C, G, tau

    def pendulum_trajectory_tracking(self, z0, t, z_ref, Kp, Kd):
        theta, theta_d = z0
        theta_ref, theta_d_ref, theta_dd_ref = z_ref

        M, C, G, tau = self.get_tau(
            Kp, Kd, theta, theta_d, theta_ref, theta_d_ref, theta_dd_ref
        )

        A = np.array([[M]])
        b = -np.array([[C + G - tau]])

        x = np.linalg.solve(A, b)
        # theta_dd = (tau - C - G) / M
        return [theta_d, x[0][0]]


def simulate(simulator, z0, z_ref, Kp, Kd, ts):
    theta_ref, theta_d_ref, theta_dd_ref = z_ref
    taus = np.zeros((len(ts)))
    zs = np.zeros((len(ts), 2))
    zs[0] = z0
    for i in range(len(ts) - 1):
        temp_ts = np.array([ts[i], ts[i + 1]])
        result = odeint(
            simulator.pendulum_trajectory_tracking,
            z0,
            temp_ts,
            args=((theta_ref[i + 1], theta_d_ref[i + 1], theta_dd_ref[i + 1]), Kp, Kd),
        )
        _, _, _, tau = simulator.get_tau(
            Kp,
            Kd,
            z0[0],
            z0[1],
            theta_ref[i + 1],
            theta_d_ref[i + 1],
            theta_dd_ref[i + 1],
        )
        z0 = result[1]
        taus[i + 1] = tau
        zs[i + 1] = z0
    return zs, taus


def animate(zs, taus):
    for i, z in enumerate(zs):
        x, y = np.cos(z[0]), np.sin(z[0])
        (bar1,) = plt.plot([0, x], [0, y], linewidth=5, color="r")

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect("equal")

        plt.pause(0.02)
        if i + 1 < len(zs):
            bar1.remove()

    plt.pause(5)
    plt.close()

    # figure control signal
    plt.figure(1)

    plt.subplot(3, 1, 1)
    plt.plot(zs[:, 0], "r", label="theta")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(zs[:, 1], "r", label="theta_d")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(taus, "r", label="tau")
    plt.legend()

    plt.show()


def create_trajectory():
    pi = np.pi
    t1, t2, t3, t4 = 0, 2.0, 4.0, 6.0

    ts1 = np.linspace(t1, t2, 101)
    ts2 = np.linspace(t2, t3, 101)
    ts3 = np.linspace(t3, t4, 101)

    ts = np.concatenate((ts1, ts2[1:], ts3[1:]))

    # calculated by simpy_eom.py
    a10 = -pi / 2
    a11 = 0
    a12 = 0.175 * pi
    a13 = -0.025 * pi
    a20 = -0.3 * pi
    a21 = -0.3 * pi
    a22 = 0.325 * pi
    a23 = -0.05 * pi
    a30 = -9.9 * pi
    a31 = 6.9 * pi
    a32 = -1.475 * pi
    a33 = 0.1 * pi

    theta1 = a10 + a11 * ts1 + a12 * ts1**2 + a13 * ts1**3
    theta2 = a20 + a21 * ts2 + a22 * ts2**2 + a23 * ts2**3
    theta3 = a30 + a31 * ts3 + a32 * ts3**2 + a33 * ts3**3

    theta1_d = a11 + 2 * a12 * ts1 + 3 * a13 * ts1**2
    theta2_d = a21 + 2 * a22 * ts2 + 3 * a23 * ts2**2
    theta3_d = a31 + 2 * a32 * ts3 + 3 * a33 * ts3**2

    theta1_dd = 2 * a12 + 6 * a13 * ts1
    theta2_dd = 2 * a22 + 6 * a23 * ts2
    theta3_dd = 2 * a32 + 6 * a33 * ts3

    theta_ref = np.concatenate((theta1, theta2[1:], theta3[1:]))
    theta_d_ref = np.concatenate((theta1_d, theta2_d[1:], theta3_d[1:]))
    theta_dd_ref = np.concatenate((theta1_dd, theta2_dd[1:], theta3_dd[1:]))

    return ts, (theta_ref, theta_d_ref, theta_dd_ref)


if __name__ == "__main__":
    simulator = PythonPendulum1J()

    # Trajectory by time and theta
    ts, z_ref = create_trajectory()

    Kp = 25
    Kd = 2 * np.sqrt(Kp)
    z0 = [-np.pi / 2, 0]
    # Trajectory by position
    zs, taus = simulate(simulator, z0, z_ref, Kp, Kd, ts)
    # animation
    animate(zs, taus)
