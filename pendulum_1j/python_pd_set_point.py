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
        self.h = 0.02  # Delta Time
        self.m = 1
        self.I = 0.1
        self.c = 0.5
        self.l = 1
        self.g = 9.81

    def get_tau(self, Kp, Kd, theta, theta_d, theta_ref):
        tau = -Kp * (theta - theta_ref) - Kd * theta_d  # pd-control
        tau = np.clip(tau, -10, 10)
        return tau

    def pendulum_pd_set_point(self, z0, t, z_ref, Kp, Kd):
        theta, theta_d = z0
        theta_ref, _ = z_ref

        M = 1.0 * self.I + 1.0 * self.c**2 * self.m
        C = 0
        G = self.c * self.g * self.m * np.cos(theta)

        A = np.array([[M]])
        tau = self.get_tau(Kp, Kd, theta, theta_d, theta_ref)
        b = -np.array([[C + G - tau]])

        x = np.linalg.solve(A, b)
        return [theta_d, x[0][0]]


def simulate(simulator, z0, z_ref, Kp, Kd, N):
    ts = np.arange(N) * simulator.h
    taus = np.zeros((len(ts)))
    zs = np.zeros((len(ts), 2))
    zs[0] = z0
    for i in range(len(ts) - 1):
        temp_ts = np.array([ts[i], ts[i + 1]])
        result = odeint(
            simulator.pendulum_pd_set_point,
            z0,
            temp_ts,
            args=(z_ref, Kp, Kd),
        )
        tau = simulator.get_tau(Kp, Kd, z0[0], z0[1], z_ref[0])
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
    z0 = [-np.pi / 2, 0]
    z_ref = [np.pi / 2, 0]
    return z0, z_ref


if __name__ == "__main__":
    simulator = PythonPendulum1J()

    # Trajectory by angle
    z0, z_ref = create_trajectory()
    Kp = 25
    Kd = 10
    # Trajectory by position
    zs, taus = simulate(simulator, z0, z_ref, Kp, Kd, 500)
    # animation
    animate(zs, taus)
