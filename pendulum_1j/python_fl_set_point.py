# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from scipy.integrate import odeint
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


class PythonPendulum1J:
    def __init__(self):
        self.h = 0.02  # Delta Time
        self.m = 1
        self.I = 0.1
        self.c = 0.5
        self.l = 1
        self.g = 9.81

    def pendulum_fl_set_point(self, z0, t, theta1_ref, Kp):
        Kd = 2 * np.sqrt(Kp)
        theta1, theta1_d = z0

        # huristic M, C, G
        M_hat = 2.5
        C_hat = 0.1
        G_hat = 0.5

        A = np.array([[M_hat]])
        tau = (
            M_hat * (-Kp * (theta1 - theta1_ref) - Kd * theta1_d)
            + C_hat * (theta1_d)
            + G_hat * (theta1)
        )
        tau = np.clip(tau, -10, 10)
        b = -np.array([[C_hat * theta1_d + G_hat * theta1 - tau]])

        x = np.linalg.solve(A, b)
        return [theta1_d, x[0][0]]


def simulate(simulator, theta1, theta1_ref, Kp, N):
    ts = np.arange(N) * simulator.h
    theta1_d = 0
    z0 = np.array([theta1, theta1_d])
    zs = odeint(simulator.pendulum_fl_set_point, z0, ts, args=(theta1_ref, Kp))
    return zs


def animate(zs):
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


if __name__ == "__main__":
    simulator = PythonPendulum1J()

    # Trajectory by angle
    theta1 = -np.pi / 2
    theta1_ref = np.pi / 2
    Kp = 25
    # Trajectory by position
    zs = simulate(simulator, theta1, theta1_ref, Kp, 500)
    # animation
    animate(zs)
