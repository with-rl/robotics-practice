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


class PythonProjectile:
    def __init__(self):
        self.h = 0.02  # Delta Time
        self.m = 1
        self.g = 9.81
        self.c = 0.47

    # forward kinematics 계산
    def projectile(self, X, t):
        x, x_d, y, y_d = X
        x_dd = -self.c * x_d * np.sqrt(x_d**2 + y_d**2) / self.m
        y_dd = -self.c * y_d * np.sqrt(x_d**2 + y_d**2) / self.m - self.g
        return [x_d, x_dd, y_d, y_dd]


def simulate(simulator, x, y, x_d, y_d, N):
    ts = np.arange(N) * simulator.h
    X0 = np.array([x, x_d, y, y_d])
    Xs = odeint(simulator.projectile, X0, ts, args=())
    return Xs


def animate(Xs):
    for i, X in enumerate(Xs):
        (traj,) = plt.plot(Xs[0:i, 0], Xs[0:i, 2], color="red")
        (prj,) = plt.plot(X[0], X[2], color="red", marker="o")

        plt.xlim(min(Xs[:, 0] - 1), max(Xs[:, 0] + 1))
        plt.ylim(min(Xs[:, 2] - 1), max(Xs[:, 2] + 1))
        plt.gca().set_aspect("equal")

        plt.pause(0.1)
        if i + 1 < len(Xs):
            traj.remove()
            prj.remove()

    plt.pause(5)


if __name__ == "__main__":
    simulator = PythonProjectile()

    # Trajectory by angle
    x, y = 0, 0
    x_d, y_d = 100, 100
    # Trajectory by position
    Xs = simulate(simulator, x, y, x_d, y_d, 101)
    # animation
    animate(Xs)
