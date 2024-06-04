# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from python_forward_kinematics import PythonDDCar

matplotlib.use("TkAgg")


def simulate(simulator, X_refs):
    X0 = [X_refs[0][0], X_refs[0][1], 0]
    us, Xs = [], []
    for i, X_ref in enumerate(X_refs):
        u = simulator.inverse_kinematics(X_ref, X0)
        np.clip(u, -1, 1)
        X1 = simulator.euler_integration(u[0], u[1], X0)
        us.append(u)
        Xs.append(X1)
        X0 = X1
    us = np.array(us)
    Xs = np.array(Xs)
    return us, Xs


def animate(Xs, X_refs, us):
    R = 0.75
    for i, X in enumerate(Xs):
        x, y, theta = X

        x2 = x + R * np.cos(theta)
        y2 = y + R * np.sin(theta)

        (robot,) = plt.plot(x, y, color="green", marker="o", markersize=15)
        (line,) = plt.plot([x, x2], [y, y2], color="black")
        (shape,) = plt.plot(Xs[0:i, 0], Xs[0:i, 1], color="red")

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.gca().set_aspect("equal")

        plt.pause(0.02)
        if i + 1 < len(Xs):
            robot.remove()
            line.remove()
            shape.remove()

    plt.pause(5)
    plt.close()

    # figure control signal
    plt.figure(1)

    plt.subplot(3, 1, 1)
    plt.plot(Xs[:, 0], "r", label="x")
    plt.plot(X_refs[:, 0], "b", label="x_ref")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(Xs[:, 1], "r", label="y")
    plt.plot(X_refs[:, 1], "b", label="y_ref")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(us[:, 0], "r", label="vel")
    plt.plot(us[:, 1], "b", label="omega")
    plt.legend()

    plt.show()


def create_trajectory():
    r = 4.0
    phi = np.linspace(0, 2 * np.pi, 1001) - np.pi / 2
    X_refs = np.stack([r * np.cos(phi), r * np.sin(phi) + r], axis=-1)
    return X_refs


if __name__ == "__main__":
    simulator = PythonDDCar()

    # Trajectory by position
    X_refs = create_trajectory()

    # Trajectory by velocity and angular velocity
    us, Xs = simulate(simulator, X_refs)

    # animation
    animate(Xs, X_refs, us)
