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


def animate(Xs):
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


if __name__ == "__main__":
    simulator = PythonDDCar()

    # Trajectory by position
    r = 5.0
    phi = np.linspace(0, 2 * np.pi, 1000) - np.pi / 2
    X_refs = np.stack([r * np.cos(phi), r * np.sin(phi) + r], axis=-1)

    # Trajectory by velocity and angular velocity
    us, Xs = simulate(simulator, X_refs)

    # animation
    animate(Xs)
