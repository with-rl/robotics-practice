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

matplotlib.use("TkAgg")

from python_forward_kinematics import PythonManipulator2D


def simulate(simulator, q, X_refs):
    Xs, qs = [], []

    O_1, E_0 = simulator.forward_kinematics(q[0], q[1])
    Xs.append(np.stack([O_1, E_0]))
    qs.append(q.copy())

    for i, X_ref in enumerate(X_refs):
        _, J_2 = simulator.jacobian(q[0], q[1])
        J_2inv = np.linalg.inv(J_2)

        dX = np.array([X_ref[0] - E_0[0], X_ref[1] - E_0[1]])

        dq = J_2inv.dot(dX)
        dq = np.clip(dq, -5, 5)
        q[0] += dq[0]
        q[1] += dq[1]

        O_1, E_0 = simulator.forward_kinematics(q[0], q[1])
        Xs.append(np.stack([O_1, E_0]))
        qs.append(q.copy())

    Xs = np.array(Xs)
    qs = np.array(qs)
    return Xs, qs


def animate(Xs, X_refs, qs):
    for i, X in enumerate(Xs):
        (bar1,) = plt.plot([0, X[0][0]], [0, X[0][1]], linewidth=5, color="r")
        (bar2,) = plt.plot(
            [X[0][0], X[1][0]], [X[0][1], X[1][1]], linewidth=5, color="b"
        )
        (shape,) = plt.plot(Xs[0:i, 1, 0], Xs[0:i, 1, 1], "k.")

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect("equal")

        plt.pause(0.02)
        if i + 1 < len(Xs):
            bar1.remove()
            bar2.remove()
            shape.remove()

    plt.pause(5)
    plt.close()

    # figure control signal
    plt.figure(1)

    plt.subplot(3, 1, 1)
    plt.plot(qs[:, 0], "r", label="theta_1")
    plt.plot(qs[:, 1], "b", label="theta_2")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(Xs[:, 1, 0], "r", label="x")
    plt.plot(X_refs[:, 0], "b", label="x_ref")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(Xs[:, 1, 1], "r", label="y")
    plt.plot(X_refs[:, 1], "b", label="y_ref")
    plt.legend()

    plt.show()


def create_trajectory(simulator):
    q = np.array([np.pi / 2, np.pi / 2])
    r = 0.5

    _, E0 = simulator.forward_kinematics(q[0], q[0])
    phi = np.linspace(0, 2 * np.pi, 500)
    X_refs = np.stack([E0[0] + r * np.cos(phi), E0[1] + r * np.sin(phi)], axis=-1)
    return q, X_refs


if __name__ == "__main__":
    simulator = PythonManipulator2D()

    q, X_refs = create_trajectory(simulator)

    Xs, qs = simulate(simulator, q, X_refs)
    # animation
    animate(Xs, X_refs, qs)
