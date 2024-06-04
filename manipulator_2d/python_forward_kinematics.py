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


class PythonManipulator2D:
    def __init__(self):
        self.l1 = 1.0
        self.l2 = 1.0

    # forward kinematics 계산
    def forward_kinematics(self, theta1, theta2):
        H_01 = np.array(
            [
                [np.cos(theta1), -np.sin(theta1), 0],
                [np.sin(theta1), np.cos(theta1), 0],
                [0, 0, 1],
            ]
        )
        O_11 = np.array([self.l1, 0, 1])
        O_1 = np.dot(H_01, O_11)

        H_02 = np.array(
            [
                [
                    np.cos(theta1 + theta2),
                    -np.sin(theta1 + theta2),
                    self.l1 * np.cos(theta1),
                ],
                [
                    np.sin(theta1 + theta2),
                    np.cos(theta1 + theta2),
                    self.l1 * np.sin(theta1),
                ],
                [0, 0, 1],
            ]
        )
        E_02 = np.array([self.l2, 0, 1])
        E_0 = np.dot(H_02, E_02)
        return O_1[:2], E_0[:2]

    def jacobian(self, theta1, theta2):
        J_1 = np.array([[-self.l1 * np.sin(theta1), 0], [self.l1 * np.cos(theta1), 0]])
        J_2 = np.array(
            [
                [
                    -self.l1 * np.sin(theta1) - self.l2 * np.sin(theta1 + theta2),
                    -self.l2 * np.sin(theta1 + theta2),
                ],
                [
                    self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2),
                    self.l2 * np.cos(theta1 + theta2),
                ],
            ]
        )
        return J_1, J_2


def simulate(simulator, qs):
    Xs = []
    for i, q in enumerate(qs):
        O_1, E_0 = simulator.forward_kinematics(q[0], q[1])
        Xs.append(np.stack([O_1, E_0]))
    Xs = np.array(Xs)
    return Xs


def animate(Xs, qs):
    for i, X in enumerate(Xs):
        (bar1,) = plt.plot([0, X[0][0]], [0, X[0][1]], linewidth=5, color="r")
        (bar2,) = plt.plot(
            [X[0][0], X[1][0]], [X[0][1], X[1][1]], linewidth=5, color="b"
        )

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect("equal")

        plt.pause(0.02)
        if i + 1 < len(Xs):
            bar1.remove()
            bar2.remove()

    plt.pause(5)
    plt.close()

    # figure control signal
    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(qs[:, 0], "r", label="theta_1")
    plt.plot(qs[:, 1], "b", label="theta_2")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(Xs[:, 1, 0], "r", label="x_e")
    plt.plot(Xs[:, 1, 1], "b", label="y_e")
    plt.legend()

    plt.show()


def create_trajectory():
    qs = np.stack(
        [np.linspace(0, 0.5 * np.pi, 500), np.linspace(0, 1.5 * np.pi, 500)],  #
        axis=1,
    )
    return qs


if __name__ == "__main__":
    simulator = PythonManipulator2D()

    # Trajectory by angle
    qs = create_trajectory()
    # Trajectory by position
    Xs = simulate(simulator, qs)
    # animation
    animate(Xs, qs)
