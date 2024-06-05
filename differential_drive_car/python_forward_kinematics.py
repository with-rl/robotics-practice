# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import platform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if platform.system() == "Darwin":
    matplotlib.use("TkAgg")


class PythonDDCar:
    def __init__(self, Kp=10):
        self.h = 0.02  # Delta Time
        self.r = 0.2
        self.b = 0.3
        # inverse kinematics
        self.Kp = Kp
        self.px = 0.05
        self.py = 0.0

    # forward kinematics 계산
    def euler_integration(self, vel, omega, X0):
        x0, y0, theta0 = X0
        xdot_c = vel * np.cos(theta0)
        ydot_c = vel * np.sin(theta0)

        x1 = x0 + xdot_c * self.h
        y1 = y0 + ydot_c * self.h
        theta1 = theta0 + omega * self.h

        return [x1, y1, theta1]

    def to_control(self, vel, omega):
        A = np.array(
            [[self.r / 2, self.r / 2], [self.r / (2 * self.b), -self.r / (2 * self.b)]]
        )
        b = np.array([vel, omega])
        x = np.linalg.solve(A, b)
        print(b, x)
        return x

    def inverse_kinematics(self, X_ref, X0):
        x0, y0, theta0 = X0
        # calculate posion of p
        H_01 = np.array(
            [
                [np.cos(theta0), -np.sin(theta0), x0],
                [np.sin(theta0), np.cos(theta0), y0],
                [0, 0, 1],
            ]
        )
        Xp = H_01 @ np.array([self.px, self.py, 1])
        # p control
        error = np.array([X_ref[0] - Xp[0], X_ref[1] - Xp[1]])
        b = self.Kp * error
        A = np.array(
            [
                [np.cos(theta0), -self.px * np.sin(theta0) - self.py * np.cos(theta0)],
                [np.sin(theta0), self.px * np.cos(theta0) - self.py * np.sin(theta0)],
            ]
        )
        A_inv = np.linalg.inv(A)
        u = A_inv @ b.T
        return u


def simulate(simulator, us):
    X0 = [0, 0, 0]
    Xs = [X0]  # 초기 x, y, theta
    for i, u in enumerate(us):
        X1 = simulator.euler_integration(u[0], u[1], X0)
        Xs.append(X1)
        X0 = X1
    Xs = np.array(Xs)
    return Xs


def animate(Xs, us):
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

    plt.subplot(1, 1, 1)
    plt.plot(us[:, 0], "r", label="vel")
    plt.plot(us[:, 1], "b", label="omega")
    plt.legend()

    plt.show()


def create_trajectory():
    us = np.zeros((600, 2))
    for i in range(0, 200):
        us[i, 0] = 1.0
        us[i, 1] = 0
    for i in range(200, 400):
        us[i, 0] = 0.5
        us[i, 1] = np.pi / 4
    for i in range(400, 600):
        us[i, 0] = 1.0
        us[i, 1] = 0
    return us


if __name__ == "__main__":
    simulator = PythonDDCar()

    # Trajectory by velocity and angular velocity
    us = create_trajectory()
    # Trajectory by position
    Xs = simulate(simulator, us)
    # animation
    animate(Xs, us)
