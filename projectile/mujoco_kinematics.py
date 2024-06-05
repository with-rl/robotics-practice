# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from python_kinematics import PythonProjectile, create_trajectory

sys.path.append("../common")
from mujoco_util import MuJoCoBase

matplotlib.use("Qt5Agg")


class MuJoCoProjectile(MuJoCoBase):
    def __init__(self, xml_fn, title):
        super().__init__(xml_fn, title)

        self.pysim = PythonProjectile()
        self.X0 = create_trajectory()
        self.Xs = []

    def init_cam(self):
        # initialize camera
        self.cam.azimuth = 90
        self.cam.elevation = -10
        self.cam.distance = 20
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    def init_controller(self, model, data):
        data.qpos[0] = self.X0[0]
        data.qpos[1] = 0
        data.qpos[2] = self.X0[2]
        data.qvel[0] = self.X0[1]
        data.qvel[1] = 0
        data.qvel[2] = self.X0[3]

    def controller_cb(self, model, data):
        vx = data.qvel[0]
        vy = data.qvel[1]
        vz = data.qvel[2]
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        c = self.pysim.c

        data.xfrc_applied[1][0] = -c * vx * v
        data.xfrc_applied[1][1] = -c * vy * v
        data.xfrc_applied[1][2] = -c * vz * v

    def trace_cb(self, mj, model, data):
        self.Xs.append([data.qpos[0], data.qvel[0], data.qpos[2], data.qvel[2]])
        # 값 출력
        print(data.qvel[:3])

    def report_cp(self):
        Xs = np.array(self.Xs)

        plt.figure(1)

        plt.subplot(2, 1, 1)
        plt.plot(Xs[:, 0], "r", label="x")
        plt.plot(Xs[:, 2], "b", label="y")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(Xs[:, 1], "r", label="x_d")
        plt.plot(Xs[:, 3], "b", label="y_d")
        plt.legend()

        plt.show()


if __name__ == "__main__":
    simulator = MuJoCoProjectile("projectile.xml", "Projectile 2D")
    simulator.init_mujoco()
    simulator.run_mujoco(500)
