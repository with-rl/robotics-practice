# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import platform
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from python_pd_set_point import PythonPendulum1J, create_trajectory

sys.path.append("../common")
from mujoco_util import MuJoCoBase

if platform.system() == "Darwin":
    matplotlib.use("Qt5Agg")


class MuJoCoPendulum1J(MuJoCoBase):
    def __init__(self, xml_fn, title):
        super().__init__(xml_fn, title)

        self.pysim = PythonPendulum1J()

        self.z0, self.z_ref = create_trajectory()
        self.taus = [0]
        self.zs = [self.z0]
        self.Kp = 25
        self.Kd = 10

    def init_cam(self):
        # initialize camera
        self.cam.azimuth = -90
        self.cam.elevation = -10
        self.cam.distance = 7.5
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    def init_controller(self, model, data):
        data.qpos[0] = self.z0[0]

    def controller_cb(self, model, data):
        theta, theta_d = data.qpos[0], data.qvel[0]
        theta_ref, _ = self.z_ref

        tau = -self.Kp * (theta - theta_ref) - self.Kd * theta_d
        tau = np.clip(tau, -10, 10)
        data.ctrl[0] = tau

    def trace_cb(self, mj, model, data):
        self.taus.append(data.ctrl[0])
        self.zs.append([data.qpos[0], data.qvel[0]])
        print(data.qpos[0])

    def report_cp(self):
        taus = np.array(self.taus)
        zs = np.array(self.zs)

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


if __name__ == "__main__":
    simulator = MuJoCoPendulum1J("pendulum_1j.xml", "Pendulum 1 Joint")
    simulator.init_mujoco()
    simulator.run_mujoco(500)
