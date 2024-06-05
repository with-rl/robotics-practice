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

from python_no_control import PythonPendulum1J, create_trajectory

sys.path.append("../common")
from mujoco_util import MuJoCoBase

matplotlib.use("Qt5Agg")


class MuJoCoPendulum1J(MuJoCoBase):
    def __init__(self, xml_fn, title):
        super().__init__(xml_fn, title)

        self.pysim = PythonPendulum1J()
        self.z0 = create_trajectory()
        self.zs = []

    def init_cam(self):
        # initialize camera
        self.cam.azimuth = -90
        self.cam.elevation = -10
        self.cam.distance = 7.5
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    def init_controller(self, model, data):
        data.qpos[0] = self.z0[0]

    def trace_cb(self, mj, model, data):
        self.zs.append([data.qpos[0], data.qvel[0]])
        print(data.qpos[0])

    def report_cp(self):
        zs = np.array(self.zs)

        plt.figure(1)

        plt.subplot(2, 1, 1)
        plt.plot(zs[:, 0], "r", label="theta")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(zs[:, 1], "r", label="theta_d")
        plt.legend()

        plt.show()


if __name__ == "__main__":
    simulator = MuJoCoPendulum1J("pendulum_1j.xml", "Pendulum 1 Joint")
    simulator.init_mujoco()
    simulator.run_mujoco(500)
