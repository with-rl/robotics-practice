# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import sys
import numpy as np

from python_trajectory_tracking import PythonPendulum1J, create_trajectory

sys.path.append("../common")
from mujoco_util import MuJoCoBase


class MuJoCoPendulum1J(MuJoCoBase):
    def __init__(self, xml_fn, title):
        super().__init__(xml_fn, title)

        self.pysim = PythonPendulum1J()

        self.m = 1
        self.I = 0.1
        self.c = 0.5
        self.l = 1
        self.g = 9.81

        self.z0 = [-np.pi / 2, 0]
        self.ts, self.z_ref = create_trajectory()
        self.Kp = 25
        self.Kd = 2 * np.sqrt(self.Kp)
        self.index = 0

    def init_cam(self):
        # initialize camera
        self.cam.azimuth = -90
        self.cam.elevation = -10
        self.cam.distance = 7.5
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    def init_controller(self, model, data):
        data.qpos[0] = self.z0[0]

    def controller_cb(self, model, data):
        if self.index >= len(self.ts):
            self.index = self.index % len(self.ts)
        theta_ref = self.z_ref[0][self.index]
        theta_d_ref = self.z_ref[1][self.index]
        theta_dd_ref = self.z_ref[2][self.index]

        theta, theta_d = data.qpos[0], data.qvel[0]

        M = 1.0 * self.I + 1.0 * self.c**2 * self.m
        C = 0
        G = self.c * self.g * self.m * np.cos(theta)

        tau = (
            M
            * (
                theta_dd_ref
                - self.Kp * (theta - theta_ref)
                - self.Kd * (theta_d - theta_d_ref)
            )
            + C
            + G
        )
        data.ctrl[0] = tau

    def trace_cb(self, mj, model, data):
        print(data.qpos[0])
        self.index += 1


if __name__ == "__main__":
    simulator = MuJoCoPendulum1J("pendulum_1j.xml", "Pendulum 1 Joint")
    simulator.init_mujoco()
    simulator.run_mujoco(len(simulator.ts))
