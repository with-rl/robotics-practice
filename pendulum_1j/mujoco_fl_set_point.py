# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import sys
import numpy as np

from python_pd_set_point import PythonPendulum1J

sys.path.append("../common")
from mujoco_util import MuJoCoBase


class MuJoCoPendulum1J(MuJoCoBase):
    def __init__(self, xml_fn, title):
        super().__init__(xml_fn, title)

        self.pysim = PythonPendulum1J()

        self.z_ref = [np.pi, 0]
        self.Kp = 25
        self.Kd = 2 * np.sqrt(self.Kp)

    def init_cam(self):
        # initialize camera
        self.cam.azimuth = -90
        self.cam.elevation = -10
        self.cam.distance = 15
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    def init_controller(self, model, data):
        data.qpos[0] = 0

    def controller_cb(self, model, data):
        theta1, theta1_d = data.qpos[0], data.qvel[0]
        theta1_ref, _ = self.z_ref
        # huristic M, C, G
        M_hat = 2.5
        C_hat = 0.1
        G_hat = 0.5

        tau = (
            M_hat * (-self.Kp * (theta1 - theta1_ref) - self.Kd * theta1_d)
            + C_hat * (theta1_d)
            + G_hat * (theta1)
        )
        tau = np.clip(tau, -10, 10)
        data.ctrl[0] = tau

    def trace_cb(self, mj, model, data):
        print(data.qpos[0])


if __name__ == "__main__":
    simulator = MuJoCoPendulum1J("pendulum_1j.xml", "Pendulum 1 Joint")
    simulator.init_mujoco()
    simulator.run_mujoco(500)
