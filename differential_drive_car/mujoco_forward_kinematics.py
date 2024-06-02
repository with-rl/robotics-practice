# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import sys
import numpy as np

from python_forward_kinematics import PythonDDCar

sys.path.append("../common")
from mujoco_util import MuJoCoBase


class MuJoCoDDCar(MuJoCoBase):
    def __init__(self, xml_fn, title):
        super().__init__(xml_fn, title)

        self.pysim = PythonDDCar()

        # Trajectory by velocity and angular velocity
        self.us = np.zeros((300, 2))
        for i in range(0, 100):
            self.us[i, 0] = 2.0
            self.us[i, 1] = 0
        for i in range(100, 200):
            self.us[i, 0] = 2.0
            self.us[i, 1] = np.pi / 2
        for i in range(200, 300):
            self.us[i, 0] = 2.0
            self.us[i, 1] = 0
        self.index = 0

    def init_cam(self):
        # initialize camera
        self.cam.azimuth = 90
        self.cam.elevation = -90
        self.cam.distance = 20
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    def controller_cb(self, model, data):
        if self.index >= len(self.us):
            self.index = self.index % len(self.us)

        ctrl = self.pysim.to_control(self.us[self.index][0], self.us[self.index][1])
        ctrl = np.clip(ctrl, -5, 5)
        data.ctrl[0] = ctrl[1]
        data.ctrl[1] = ctrl[0]

    def trace_cb(self, mj, model, data):
        # 두 값 출력
        print(self.us[self.index], "==", data.ctrl)
        self.index += 1


if __name__ == "__main__":
    simulator = MuJoCoDDCar("differential_drive_car.xml", "Differential Drive Car")
    simulator.init_mujoco()
    simulator.run_mujoco(len(simulator.us))
