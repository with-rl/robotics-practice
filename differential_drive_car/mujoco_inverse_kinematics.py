# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import sys
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R

from python_forward_kinematics import PythonDDCar

sys.path.append("../common")
from mujoco_util import MuJoCoBase


class MuJoCoDDCar(MuJoCoBase):
    def __init__(self, xml_fn, title):
        super().__init__(xml_fn, title)

        self.pysim = PythonDDCar()

        # Trajectory by position
        r = 5.0
        phi = np.linspace(0, 2 * np.pi, 1000) - np.pi / 2
        self.X_refs = np.stack([r * np.cos(phi), r * np.sin(phi) + r], axis=-1)
        self.index = 0

    def init_cam(self):
        # initialize camera
        self.cam.azimuth = 90
        self.cam.elevation = -90
        self.cam.distance = 20
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    def controller_cb(self, model, data):
        if self.index >= len(self.X_refs):
            self.index = self.index % len(self.X_refs)

        angle = R.from_matrix(data.site_xmat[0].reshape(3, 3)).as_euler("zyx")
        X0 = [data.site_xpos[0][0], data.site_xpos[0][1], angle[0]]
        u = self.pysim.inverse_kinematics(self.X_refs[self.index], X0)

        ctrl = self.pysim.to_control(u[0], u[1])
        ctrl = np.clip(ctrl, -5, 5)
        data.ctrl[0] = ctrl[1]
        data.ctrl[1] = ctrl[0]

    def trace_cb(self, mj, model, data):
        # 두 값 출력
        print(data.site_xpos[0][:2], "==", self.X_refs[self.index])
        self.index += 1


if __name__ == "__main__":
    simulator = MuJoCoDDCar("differential_drive_car.xml", "Differential Drive Car")
    simulator.init_mujoco()
    simulator.run_mujoco(len(simulator.X_refs))
