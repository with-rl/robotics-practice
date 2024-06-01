# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import sys
import numpy as np

from python_forward_kinematics import PythonManipulator2D

sys.path.append("../common")
from mujoco_util import MuJoCoBase


class MuJoCoManipulator2D(MuJoCoBase):
    def __init__(self, xml_fn, title):
        super().__init__(xml_fn, title)

        self.pysim = PythonManipulator2D()

        # Trajectory by angle
        self.qs = np.stack(
            [np.linspace(0, 0.5 * np.pi, 500), np.linspace(0, 1.5 * np.pi, 500)],  #
            axis=1,
        )
        self.index = 0

    def init_cam(self):
        # initialize camera
        self.cam.azimuth = 90
        self.cam.elevation = -90
        self.cam.distance = 7.5
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    def controller_cb(self, model, data):
        if self.index >= len(self.qs):
            self.index = self.index % len(self.qs)
        data.qpos[0] = self.qs[self.index, 0]
        data.qpos[1] = self.qs[self.index, 1]

    def trace_cb(self, mj, model, data):
        self.index += 1
        # Python을 이용해서 직접 계산
        theta1 = data.qpos[0]
        theta2 = data.qpos[1]
        _, E_0 = self.pysim.forward_kinematics(theta1, theta2)
        # 두 값 출력
        print(data.site_xpos[0][:2], "==", E_0)


if __name__ == "__main__":
    simulator = MuJoCoManipulator2D("manipulator_2d.xml", "Manipulator 2D")
    simulator.init_mujoco()
    simulator.run_mujoco(100)
