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

from python_forward_kinematics import PythonManipulator2D

sys.path.append("../common")
from mujoco_util import MuJoCoBase


class MuJoCoManipulator2D(MuJoCoBase):
    def __init__(self, xml_fn, title):
        super().__init__(xml_fn, title)

        self.pysim = PythonManipulator2D()

        self.q = np.array([np.pi / 2, np.pi / 2])
        r = 0.5

        # Trajectory by position
        _, E0 = self.pysim.forward_kinematics(self.q[0], self.q[0])
        phi = np.linspace(0, 2 * np.pi, 500)
        self.X_refs = np.stack(
            [E0[0] + r * np.cos(phi), E0[1] + r * np.sin(phi)], axis=-1
        )

        self.index = 0

    def init_cam(self):
        self.cam.azimuth = 90
        self.cam.elevation = -90
        self.cam.distance = 7.5
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    # 초기 MuJoCo 제어 정보 입력
    def init_controller(self, model, data):
        data.qpos[0] = self.q[0]
        data.qpos[1] = self.q[1]

    def controller_cb(self, model, data):
        if self.index >= len(self.X_refs):
            self.index = self.index % len(self.X_refs)
        E_0 = data.site_xpos[0]

        jacp = np.zeros((3, 2))
        mj.mj_jac(model, data, jacp, None, E_0, 2)
        J_2 = jacp[[0, 1], :]
        J_2inv = np.linalg.inv(J_2)

        dX = np.array(
            [
                self.X_refs[self.index][0] - E_0[0],
                self.X_refs[self.index][1] - E_0[1],
            ]
        )
        dq = J_2inv.dot(dX)
        dq = np.clip(dq, -5, 5)

        data.qpos[0] += dq[0]
        data.qpos[1] += dq[1]

    def trace_cb(self, mj, model, data):
        # 두 값 비교
        print(data.site_xpos[0][:2], "==", self.X_refs[self.index])
        # index 증가
        self.index += 1


if __name__ == "__main__":
    simulator = MuJoCoManipulator2D("manipulator_2d.xml", "Manipulator 2D")
    simulator.init_mujoco()
    simulator.run_mujoco(len(simulator.X_refs))
