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
import matplotlib
import matplotlib.pyplot as plt

from python_forward_kinematics import PythonManipulator2D
from python_inverse_kinematics import create_trajectory

sys.path.append("../common")
from mujoco_util import MuJoCoBase

matplotlib.use("Qt5Agg")


class MuJoCoManipulator2D(MuJoCoBase):
    def __init__(self, xml_fn, title):
        super().__init__(xml_fn, title)

        self.pysim = PythonManipulator2D()
        self.Xs = []
        self.qs = []

        # Trajectory by position
        self.q, self.X_refs = create_trajectory(self.pysim)
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
        self.qs.append(data.qpos[:2].copy())
        self.Xs.append(data.site_xpos[0][:2].copy())
        # 두 값 비교
        print(data.site_xpos[0][:2], "==", self.X_refs[self.index])
        # index 증가
        self.index += 1

    def report_cp(self):
        qs = np.array(self.qs)
        Xs = np.array(self.Xs)
        X_refs = np.array(self.X_refs)

        plt.figure(1)

        plt.subplot(3, 1, 1)
        plt.plot(qs[:, 0], "r", label="theta_1")
        plt.plot(qs[:, 1], "b", label="theta_2")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(Xs[:, 0], "r", label="x")
        plt.plot(X_refs[:, 0], "b", label="x_ref")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(Xs[:, 1], "r", label="y")
        plt.plot(X_refs[:, 1], "b", label="y_ref")
        plt.legend()

        plt.show()


if __name__ == "__main__":
    simulator = MuJoCoManipulator2D("manipulator_2d.xml", "Manipulator 2D")
    simulator.init_mujoco()
    simulator.run_mujoco(len(simulator.X_refs))
