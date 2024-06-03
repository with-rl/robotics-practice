# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import sys
import numpy as np

from python_kinematics import PythonProjectile

sys.path.append("../common")
from mujoco_util import MuJoCoBase


class MuJoCoProjectile(MuJoCoBase):
    def __init__(self, xml_fn, title):
        super().__init__(xml_fn, title)

        self.pysim = PythonProjectile()

    def init_cam(self):
        # initialize camera
        self.cam.azimuth = 90
        self.cam.elevation = -45
        self.cam.distance = 20
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    def init_controller(self, model, data):
        data.qpos[0] = 0
        data.qpos[1] = 0
        data.qpos[2] = 0.1
        data.qvel[0] = 10
        data.qvel[1] = 0
        data.qvel[2] = 10

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
        # 값 출력
        print(data.qvel[:3])


if __name__ == "__main__":
    simulator = MuJoCoProjectile("projectile.xml", "Projectile 2D")
    simulator.init_mujoco()
    simulator.run_mujoco(100)
