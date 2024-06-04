# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import sys
import numpy as np

from python_no_control import PythonPendulum1J

sys.path.append("../common")
from mujoco_util import MuJoCoBase


class MuJoCoPendulum1J(MuJoCoBase):
    def __init__(self, xml_fn, title):
        super().__init__(xml_fn, title)

        self.pysim = PythonPendulum1J()
        self.z0 = [np.pi / 2 - 0.5, 0]

    def init_cam(self):
        # initialize camera
        self.cam.azimuth = -90
        self.cam.elevation = -10
        self.cam.distance = 7.5
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    def init_controller(self, model, data):
        data.qpos[0] = self.z0[0]

    def trace_cb(self, mj, model, data):
        print(data.qpos[0])


if __name__ == "__main__":
    simulator = MuJoCoPendulum1J("pendulum_1j.xml", "Pendulum 1 Joint")
    simulator.init_mujoco()
    simulator.run_mujoco(500)
