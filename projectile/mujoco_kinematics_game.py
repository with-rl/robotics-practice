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
from mujoco.glfw import glfw

from python_kinematics import PythonProjectile

sys.path.append("../common")
from mujoco_util import MuJoCoBase


class MuJoCoProjectile(MuJoCoBase):
    def __init__(self, xml_fn, title):
        super().__init__(xml_fn, title)

        self.pysim = PythonProjectile()

        self.vel_x = 0
        self.vel_z = 0

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
        data.qvel[0] = 0
        data.qvel[1] = 0
        data.qvel[2] = 0

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

    def keyboard_cb(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_R:
            mj.mj_resetData(self.model, self.data)
            self.init_controller(self.model, self.data)
            mj.mj_forward(self.model, self.data)

        if act == glfw.PRESS and key == glfw.KEY_S:
            self.data.qvel[0] = self.vel_x
            self.data.qvel[2] = self.vel_z

        if act == glfw.PRESS and key == glfw.KEY_UP:
            self.vel_z += 2

        if act == glfw.PRESS and key == glfw.KEY_DOWN:
            self.vel_z -= 2

        if act == glfw.PRESS and key == glfw.KEY_LEFT:
            self.vel_x -= 2

        if act == glfw.PRESS and key == glfw.KEY_RIGHT:
            self.vel_x += 2

    def create_overlay(self, model, data):
        topleft = ["", ""]
        bottomleft = ["", ""]

        bottomleft[0] += "Restart\n"
        bottomleft[1] += "r\n"

        bottomleft[0] += "Start\n"
        bottomleft[1] += "s\n"

        bottomleft[0] += "Time\n"
        bottomleft[1] += f"{data.time:.2f}\n"

        topleft[0] += "vel x (left/right)\n"
        topleft[1] += f"{self.vel_x:.2f}\n"

        topleft[0] += "vel z (up/down)\n"
        topleft[1] += f"{self.vel_z:.2f}\n"

        return {
            mj.mjtGridPos.mjGRID_TOPLEFT: topleft,
            mj.mjtGridPos.mjGRID_BOTTOMLEFT: bottomleft,
        }


if __name__ == "__main__":
    simulator = MuJoCoProjectile("projectile.xml", "Projectile 2D")
    simulator.init_mujoco()
    simulator.run_mujoco(10**10)
