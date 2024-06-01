# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import mujoco as mj
from mujoco.glfw import glfw


class MuJoCoBase:
    def __init__(self, xml_fn, title):
        self.xml_fn = xml_fn
        self.title = title

        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

    # MuJoCo 환경 초기화
    def init_mujoco(self):
        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(self.xml_fn)  # MuJoCo model
        self.data = mj.MjData(self.model)  # MuJoCo data
        self.cam = mj.MjvCamera()  # Abstract camera
        self.opt = mj.MjvOption()  # visualization options

        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(1200, 900, self.title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # initialize visualization data structures
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # install GLFW mouse and keyboard callbacks
        glfw.set_key_callback(self.window, self.keyboard_cb)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_cb)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_cb)
        glfw.set_scroll_callback(self.window, self.scroll_cb)

        # initialize camera
        self.init_cam()

        # initialize the controller
        self.init_controller(self.model, self.data)

        # set the controller
        mj.set_mjcb_control(self.controller_cb)

    # 카메라 위치 초기화 (카메라 위치를 변경하려면 이 부분을 재 정의 하세요.)
    def init_cam(self):
        # initialize camera
        self.cam.azimuth = 90
        self.cam.elevation = -45
        self.cam.distance = 13
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

    # MuJoCo 시뮬레이션 실행
    def run_mujoco(self, simend, ft=0.1):
        while not glfw.window_should_close(self.window):
            time_prev = self.data.time
            while self.data.time - time_prev < ft:
                mj.mj_step(self.model, self.data)
            self.trace_cb(mj, self.model, self.data)

            if 0 < simend and simend <= self.data.time:
                break

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # create overlay
            overlay = self.create_overlay(self.model, self.data)

            # Update scene and render
            mj.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                None,
                self.cam,
                mj.mjtCatBit.mjCAT_ALL.value,
                self.scene,
            )
            mj.mjr_render(viewport, self.scene, self.context)

            # overlay items
            for gridpos, [t1, t2] in overlay.items():
                mj.mjr_overlay(
                    mj.mjtFontScale.mjFONTSCALE_150,
                    gridpos,
                    viewport,
                    t1,
                    t2,
                    self.context,
                )

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()

    # 프레임 별 MuJoCo 제어 정보 입력 (각 프레임 내에서 한번만 호출 됨)
    def trace_cb(self, mj, model, data):
        pass

    # 초기 MuJoCo 제어 정보 입력
    def init_controller(self, model, data):
        pass

    # 스텝별 MuJoCo 제어 정보 입력 (각 프레임 내에서 여러번 호출 됨)
    def controller_cb(self, model, data):
        pass

    # 키 입력 처리
    def keyboard_cb(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            self.init_controller(self.model, self.data)
            mj.mj_forward(self.model, self.data)

    # 마우스 버튼 클릭 처리
    def mouse_button_cb(self, window, button, act, mods):
        self.button_left = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self.button_middle = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        )
        self.button_right = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )
        # update mouse position
        glfw.get_cursor_pos(window)

    # 마우스 이동 처리
    def mouse_move_cb(self, window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        # no buttons down: nothing to do
        if (
            (not self.button_left)
            and (not self.button_middle)
            and (not self.button_right)
        ):
            return

        # get current window size
        width, height = glfw.get_window_size(window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT

        # determine action based on mouse button
        if self.button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(
            self.model, action, dx / height, dy / height, self.scene, self.cam
        )

    # 마우스 스코롤 처리
    def scroll_cb(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(
            self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam
        )

    # 화면에 정보 출력
    def create_overlay(self, model, data):
        return {}
