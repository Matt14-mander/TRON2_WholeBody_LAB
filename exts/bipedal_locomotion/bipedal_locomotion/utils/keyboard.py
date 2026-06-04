# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(2) control."""

import numpy as np
import weakref
from collections.abc import Callable
import torch
import carb
import omni
import isaaclab.sim as sim_utils

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.devices.device_base import DeviceBase


class Keyboard(DeviceBase):

    def __init__(self, env: ManagerBasedRLEnv):
        """Initialize the keyboard layer."""
        self.env = env
        self.device = self.env.device
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # dictionary for additional callbacks
        self._additional_callbacks = dict()
        self.lookat_vec = torch.tensor([-0, 2, 1], requires_grad=False, device=self.device)
        self.fixed_cam = False
        self.key = ""
        if not hasattr(self.env, "look_at_id"):
            setattr(self.env, "look_at_id", torch.tensor(0, device=self.device))
        self.last_look_at_id = self.env.look_at_id.clone()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for ManagerBasedRLEnv: {self.__class__.__name__}\n"
        return msg

    """
    Operations
    """

    def reset(self):
        pass

    def add_callback(self, key: str, func: Callable):
        pass

    def advance(self):
        pass

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self.key = event.input.name
                if event.input.name == "R":
                    self.env.episode_length_buf = torch.ones_like(self.env.episode_length_buf) * 1e6
                if event.input.name == "LEFT_BRACKET":
                    self.env.look_at_id  = (self.env.look_at_id-1) % self.env.num_envs
                    self.look_at()
                if event.input.name == "RIGHT_BRACKET":
                    self.env.look_at_id  = (self.env.look_at_id+1) % self.env.num_envs
                    self.look_at()
                if event.input.name == "SLASH":
                    self.look_at()
                if event.input.name == "PERIOD":
                    self.fixed_cam = not self.fixed_cam
        # since no error, we are fine :)
        return True
    
    def look_at(self):
        if self.fixed_cam:
            return
        # 获取机器人资产
        robot = self.env.scene["robot"]

        # 检查是否切换了机器人 ID
        is_id_changed = (self.last_look_at_id != self.env.look_at_id)
        self.last_look_at_id = self.env.look_at_id.clone()
        
        # 获取当前目标位置 (root 位置)
        look_at_pos = robot.data.root_pos_w[self.env.look_at_id, :3].clone()

        # 只有在 ID 没变时，才捕获用户的手动视角调整
        if not is_id_changed:
            try:
                from pxr import UsdGeom
                stage = sim_utils.get_current_stage()
                if stage is not None:
                    camera_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
                    if camera_prim and camera_prim.IsValid():
                        camera_xform = UsdGeom.Xformable(camera_prim)
                        world_transform = camera_xform.ComputeLocalToWorldTransform(0)
                        p = world_transform.ExtractTranslation()
                        cam_trans = torch.tensor([p[0], p[1], p[2]], requires_grad=False, device=self.device)
                        
                        # 获取当前相机的目标点 (lookat)
                        # 注意：UsdGeom 无法直接简单获取 lookat 点，我们假设用户调整的是相对偏移
                        # 如果相机位置发生了变化，更新相对偏移向量
                        self.lookat_vec = cam_trans - look_at_pos
            except Exception:
                pass
        
        # 计算新相机位置 (保持相对偏移量)
        cam_pos = look_at_pos + self.lookat_vec
        
        # 转换为列表
        cam_pos_list = [float(x) for x in cam_pos.detach().cpu().numpy().reshape(-1)]
        look_at_pos_list = [float(x) for x in look_at_pos.detach().cpu().numpy().reshape(-1)]
        
        eye = (cam_pos_list[0], cam_pos_list[1], cam_pos_list[2])
        target = (look_at_pos_list[0], look_at_pos_list[1], look_at_pos_list[2])
        
        # 关键：使用更通用的 API 设置视口
        try:
            from isaaclab.utils.viewport import set_camera_view
            set_camera_view(eye, target)
        except ImportError:
            # 回退到 sim 接口
            self.env.sim.set_camera_view(eye, target)

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # forward command
            "R": "reset envs",
            "LEFT_BRACKET": "prev_id,[",
            "RIGHT_BRACKET": "next_id,]",
            "SLASH": "lookat,/",
            "PERIOD": "fixed_cam,."
        }