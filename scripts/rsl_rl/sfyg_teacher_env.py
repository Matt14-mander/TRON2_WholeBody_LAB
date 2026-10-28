"""Isolated Isaac T0/T1 adapter over the unchanged SFYG carry configuration."""
from dataclasses import asdict
import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import DistantLightCfg
from isaaclab.managers import (ActionTerm, ActionTermCfg, ObservationGroupCfg,
    ObservationTermCfg, RewardTermCfg, TerminationTermCfg, SceneEntityCfg)
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from bipedal_locomotion.tasks.locomotion import mdp as original
from bipedal_locomotion.tasks.locomotion.robots.limx_sfyg_carry_env_cfg import SFYGCarryFlatEnvCfg
from sfyg_teacher_core import (Settings,History,envelope,sample_force,process_action,
    low_observation,reward_terms,frozen_low)


def controller(env):
    return env.action_manager.get_term('transport')


class TeacherAction(ActionTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.settings = Settings(**env.cfg.teacher_settings).validate()
        self.low,self.low_cfg,self.low_sha,self.low_saved = frozen_low(cfg.low_path,self.device)
        self.leg_ids,names = self._asset.find_joints(self.low_cfg['leg_names'],preserve_order=True)
        if names != self.low_cfg['leg_names']:
            raise ValueError('Leg joint order differs from the frozen policy')
        self.hold_ids,_ = self._asset.find_joints(self.low_cfg['names'][10:],preserve_order=True)
        self.arm_ids = self.hold_ids[:6]
        self.body_ids = [self._asset.find_bodies(name)[0][0]
                         for name in ('gripper_base_Link','paint_payload_Link')]
        self.payload_id = self.body_ids[1]
        defaults = self._asset.data.default_joint_pos[:,self.leg_ids+self.hold_ids]
        torch.testing.assert_close(defaults,defaults.new_tensor(self.low_cfg['defaults']).expand_as(defaults))
        self.nominal = defaults[:,10:].clone()
        self.raw = torch.zeros(self.num_envs,9,device=self.device)
        self.previous = torch.zeros_like(self.raw)
        self.command = torch.zeros(self.num_envs,3,device=self.device)
        self.last_leg = torch.zeros(self.num_envs,10,device=self.device)
        self.leg_target = defaults[:,:10].clone()
        self.low_history = History(self.num_envs,10,42,self.device,repeat_first=True)
        self.arm_history = History(self.num_envs,8,18,self.device)
        self.target_force = torch.zeros(self.num_envs,device=self.device)
        self.interval_reference = torch.zeros_like(self.command)
        self.world_force = torch.zeros(self.num_envs,1,3,device=self.device)
        self.local_force = torch.zeros_like(self.world_force)
        self.zero_torque = torch.zeros_like(self.world_force)
        self.low_calls = 0

    @property
    def action_dim(self): return 9

    @property
    def raw_actions(self): return self.raw

    @property
    def processed_actions(self):
        return torch.cat((self.command,self.nominal[:,:6]),-1)

    def time(self):
        return self._env.episode_length_buf*self._env.step_dt

    def reference(self):
        ref = torch.zeros_like(self.command)
        ref[:,0] = self.target_force*envelope(self.time(),self.settings)/40.
        return ref

    @torch.no_grad()
    def process_actions(self,actions):
        self.previous.copy_(self.raw)
        self.raw.copy_(actions)
        self.command.copy_(process_action(actions,self.time(),self.settings))
        self.interval_reference.copy_(self.reference())
        data = self._asset.data
        obs = low_observation(data.root_ang_vel_b,data.projected_gravity_b,
            data.joint_pos[:,self.leg_ids],data.joint_vel[:,self.leg_ids],self.last_leg,
            self._env.episode_length_buf,self.low_cfg)
        history = self.low_history.append(obs,self._env.common_step_counter)
        output = self.low(obs,history,self.command)
        if output.shape != self.last_leg.shape or not torch.isfinite(output).all():
            raise FloatingPointError('Invalid frozen SFYG leg action')
        self.last_leg.copy_(output)
        self.leg_target.copy_(data.default_joint_pos[:,self.leg_ids]+.25*output)
        self.low_calls += 1

    def apply_actions(self):
        self._asset.set_joint_position_target(self.leg_target,joint_ids=self.leg_ids)
        self._asset.set_joint_position_target(self.nominal,joint_ids=self.hold_ids)
        self._asset.set_joint_velocity_target(torch.zeros_like(self.nominal),joint_ids=self.hold_ids)
        self._asset.set_joint_effort_target(torch.zeros_like(self.nominal),joint_ids=self.hold_ids)
        force_base = self.interval_reference*40.
        self.world_force[:,0] = quat_apply(self._asset.data.root_quat_w,force_base)
        self.local_force[:,0] = quat_apply_inverse(self._asset.data.body_quat_w[:,self.payload_id],self.world_force[:,0])
        self._asset.set_external_force_and_torque(self.local_force,self.zero_torque,
            body_ids=[self.payload_id],is_global=False)

    def reset(self,env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.raw,self.previous,self.command,self.last_leg,self.interval_reference,
                      self.world_force,self.local_force):
            value[ids] = 0
        self.leg_target[ids] = self._asset.data.default_joint_pos[ids][:,self.leg_ids]
        for history in (self.low_history,self.arm_history): history.reset(ids)
        n = self.num_envs if env_ids is None else len(env_ids)
        self.target_force[ids] = sample_force(n,self.device,self.settings)
        if self._env.cfg.fixed_force is not None:
            self.target_force[ids] = self._env.cfg.fixed_force
        # Clear the actual simulator buffer, not just the diagnostic tensor.
        self._asset.set_external_force_and_torque(self.local_force[ids],self.zero_torque[ids],
            body_ids=[self.payload_id],env_ids=env_ids,is_global=False)

    def history(self):
        data = self._asset.data
        frame = torch.cat((data.joint_pos[:,self.arm_ids]-self.nominal[:,:6],
            data.joint_vel[:,self.arm_ids],torch.zeros_like(self.nominal[:,:6])),-1)
        return self.arm_history.append(frame,self._env.common_step_counter,
            valid=self._env.episode_length_buf>0).flatten(1)

    def actual(self):
        data = self._asset.data
        return torch.cat((data.root_lin_vel_b[:,:2],data.root_ang_vel_b[:,2:3]),-1)

    def costs(self):
        return reward_terms(self.actual(),self.interval_reference,self.command,
                            self.raw,self.previous,self._asset.data.projected_gravity_b)


@configclass
class TeacherActionCfg(ActionTermCfg):
    class_type: type = TeacherAction
    asset_name: str = 'robot'
    low_path: str = ''


def teacher_observation(env):
    term = controller(env)
    return torch.cat((term.history(),term.reference()),-1)


def critic_observation(env):
    term = controller(env)
    data = term._asset.data
    values = [teacher_observation(env)]
    for body in term.body_ids:
        vectors = (data.body_com_pos_w[:,body]-data.root_pos_w,
            data.body_com_lin_vel_w[:,body],data.body_com_ang_vel_w[:,body],
            data.body_com_lin_acc_w[:,body],data.body_com_ang_acc_w[:,body])
        values.extend(quat_apply_inverse(data.root_quat_w,value) for value in vectors)
    return torch.cat(values,-1)


def empty_commands(env): return torch.zeros(env.num_envs,0,device=env.device)


def compatibility_history(env):
    # Original runner requires history width = actor width, but its estimator
    # is disabled (output=0). This buffer never enters the frozen LL.
    return torch.zeros(env.num_envs,147,device=env.device)


def cost(env,name): return controller(env).costs()[name]


def failure(env): return env.termination_manager.terminated.float()


def fall(env):
    data = env.scene['robot'].data
    return ((-data.projected_gravity_b[:,2] < math.cos(math.radians(35)))
            | (data.root_pos_w[:,2]-env.scene.env_origins[:,2] < .45))


@configclass
class Actions:
    transport = TeacherActionCfg()


@configclass
class Commands:
    pass


@configclass
class Observations:
    @configclass
    class Policy(ObservationGroupCfg):
        state = ObservationTermCfg(func=teacher_observation)
        enable_corruption = False
    @configclass
    class Critic(ObservationGroupCfg):
        state = ObservationTermCfg(func=critic_observation)
        enable_corruption = False
    @configclass
    class Empty(ObservationGroupCfg):
        state = ObservationTermCfg(func=empty_commands)
    @configclass
    class History(ObservationGroupCfg):
        state = ObservationTermCfg(func=compatibility_history)
    policy,critic,commands,obsHistory = Policy(),Critic(),Empty(),History()


@configclass
class Rewards:
    linear_tracking = RewardTermCfg(func=cost,weight=1.,params={'name':'linear_tracking'})
    yaw_tracking = RewardTermCfg(func=cost,weight=1.,params={'name':'yaw_tracking'})
    command_tracking = RewardTermCfg(func=cost,weight=1.,params={'name':'command_tracking'})
    base_tilt = RewardTermCfg(func=cost,weight=1.,params={'name':'base_tilt'})
    raw_action = RewardTermCfg(func=cost,weight=1.,params={'name':'raw_action'})
    failure = RewardTermCfg(func=failure,weight=-2500.)  # RewardManager multiplies dt exactly once


@configclass
class Terminations:
    time_out = TerminationTermCfg(func=original.time_out,time_out=True)
    base_contact = TerminationTermCfg(func=original.illegal_contact,params={
        'sensor_cfg':SceneEntityCfg('contact_forces',body_names=['base_Link','proximal_pitch_[RL]_Link']),
        'threshold':1.})
    fall = TerminationTermCfg(func=fall)
    arm_payload_contact = TerminationTermCfg(func=original.illegal_contact,params={
        'sensor_cfg':SceneEntityCfg('contact_forces',body_names=['arm[1-6]_Link','gripper.*_Link','paint_payload_Link']),
        'threshold':1.})


@configclass
class TeacherEnvCfg(SFYGCarryFlatEnvCfg):
    teacher_settings: dict = asdict(Settings())
    fixed_force: float | None = None

    def __post_init__(self):
        super().__post_init__()
        self.actions,self.commands,self.observations = Actions(),Commands(),Observations()
        self.rewards,self.terminations = Rewards(),Terminations()
        self.episode_length_s = Settings(**self.teacher_settings).horizon
        self.scene.num_envs = 512
        self.scene.terrain.visual_material = None
        self.scene.light.spawn.texture_file = None
        self.scene.teacher_light = AssetBaseCfg(prim_path='/World/TeacherLight',
            spawn=DistantLightCfg(intensity=3000.,color=(1.,1.,1.)))
        self.sim.physx.enable_external_forces_every_iteration = True
        self.viewer.eye,self.viewer.lookat = (2.5,2.5,1.8),(0.,0.,.7)
        for field in ('pose_range','velocity_range'):
            self.events.reset_robot_base.params[field] = {
                key:(0.,0.) for key in self.events.reset_robot_base.params[field]}
        self.events.reset_robot_joints.params.update(position_range=(0.,0.),velocity_range=(0.,0.))
        for actuator in self.scene.robot.actuators.values():
            if hasattr(actuator,'min_delay'):
                actuator.min_delay = actuator.max_delay = self.teacher_settings['delay_steps']


class TeacherEnv(ManagerBasedRLEnv):
    def __init__(self,*args,**kwargs):
        self.terminal_sample = None
        super().__init__(*args,**kwargs)
        self.metric_sum = torch.zeros(8,device=self.device)
        self.metric_count = 0

    def state(self):
        term = controller(self)
        data = term._asset.data
        gravity = data.projected_gravity_b
        return dict(time_s=self.episode_length_buf*self.step_dt,
            velocity=term.actual().clone(),command=term.command.clone(),reference=term.interval_reference.clone(),
            raw_action=term.raw.clone(),leg_action=term.last_leg.clone(),
            tilt_deg=torch.rad2deg(torch.acos((-gravity[:,2]).clamp(-1,1))),
            position=data.root_pos_w.clone(),payload_position=data.body_com_pos_w[:,term.payload_id].clone(),
            arm_error=(data.joint_pos[:,term.arm_ids]-term.nominal[:,:6]).clone(),
            force_world=term.world_force[:,0].clone(),
            force_base=term.interval_reference.clone()*40.)

    def _reset_idx(self,ids):
        active = ids[self.episode_length_buf[ids]>0]
        if len(active):
            self.terminal_sample = (active.clone(),{k:v[active].clone() for k,v in self.state().items()})
        super()._reset_idx(ids)

    def step(self,actions):
        self.terminal_sample = None
        obs,reward,terminated,truncated,info = super().step(actions)
        state = self.state()
        if self.terminal_sample is not None:
            ids,sample = self.terminal_sample
            for key in state: state[key][ids] = sample[key]
        info['teacher_state'] = state
        info['teacher_failed'] = terminated.clone()
        info['teacher_timeout'] = truncated.clone() & ~terminated
        info['teacher_rewards'] = {name:self.reward_manager._step_reward[:,i].clone()*self.step_dt
                                  for i,name in enumerate(self.reward_manager.active_terms)}
        # A failure coinciding with timeout must never be bootstrapped as success.
        info['time_outs'] = truncated.clone() & ~terminated
        active = state['time_s'] > controller(self).settings.settle
        error = state['velocity']-state['reference']
        n = int(active.sum())
        if n:
            self.metric_sum += torch.stack((error[active,0].square().sum(),error[active,1].square().sum(),
                error[active,2].square().sum(),state['command'][active,0].sum(),
                state['velocity'][active,0].sum(),state['reference'][active,0].sum(),
                ((state['raw_action'][active,0]<0)|(state['raw_action'][active,0]>.3)).float().sum(),
                state['tilt_deg'][active].sum()))
            self.metric_count += n
        return obs,reward,terminated,truncated & ~terminated,info

    def pop_metrics(self):
        values = (self.metric_sum/max(1,self.metric_count)).tolist()
        self.metric_sum.zero_()
        self.metric_count = 0
        names = ('vx_rmse','vy_rmse','wz_rmse','command_vx','actual_vx','reference_vx','raw_vx_outside_limits','tilt_mean_deg')
        return {'Teacher/'+k:math.sqrt(v) if i<3 else v for i,(k,v) in enumerate(zip(names,values))}
