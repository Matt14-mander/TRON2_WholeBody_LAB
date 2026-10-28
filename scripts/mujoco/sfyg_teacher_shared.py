"""EXPERIMENTAL grasp-reaction proxy for the existing privileged T1 actor.

This is an explicit input distribution change, not the original external force
oracle and not a trained student. No leader state or object force share is used.
"""
import numpy as np

from sfyg_teacher_policy import TeacherPolicy


class TeacherGraspProxy:
    def __init__(self,teacher_bundle,low_config,*,input_mode,axes='forward',max_speed=.3,**unused):
        if input_mode!='grasp-proxy':
            raise ValueError('Shared teacher requires explicit --teacher_input grasp-proxy')
        if axes!='forward': raise ValueError('Current T1 teacher supports forward only')
        self.teacher=TeacherPolicy(teacher_bundle,low_config)
        if max_speed+1e-8<self.teacher.settings.command_max:
            raise ValueError('Use --max_speed .3 to preserve teacher execution limits')
        self.reset()

    def reset(self):
        self.teacher.reset()
        self.bias=np.zeros(6)
        self.bias_samples=0
        self.corrected_local=np.zeros(3)
        self.corrected_wrench=np.zeros(6)
        self.proxy_force=0.
        self.suggestion=np.zeros(3)
        self.samples=self.negative_samples=self.above_range_samples=0
        self.last_tick=-1

    def update(self,o):
        from sfyg_team_control import HighAction
        if abs(o.dt-self.teacher.settings.dt)>1e-9:
            raise ValueError('Shared teacher must run at its saved 50Hz rate')
        tick=round(o.time_s/o.dt)
        if abs(tick*o.dt-o.time_s)>1e-5 or tick<0 or tick<=self.last_tick:
            raise ValueError('Teacher requires one monotonic update per local control tick; reset on episode reset')
        self.last_tick=tick
        wrench=np.asarray(o.grasp_world,float)
        rotation=np.asarray(o.base_rotation,float)
        if wrench.shape!=(6,) or rotation.shape!=(3,3) or not np.isfinite(np.r_[wrench,rotation.ravel()]).all():
            raise ValueError('Finite own grasp wrench6 and base rotation required')
        if o.calibrating and not o.settling: raise ValueError('Bias calibration is only valid while settling')
        if o.settling and o.calibrating:
            self.bias_samples+=1
            self.bias+=(wrench-self.bias)/self.bias_samples
        self.corrected_wrench=np.r_[rotation.T@(wrench[:3]-self.bias[:3]),rotation.T@(wrench[3:]-self.bias[3:])]
        self.corrected_local=self.corrected_wrench[[0,1,5]].copy()
        fx=float(self.corrected_wrench[0])
        self.proxy_force=0. if o.settling else float(np.clip(fx,0.,self.teacher.settings.force_max))
        if not o.settling:
            if not self.bias_samples: raise ValueError('Teacher proxy requires settled bias calibration')
            self.samples+=1
            self.negative_samples+=int(fx<0.)
            self.above_range_samples+=int(fx>self.teacher.settings.force_max)
        self.suggestion=self.teacher.command(o.arm_position_error,o.arm_velocity,o.previous_arm_target_offset,
            np.array([self.proxy_force,0.,0.],np.float32),tick)
        if o.settling: self.suggestion[:]=0.
        return HighAction(self.suggestion.copy(),np.zeros(6))

    def diagnostics(self):
        return dict(kind='teacher_grasp_proxy_EXPERIMENTAL',input_equivalent_to_training=False,
            bias_world=self.bias.tolist(),bias_samples=self.bias_samples,
            corrected_local_wrench=self.corrected_wrench.tolist(),proxy_fx_n=self.proxy_force,
            raw_action=self.teacher.raw.tolist(),suggestion=self.suggestion.tolist(),
            samples=self.samples,negative_fx_samples=self.negative_samples,above_8n_samples=self.above_range_samples)

    def metadata(self):
        return dict(kind='teacher_grasp_proxy_EXPERIMENTAL',teacher_contract=self.teacher.manifest,
            teacher_sha256=self.teacher.sha,input_equivalent_to_training=False,
            input='own grasp WORLD wrench - static WORLD bias; rotate to own BASE; clip Fx to [0,8] N; normalize /40',
            ignored_inputs='Fy/Fz/Tx/Ty/Tz retained in diagnostics but not supplied to current forward T1 actor',
            history='original 8x18 arm history; zero reset; one update per control tick',
            signal_filter='none; no added admittance, command smoothing, force gain or minimum speed',
            limitations=['Grasp reaction differs from training external payload COM force',
                'Shared load mass/inertia and force distribution differ from single 0.5kg load',
                'Current T1 cannot reverse, move sideways or turn; backward Fx is clipped to zero',
                'Not a student and not PAINT decentralized deployment acceptance'])


def validate_proxy_timing(controllers,settle_seconds,delay_steps):
    for controller in controllers:
        settings=controller.teacher.settings
        if abs(settings.settle-settle_seconds)>1e-9 or settings.delay_steps!=delay_steps:
            raise ValueError(f'Teacher requires --settle_seconds {settings.settle:g} and --delay_steps {settings.delay_steps}')
