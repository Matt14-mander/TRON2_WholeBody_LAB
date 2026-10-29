"""Actual independent PPO, reset boundaries, contracts, resume and export."""
import contextlib
from dataclasses import replace
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/rsl_rl'),str(ROOT/'rsl_rl')]
from sfyg_student_runner import StudentRunner,Config,gae
from sfyg_teacher_core import Settings,execution_contract,fingerprint
from rsl_rl.modules import ActorCritic


class FakeEnv:
    num_envs = 3
    def __init__(self):
        self.tick = 0
        self.obs = {'policy':torch.zeros(3,147),'critic':torch.zeros(3,177)}

    def step(self,action):
        assert action.shape==(3,9)
        self.tick += 1
        # In-place mutation makes forgotten pre-action clones visible.
        self.obs['policy'].fill_(self.tick*.01)
        self.obs['critic'].fill_(self.tick*.02)
        return self.obs,torch.tensor([.1,.2,-1.]),torch.tensor([False,False,True]),torch.tensor([False,True,True]),{
            'student_terminal_critic':torch.full((3,177),.7)}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        (ROOT/'.test-temp').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/'.test-temp')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.contract = execution_contract('low-sha',{'defaults':[0.]},Settings())
        with contextlib.redirect_stdout(io.StringIO()):
            teacher = ActorCritic(147,177,9,actor_hidden_dims=[128]*3,critic_hidden_dims=[128]*3,activation='elu')
        with torch.no_grad(): teacher.logstd.fill_(-2.3)
        self.teacher_path = self.path/'teacher.pt'
        torch.save(dict(model_state_dict=teacher.state_dict(),sfyg_teacher_contract=self.contract),self.teacher_path)

    def runner(self,config=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return StudentRunner(self.teacher_path,self.contract,config=config or Config(steps=3,epochs=1,batches=2))

    def test_collection_pairs_pre_action_observations_and_true_terminal_values(self):
        r,env = self.runner(),FakeEnv()
        data,_ = r.collect(env,env.obs)
        self.assertEqual(len(data['history']),9)
        self.assertFalse(bool(data['teacher'][:3].any()))
        torch.testing.assert_close(data['history'],data['teacher'][:,:144])
        self.assertEqual(int(data['timeout'].sum()),3)
        self.assertEqual(int(data['failed'].sum()),3)
        with torch.no_grad(): terminal = r.critic(torch.full((1,177),.7)).squeeze()
        torch.testing.assert_close(data['next_value'][data['timeout']],terminal.expand(3))

    def test_gae_timeout_bootstraps_but_failure_and_reset_do_not_leak(self):
        reward = torch.tensor([[1.,1.],[100.,100.]])
        value = torch.zeros_like(reward)
        nxt = torch.full_like(reward,2.)
        done = torch.ones_like(reward,dtype=torch.bool)
        failed = torch.tensor([[False,True],[False,False]])
        advantage,returns = gae(reward,value,nxt,failed,done,.9,.95)
        torch.testing.assert_close(advantage[0],torch.tensor([2.8,1.]))
        torch.testing.assert_close(returns,advantage)

    def test_actual_updates_resume_and_export(self):
        r,env = self.runner(),FakeEnv()
        before = fingerprint(r.policy)
        teacher = fingerprint(r.teacher)
        data,_ = r.collect(env,env.obs)
        stats = r.update(data)
        self.assertEqual(stats['optimizer_steps'],2)
        self.assertEqual(stats['estimator_steps'],2)
        self.assertNotEqual(before,fingerprint(r.policy))
        self.assertEqual(teacher,fingerprint(r.teacher))
        self.assertTrue(all(p.grad is None for p in r.teacher.parameters()))
        r.save(self.path/'model_1.pt')
        restored = self.runner()
        restored.load(self.path/'model_1.pt',resume=True)
        self.assertEqual(restored.iteration,1)
        self.assertTrue(restored.optimizer.state)
        self.assertTrue(restored.est_optimizer.state)
        self.assertEqual(fingerprint(r.policy),fingerprint(restored.policy))
        data,_ = restored.collect(env,env.obs)
        restored.update(data)
        self.assertEqual(restored.iteration,2)
        restored.export(self.path/'export')
        with (self.path/'export/policy.pt').open('rb') as stream:
            exported = torch.jit.load(stream)
        sample = torch.randn(4,144)
        with torch.no_grad(): torch.testing.assert_close(exported(sample),restored.policy(sample))

    def test_binding_and_resume_config_reject_mismatch(self):
        r = self.runner()
        r.save(self.path/'model.pt')
        r.contract['teacher_sha256'] = 'other'
        with self.assertRaises(ValueError): r.load(self.path/'model.pt')
        other = self.runner(replace(Config(steps=3,epochs=1,batches=2),learning_rate=.0002))
        with self.assertRaises(ValueError): other.load(self.path/'model.pt',resume=True)
        # Explicit weight-only warm start may use a different optimizer config.
        other.load(self.path/'model.pt',resume=False)
        self.assertEqual(other.iteration,0)

    def test_missing_terminal_observation_fails_closed(self):
        env = FakeEnv()
        step = env.step
        def missing(action):
            obs,reward,failed,timeout,_ = step(action)
            return obs,reward,failed,timeout,{}
        env.step = missing
        with self.assertRaises(RuntimeError): self.runner().collect(env,env.obs)

    def test_cli_resume_eval_and_development_validation(self):
        from train_sfyg_student import parser_for_task,validate
        base = ['--low_checkpoint','low','--teacher_checkpoint','teacher','--output_dir','out']
        for extra in (['--resume'],['--mode','eval'],['--development']):
            with self.assertRaises(ValueError): validate(parser_for_task().parse_args(base+extra))
        validate(parser_for_task().parse_args(base+['--development','--num_envs','4','--max_iterations','2']))


if __name__=='__main__': unittest.main()
