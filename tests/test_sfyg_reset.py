"""Source identity, original PPO execution and explicit physical-stage transfer."""
import contextlib
import ast
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))
spec = importlib.util.spec_from_file_location("reset_core",
    ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers/sfyg_reset.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class ResetTests(unittest.TestCase):
    def test_training_core_is_exactly_the_initial_commit(self):
        self.assertEqual(len(core.verify_source(ROOT)), len(core.CORE_FILES))

    def model(self, critic_width=125):
        from rsl_rl.modules import ActorCritic, MLP_Encoder
        with contextlib.redirect_stdout(io.StringIO()):
            actor = ActorCritic(48,critic_width,10,actor_hidden_dims=[16],critic_hidden_dims=[16])
            encoder = MLP_Encoder(420,3,hidden_dims=[16],output_detach=True)
        return types.SimpleNamespace(actor_critic=actor,encoder=encoder)

    def saved(self, model, phase="source"):
        return {"sfyg_reset_contract": {"schema":core.SCHEMA,"source_commit":core.BASE,"phase":phase},
            "model_state_dict":deepcopy(model.actor_critic.state_dict()),
            "encoder_state_dict":deepcopy(model.encoder.state_dict())}

    def test_source_to_carry_transfers_exact_actor_but_keeps_fresh_critic(self):
        source, target = self.model(), self.model(126)
        before = deepcopy(target.actor_critic.critic.state_dict())
        core.load_weights(target,self.saved(source),"carry")
        for k,v in source.actor_critic.actor.state_dict().items():
            torch.testing.assert_close(v,target.actor_critic.actor.state_dict()[k],rtol=0,atol=0)
        for k,v in before.items():
            torch.testing.assert_close(v,target.actor_critic.critic.state_dict()[k],rtol=0,atol=0)
        for k,v in source.encoder.state_dict().items():
            torch.testing.assert_close(v,target.encoder.state_dict()[k],rtol=0,atol=0)

    def test_rejects_old_checkpoints_and_wrong_evaluation_phase(self):
        model = self.model()
        with self.assertRaises(ValueError):
            core.load_weights(model,{"v4_contract":{}},"source")
        with self.assertRaises(ValueError):
            core.load_weights(model,self.saved(model),"carry",evaluation=True)
        with self.assertRaises(ValueError):
            core.load_weights(model,self.saved(model,"carry"),"source")

    def test_nonfinite_and_incompatible_actor_are_rejected(self):
        model = self.model()
        saved = self.saved(model)
        saved["model_state_dict"]["logstd"][0] = float("nan")
        with self.assertRaises(ValueError):
            core.load_weights(model,saved,"source")
        saved = self.saved(model)
        saved["model_state_dict"]["actor.0.weight"] = torch.zeros(16,141)
        with self.assertRaises(ValueError):
            core.load_weights(model,saved,"carry")

    def test_carry_to_robust_keeps_actor_critic_and_encoder_weights(self):
        source,target = self.model(126),self.model(126)
        core.load_weights(target,self.saved(source,"carry"),"robust")
        for k,v in source.actor_critic.state_dict().items():
            torch.testing.assert_close(v,target.actor_critic.state_dict()[k],rtol=0,atol=0)
        core.load_weights(target,self.saved(source,"carry"),"robust",evaluation=True)
        with self.assertRaises(ValueError):
            core.load_weights(target,self.saved(source,"robust"),"carry")

    def test_original_ppo_runs_without_any_new_kl_guards(self):
        from rsl_rl.algorithm import PPO
        model = self.model()
        algo = PPO(4,model.encoder,model.actor_critic,num_learning_epochs=2,num_mini_batches=2,
                   learning_rate=1e-3,est_learning_rate=1e-3)
        self.assertFalse(algo.early_stop)
        self.assertFalse(hasattr(algo,"update_kl_guard"))
        algo.init_storage(4,4,[42],[125],[420],[3],[10])
        before = deepcopy(model.actor_critic.actor.state_dict())
        for _ in range(4):
            p,h,c,v = torch.randn(4,42),torch.randn(4,420),torch.randn(4,3),torch.randn(4,122)
            with torch.no_grad():
                algo.act(p,h,c,v)
                algo.process_env_step(torch.randn(4),torch.zeros(4,dtype=torch.bool),{},p)
        algo.compute_returns(torch.cat((v,c),-1))
        losses = algo.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses))
        self.assertFalse(torch.equal(before["0.weight"],model.actor_critic.actor[0].weight))
        self.assertEqual(int(next(iter(algo.optimizer.state.values()))["step"]),4)
        self.assertEqual(int(next(iter(algo.extra_optimizer.state.values()))["step"]),4)

    def test_carry_actuation_holds_arm_every_substep_without_changing_leg_actions(self):
        path = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_sfyg_carry_env_cfg.py"
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if getattr(node,"name",None) in ("CarryLegAction","reset_held_joints")]
        calls = []
        defaults = torch.arange(36,dtype=torch.float32).reshape(2,18)/10
        ids = list(range(10,18))
        robot = types.SimpleNamespace(data=types.SimpleNamespace(default_joint_pos=defaults,
                default_joint_vel=torch.zeros_like(defaults)),
            find_joints=lambda *a,**k:(ids,[str(i) for i in ids]),
            set_joint_position_target=lambda q,**k:calls.append(("q",q.clone(),k)),
            set_joint_velocity_target=lambda q,**k:calls.append(("dq",q.clone(),k)),
            set_joint_effort_target=lambda q,**k:calls.append(("tau",q.clone(),k)),
            write_joint_state_to_sim=lambda q,dq,**k:calls.append(("reset",q.clone(),k)))
        class Base:
            action_dim = 10
            def __init__(self,cfg,env): self._asset = robot
            def apply_actions(self): calls.append(("legs",None,None))
        namespace = {"torch":torch,"JointPositionAction":Base}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
        action = namespace["CarryLegAction"](None,None)
        for _ in range(3): action.apply_actions()
        self.assertEqual([name for name,_,_ in calls],["legs","q","dq","tau"]*3)
        torch.testing.assert_close(calls[1][1],defaults[:,ids])
        torch.testing.assert_close(calls[2][1],torch.zeros(2,8))
        env = types.SimpleNamespace(scene={"robot":robot})
        namespace["reset_held_joints"](env,torch.tensor([1]),types.SimpleNamespace(name="robot",joint_ids=ids))
        torch.testing.assert_close(calls[-2][1],defaults[1:2,ids])


if __name__ == "__main__":
    unittest.main()
