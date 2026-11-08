import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from compare_sfyg_arm_hold import compare


class ArmComparisonTests(unittest.TestCase):
    def test_each_sim_uses_its_own_zero_and_rejects_mismatched_or_partial_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            mj, isaac=root/'mujoco',root/'isaac'
            mj.mkdir()
            isaac.mkdir()
            settings=dict(settle=.02,ramp=.02,hold=.06,dt=.02)
            info=dict(seed=42,iteration=500,student_sha256='s',low_checkpoint_sha256='l',settings=settings)
            (mj/'summary.json').write_text(json.dumps(info))
            (isaac/'run.json').write_text(json.dumps(dict(parent_student=dict(iteration=500,sha256='s'),
                arguments=dict(seed=42,num_envs=1,episodes=1))))
            (isaac/'contract.json').write_text(json.dumps(dict(teacher_contract=dict(settings=settings,low_sha256='l'))))
            all_isaac=[]
            for f in (0,4,8):
                mj_rows=[]
                for tick in range(2,5):
                    shared=dict(tick=tick,action_time_s=.02*tick,arm_names=[f'arm{i}_Joint' for i in range(1,7)],
                                pre_arm_velocity_rad_s=[-.02]*6)
                    mj_rows.append(json.dumps(dict(shared,cmd_vx=0.,force_fx_n=f,estimated_fx_n=f-1.,
                                                   pre_arm_error_rad=[.5+f*.001]*6)))
                    all_isaac.append(json.dumps(dict(shared,controller='hold',episode=0,environment=0,
                        force_n=f,force_base=[f,0,0],command=[0.,0.,0.],pre_action_estimate=[f/40.,0.,0.],
                        pre_arm_error_rad=[.2+f*.002]*6)))
                (mj/f'low_{f}N_vx0.00.jsonl').write_text('\n'.join(mj_rows))
            steps=isaac/'steps.jsonl'
            steps.write_text('\n'.join(all_isaac))
            compare(root)
            with (root/'arm_comparison.csv').open() as stream:
                records=list(csv.DictReader(stream))
            self.assertEqual(len(records),36)
            self.assertAlmostEqual(float(records[-1]['delta_q_from_0N_rad']),.008)
            self.assertAlmostEqual(float(records[17]['delta_q_from_0N_rad']),.016)
            steps.write_text('\n'.join(all_isaac[:-1]))
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                compare(root)
            (mj/'summary.json').write_text(json.dumps(dict(info,student_sha256='other')))
            with self.assertRaisesRegex(ValueError,'matched'):
                compare(root)
