import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fukuwarai_teacher_replay import TeacherReplay


class ReplayTest(unittest.TestCase):
    def build(self, root):
        root = Path(root)
        (root/'result.json').write_text(json.dumps({'seed':10251,'accepted_for_training':True,'initial_mouth_xyz_m':[1,2,3]}))
        rows=[{'sim_time_ns':1000000000+i*100000000,'action':[i,i+1,i+2,i+3,i+4], 'observation.state':[0]*5} for i in range(3)]
        (root/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return root
    def test_time_alignment_order_and_diagnostic_identity(self):
        with tempfile.TemporaryDirectory() as t:
            r=TeacherReplay(self.build(t),10251)
            np.testing.assert_array_equal(r.action(.099),[0,1,3,2,4])
            np.testing.assert_array_equal(r.action(.1),[1,2,4,3,5])
            np.testing.assert_array_equal(r.action(100),[2,3,5,4,6])
            self.assertFalse(r.finished(2.19));self.assertTrue(r.finished(2.2))
            self.assertFalse(r.metadata['model_success']);self.assertFalse(r.metadata['learned_model'])
    def test_slower_schedule_preserves_commands_and_bounds(self):
        with tempfile.TemporaryDirectory() as t:
            root=self.build(t);a=TeacherReplay(root,10251);b=TeacherReplay(root,10251,3.)
            np.testing.assert_array_equal(a.action(.1),b.action(.3))
            self.assertFalse(b.finished(2.59));self.assertTrue(b.finished(2.61))
            with self.assertRaises(ValueError):TeacherReplay(root,10251,0.)
    def test_observed_next_targets_are_shifted_with_terminal_hold(self):
        with tempfile.TemporaryDirectory() as t:
            root=self.build(t);p=root/'frames.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
            for i,r in enumerate(rows):r['observation.state']=[i+j/10 for j in range(5)]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            replay=TeacherReplay(root,10251,observed_next_state=True)
            np.testing.assert_allclose(replay.action(0),[1,1.1,1.3,1.2,1.4])
            np.testing.assert_allclose(replay.action(100),[2,2.1,2.3,2.2,2.4])
            self.assertFalse(replay.metadata['future_state_in_policy_input'])

    def test_source_seed_and_acceptance_required(self):
        with tempfile.TemporaryDirectory() as t:
            root=self.build(t)
            with self.assertRaises(ValueError):TeacherReplay(root,999)
            (root/'result.json').write_text(json.dumps({'seed':10251,'accepted_for_training':False}))
            with self.assertRaises(ValueError):TeacherReplay(root,10251)
    def test_invalid_times_and_actions_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=self.build(t);p=root/'frames.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[1]['sim_time_ns']=rows[0]['sim_time_ns'];p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):TeacherReplay(root,10251)


if __name__=='__main__':unittest.main()
