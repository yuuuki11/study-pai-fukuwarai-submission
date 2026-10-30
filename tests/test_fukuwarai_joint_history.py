from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fukuwarai_joint_history import CausalJointHistory, JointHistoryStream, velocity_cue
from train_fukuwarai_act import CAMERA, prepare, preprocessing_spec


class Base:
    meta = SimpleNamespace()
    episode_data_index = {'from': [0, 3], 'to': [3, 5]}
    hf_dataset = {'observation.state': torch.arange(25).reshape(5, 5).float()}
    def __len__(self): return 5
    def __getitem__(self, i): return {'observation.state': self.hf_dataset['observation.state'][i].clone()}


class JointHistoryTest(unittest.TestCase):
    def test_raw_previous_frame_and_episode_boundary(self):
        d = CausalJointHistory(Base())
        for i, previous in [(0,0),(1,0),(2,1),(3,3),(4,3)]:
            torch.testing.assert_close(d[i]['observation.previous_state'], d.states[previous])
        # Frame 1 may be loss-masked; frame 2 still uses raw frame 1.
        torch.testing.assert_close(d[2]['observation.previous_state'], Base.hf_dataset['observation.state'][1])
        base=Base()
        base.hf_dataset={'observation.state':list(base.hf_dataset['observation.state'])}
        torch.testing.assert_close(CausalJointHistory(base).states,d.states)

    def test_stream_reset_and_no_commit_on_retry(self):
        stream = JointHistoryStream(); q = torch.ones(1,5)
        torch.testing.assert_close(stream.previous(q),q)
        stream.commit(q); newer=q+.1
        torch.testing.assert_close(stream.previous(newer),q)
        torch.testing.assert_close(stream.previous(newer),q)
        stream.commit(newer);stream.reset()
        torch.testing.assert_close(stream.previous(q),q)

    def test_preparation_and_bounded_motion_cue(self):
        q=torch.ones(1,5);prev=q-.04
        batch={CAMERA:torch.zeros(1,3,480,640),'observation.state':q,'observation.previous_state':prev}
        out=prepare(batch,'cpu',None,True,True)
        self.assertEqual(out['observation.environment_state'].shape,(1,10))
        torch.testing.assert_close(out['observation.environment_state'][:,5:],torch.full((1,5),.1))
        self.assertEqual(preprocessing_spec(None,True,True)['schema_version'],3)
        self.assertEqual(float(velocity_cue(q,prev-100).max()),1.)
        with self.assertRaises(ValueError):preprocessing_spec(None,False,True)


if __name__ == '__main__':unittest.main()
