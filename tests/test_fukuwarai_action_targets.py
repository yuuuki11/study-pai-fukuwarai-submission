import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fukuwarai_action_targets import BoundedActionTargets


class Fake:
    meta = None
    episode_data_index = {'from': [0, 3], 'to': [3, 5]}
    def __init__(self):
        self.hf_dataset = {'observation.state': torch.tensor([[0.]*5, [.2]*5, [.4]*5, [10.]*5, [20.]*5]),
                           'action': torch.tensor([[1.]*5, [-1.]*5, [.45]*5, [100.]*5, [200.]*5])}
    def __len__(self):
        return 5
    def __getitem__(self, i):
        return {'episode_index': int(i >= 3), 'action': torch.zeros(4, 5),
                'observation.state': self.hf_dataset['observation.state'][i],
                'action_is_pad': torch.tensor([False, False, True, True])}


class TargetTest(unittest.TestCase):
    def test_future_slot_state_and_episode_boundary(self):
        base = Fake(); wrapped = BoundedActionTargets(base)
        result = wrapped[1]
        torch.testing.assert_close(result['action'][:, 0], torch.tensor([.1, .45, .45, .45]))
        torch.testing.assert_close(result['observation.state'], base[1]['observation.state'])
        torch.testing.assert_close(result['action_is_pad'], base[1]['action_is_pad'])
        self.assertTrue(torch.equal(base[1]['action'], torch.zeros(4, 5)))
        self.assertEqual(set(result), set(base[1]))  # No future state in inputs.

    def test_train_stats_exclude_validation_and_are_finite(self):
        wrapped = BoundedActionTargets(Fake()); stats = wrapped.train_stats([0])
        self.assertEqual(stats['count'].tolist(), [3])
        self.assertAlmostEqual(float(stats['mean'][0]), (.1+.1+.45)/3, places=6)
        self.assertTrue((stats['max'] < 1).all())
        self.assertTrue((stats['std'] > 0).all())

    def test_nonfinite_rejected(self):
        base = Fake(); base.hf_dataset['action'][0, 0] = float('nan')
        with self.assertRaises(ValueError): BoundedActionTargets(base)


if __name__ == '__main__': unittest.main()
