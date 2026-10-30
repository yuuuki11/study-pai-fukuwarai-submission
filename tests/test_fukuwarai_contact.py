import math
from pathlib import Path
import sys
import unittest
import copy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_fukuwarai_contact import contact_weight
from fukuwarai_sampling import selected_weights


class ContactWeightTest(unittest.TestCase):
    def test_joint_fk_height_rule(self):
        self.assertEqual(contact_weight(1.022, 1.019), 8.)
        self.assertEqual(contact_weight(1.10, 1.019), 1.)
        with self.assertRaises(ValueError):
            contact_weight(math.nan, 1.019)

    def test_alignment_exclusion_and_hash_guards(self):
        sources = [{'source_episode': 'a'}, {'source_episode': 'b'}]
        bounds = {'from': [0, 3], 'to': [3, 5]}
        report = {'scene_config_sha256': 'scene', 'conversion_manifest_sha256': 'dataset',
                  'episodes': [dict(episode_index=0, source_episode='a', frames=3, weights=[1,8,1]),
                               dict(episode_index=1, source_episode='b', frames=2, weights=[8,8])]}
        # Excluded frame 1 and all validation frames 3/4 are not sampled.
        self.assertEqual(selected_weights(report,sources,bounds,[0,2],'scene','dataset').tolist(), [1,1])
        for mode in ['source', 'missing', 'shape', 'negative']:
            bad = copy.deepcopy(report)
            if mode == 'source': bad['conversion_manifest_sha256'] = 'other'
            if mode == 'missing': bad['episodes'].pop()
            if mode == 'shape': bad['episodes'][0]['weights'] = [1]
            if mode == 'negative': bad['episodes'][0]['weights'][0] = -1
            with self.assertRaises(RuntimeError):
                selected_weights(bad,sources,bounds,[0,2],'scene','dataset')


if __name__ == '__main__':
    unittest.main()
