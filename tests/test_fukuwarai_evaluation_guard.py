import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_fukuwarai_batch import demonstration_seeds


class EvaluationSeedGuard(unittest.TestCase):
    def test_original_and_versioned_collections_including_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,seed,success in [('expert_images',11,True),('expert_images_v2_pilot',12,True),('expert_images_v2',13,False)]:
                p=root/name/'episode';p.mkdir(parents=True)
                (p/'result.json').write_text(json.dumps({'seed':seed,'success':success}))
            self.assertEqual(demonstration_seeds(root),{11,12,13})

    def test_corrupt_record_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'expert_images_v2'/'episode';p.mkdir(parents=True)
            (p/'result.json').write_text('{')
            with self.assertRaises(json.JSONDecodeError):demonstration_seeds(root)

if __name__=='__main__':unittest.main()
