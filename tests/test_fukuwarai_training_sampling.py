from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_fukuwarai_act import validation_indices


class ValidationSampling(unittest.TestCase):
    def test_small_split_covers_task_instead_of_three_frames(self):
        rows=list(range(100,300))
        result=validation_indices(rows,1)
        self.assertEqual(len(result),60)
        self.assertEqual((result[0],result[-1]),(100,299))
        self.assertEqual(len(set(result)),60)

    def test_old_twenty_episode_protocol_is_preserved(self):
        rows=list(range(100,300))
        self.assertEqual(validation_indices(rows,20),[100,200,299])
        self.assertEqual(validation_indices([5,6],1),[5,6])

if __name__=='__main__':unittest.main()
