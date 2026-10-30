from pathlib import Path
import sys
import math
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fukuwarai_visual_goal import visual_goal_pass, visual_outline_bound


class VisualGoalTest(unittest.TestCase):
    def test_old_separate_thresholds_have_no_contour_margin(self):
        self.assertGreater(visual_outline_bound(.0012, math.radians(2), .036), .002)
        self.assertFalse(visual_goal_pass(.0012, math.radians(2), .036))

    def test_tighter_image_margin(self):
        self.assertTrue(visual_goal_pass(.0004, math.radians(.3), .036))
        self.assertFalse(visual_goal_pass(.0008, math.radians(1), .036))
        self.assertFalse(visual_goal_pass(.0001, math.pi, .036))
        with self.assertRaises(ValueError):
            visual_goal_pass(float('nan'), 0, .036)


if __name__ == '__main__':
    unittest.main()
