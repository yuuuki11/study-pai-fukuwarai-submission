from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fukuwarai_contact_calibration import update_contact_bias, update_mode_bias


class ContactCalibrationTest(unittest.TestCase):
    def test_angular_bias_does_not_cancel_translation_contact(self):
        before = dict(angular=0., translation=0.)
        updated = update_mode_bias(before, True, .00025, .002)
        self.assertGreater(updated['angular'], 0)
        self.assertEqual(updated['translation'], 0)
        self.assertEqual(before, dict(angular=0., translation=0.))
        again = update_mode_bias(updated, False, .0002, 0)
        self.assertEqual(again['angular'], updated['angular'])
        self.assertLess(again['translation'], 0)

    def test_overshoot_reduces_next_gap_and_miss_increases(self):
        self.assertGreater(update_contact_bias(0, .0003, .002), 0)
        self.assertLess(update_contact_bias(0, .0003, 0), 0)
        self.assertEqual(update_contact_bias(.001, .0003, .0003), .001)

    def test_limits_and_unreliable_motion(self):
        self.assertEqual(update_contact_bias(.004, .0003, .01), .004)
        self.assertEqual(update_contact_bias(-.002, .006, 0), -.002)
        self.assertEqual(update_contact_bias(.001, .001, .05), .001)
        self.assertEqual(update_contact_bias(.001, .001, -.003), .001)
        with self.assertRaises(ValueError):
            update_contact_bias(0, .001, float('nan'))


if __name__ == '__main__':
    unittest.main()
