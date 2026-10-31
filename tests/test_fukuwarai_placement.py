import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from score_fukuwarai_placement import update_hold


class PlacementHold(unittest.TestCase):
    def test_gap_breaks_success_evidence(self):
        since,held=update_hold(1.2,True,0.,.8,.2)
        self.assertEqual((since,held),(1.2,0.))
    def test_failure_resets_and_cannot_accumulate_disjoint_intervals(self):
        self.assertEqual(update_hold(.9,False,0.,.8,.2),(None,0.))
        self.assertEqual(update_hold(1.,True,None,.9,.2),(1.,0.))
    def test_continuous_hold_and_time_reversal(self):
        self.assertEqual(update_hold(1.,True,0.,.9,.2),(0.,1.))
        self.assertEqual(update_hold(.5,True,0.,.9,.2),(.5,0.))
