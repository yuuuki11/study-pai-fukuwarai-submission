import json
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fukuwarai_success import MouthScorer, HoldScorer, ROOT


class SuccessTests(unittest.TestCase):
    def setUp(self):
        self.scene=json.loads((ROOT/'config/fukuwarai.json').read_text())
        self.scorer=MouthScorer(self.scene)
        self.xyz=[*self.scene['target_xy_m'], self.scene['board_surface_z_m']+.004]
        self.tool_z=self.scene['board_surface_z_m']+.05

    def metric(self, dx=0., yaw=0., tilt=0.):
        xyz=self.xyz.copy();xyz[0]+=dx
        # q = yaw(z) * tilt(x)
        a,b=math.radians(yaw)/2,math.radians(tilt)/2
        q=[math.cos(a)*math.sin(b),math.sin(a)*math.sin(b),math.sin(a)*math.cos(b),math.cos(a)*math.cos(b)]
        return self.scorer.metrics(xyz,q,self.tool_z)

    def test_aligned(self):
        self.assertTrue(self.metric()['instantaneous_pass'])

    def test_old_15mm_success_is_failure(self):
        self.assertFalse(self.metric(dx=.0145)['instantaneous_pass'])

    def test_rotation(self):
        self.assertFalse(self.metric(yaw=30)['instantaneous_pass'])
        self.assertFalse(self.metric(yaw=180)['instantaneous_pass'])

    def test_contour_is_checked_even_with_center_within_3mm(self):
        m=self.metric(dx=.0028)
        self.assertTrue(m['checks']['center'])
        self.assertFalse(m['checks']['outline'])

    def test_tilt(self):
        self.assertFalse(self.metric(tilt=20)['checks']['tilt'])

    def test_release_and_height(self):
        self.assertFalse(self.scorer.metrics(self.xyz,[0,0,0,1],1.02)['instantaneous_pass'])
        xyz=self.xyz.copy();xyz[2]+=.01
        self.assertFalse(self.scorer.metrics(xyz,[0,0,0,1],self.tool_z)['instantaneous_pass'])

    def test_hold_and_reset_on_bad_sample(self):
        h=HoldScorer(self.scorer)
        for i in range(11):m=h.update(i/10,self.xyz,[0,0,0,1],self.tool_z)
        self.assertTrue(m['success'])
        xyz=self.xyz.copy();xyz[0]+=.01
        self.assertFalse(h.update(1.1,xyz,[0,0,0,1],self.tool_z)['success'])
        self.assertFalse(h.update(1.2,self.xyz,[0,0,0,1],self.tool_z)['success'])

    def test_gap_does_not_prove_hold(self):
        h=HoldScorer(self.scorer)
        h.update(0.,self.xyz,[0,0,0,1],self.tool_z)
        self.assertFalse(h.update(2.,self.xyz,[0,0,0,1],self.tool_z)['success'])

    def test_invalid_quaternion(self):
        with self.assertRaises(ValueError):self.scorer.metrics(self.xyz,[0,0,0,0],self.tool_z)

if __name__=='__main__':unittest.main()
