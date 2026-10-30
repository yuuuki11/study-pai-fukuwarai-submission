import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fukuwarai_path_projection import project_waypoints
from fukuwarai_joint_path import prepare_path
class Kine:
    def minimum_z(self,q):return {'tool':float(q[2])}
    def tool(self,q):return np.array(q[:3])
    def solve(self,xyz,q):return np.r_[xyz,q[3]]
class TestProjection(unittest.TestCase):
    def test_small_lift_preserves_xy_and_rate_contract(self):
        points,report=project_waypoints(Kine(),[(.1,[.2,.1,.999,0.])],1.)
        np.testing.assert_allclose(points[0][1],[.2,.1,1.003,0.]);self.assertAlmostEqual(report[0]['lift_m'],.004)
        prepared=prepare_path([.2,.1,1.01,0.],points,[(-3,3)]*4,Kine().minimum_z,1.)
        self.assertGreaterEqual(prepared[0][0],.1)
    def test_safe_paths_are_identity(self):
        points=[(.1,[.2,.1,1.01,0.])];self.assertEqual(project_waypoints(Kine(),points,1.),(points,[]))
    def test_reject_large_lift_and_xy_changing_ik(self):
        with self.assertRaisesRegex(RuntimeError,'10mm'):project_waypoints(Kine(),[(.1,[0,0,.98,0])],1.)
        class Bad(Kine):
            def solve(self,xyz,q):return np.r_[xyz+[.001,0,0],q[3]]
        with self.assertRaisesRegex(RuntimeError,'XY'):project_waypoints(Bad(),[(.1,[0,0,.999,0])],1.)
