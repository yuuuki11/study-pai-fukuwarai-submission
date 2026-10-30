import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fukuwarai_joint_path import prepare_path

class PathTest(unittest.TestCase):
    def test_retimes_without_endpoint_clipping(self):
        points=prepare_path([0.,0.],[(.1,[.3,.1]),(.2,[.4,.2])],[(-1,1)]*2)
        np.testing.assert_allclose([t for t,q in points],[.3,.4])
        np.testing.assert_allclose([q for t,q in points],[[.3,.1],[.4,.2]])
    def test_preserves_already_slow_timing(self):
        self.assertEqual(prepare_path([0.],[(1.,[.2]),(2.,[.3])],[(-1,1)]),[(1.,[.2]),(2.,[.3])])
    def test_rejects_crossing_board_even_with_safe_endpoints(self):
        with self.assertRaisesRegex(RuntimeError,'intersects'):
            prepare_path([0.],[(1.,[1.])],[(-1,1)],lambda q:{'tool':.9 if .4<q[0]<.6 else 1.1},1.)
    def test_rejects_invalid_times_and_joint_bounds(self):
        for points in [[(.1,[2.])],[(0.,[0.])],[(.1,[float('nan')])]]:
            with self.assertRaises(ValueError):prepare_path([0.],points,[(-1,1)])
