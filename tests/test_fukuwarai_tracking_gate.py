import sys,unittest,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fukuwarai_tracking_gate import EndpointTrackingGate
class TrackingTest(unittest.TestCase):
 def test_measured_completion_and_bounded_timeout(self):
  gate=EndpointTrackingGate();self.assertTrue(gate.ready([0]*5,0));gate.arm([.1]*5,1)
  self.assertFalse(gate.ready([.1]*5,.9));self.assertFalse(gate.ready([0]*5,1.2));self.assertTrue(gate.ready([.09]*5,1.5))
  with self.assertRaisesRegex(RuntimeError,'timeout'):gate.ready([0]*5,2)
  with self.assertRaises(ValueError):gate.ready([np.nan]*5,2)
