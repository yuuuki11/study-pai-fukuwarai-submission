import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from balance_fukuwarai_terminal_sampling import terminal_multiplier
class TerminalMassTest(unittest.TestCase):
 def test_exact_mass_and_missing_group_rejection(self):
  factor=terminal_multiplier(581,147632)
  self.assertAlmostEqual(581*factor/(147632+581*factor),.2)
  with self.assertRaises(ValueError):terminal_multiplier(0,100)
  with self.assertRaises(ValueError):terminal_multiplier(100,0)
