import sys,unittest,numpy as np,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fukuwarai_torch_kinematics import tool_xyz_course_tensor
from fukuwarai_tool_cues import tool_xyz_course
class GeometryTest(unittest.TestCase):
 def test_matches_authored_fk_and_has_finite_nonzero_gradient(self):
  q=torch.tensor([[.1,.2,-.3,.4,.1745],[-.2,-.1,.2,.1,.1745]],dtype=torch.float64,requires_grad=True)
  xyz=tool_xyz_course_tensor(q);np.testing.assert_allclose(xyz.detach().numpy(),np.array([tool_xyz_course(x) for x in q.detach().numpy()]),atol=1e-10)
  xyz.square().sum().backward();self.assertTrue(torch.isfinite(q.grad).all());self.assertGreater(q.grad[:,:4].abs().sum(),0)
  self.assertEqual(tuple(tool_xyz_course_tensor(q[:,None,:].expand(-1,16,-1)).shape),(2,16,3))
