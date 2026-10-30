import sys,unittest,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fukuwarai_tcp_loss import masked_tcp_loss
class TCPLossTest(unittest.TestCase):
 def test_padding_and_no_valid_slots_have_zero_gradient(self):
  target=torch.tensor([[[.1,.2,-.3,.4,.1745],[.1,.2,-.3,.4,.1745]]],dtype=torch.float64)
  pred=(target+.01).requires_grad_();loss=masked_tcp_loss(pred,target,torch.tensor([[False,True]]),10.);loss.backward()
  self.assertGreater(pred.grad[0,0,:4].abs().sum(),0);torch.testing.assert_close(pred.grad[0,1],torch.zeros(5,dtype=torch.float64))
  pred=target.clone().requires_grad_();zero=masked_tcp_loss(pred,target,torch.ones((1,2),dtype=torch.bool),10.);self.assertEqual(float(zero),0);zero.backward();self.assertTrue(torch.isfinite(pred.grad).all())
 def test_identical_actions_and_far_above_board_no_loss(self):
  q=torch.zeros((1,16,5),requires_grad=True);self.assertEqual(float(masked_tcp_loss(q,q,torch.zeros((1,16),dtype=torch.bool),1.019)),0)

class TrainingGraphTest(unittest.TestCase):
 def test_auxiliary_uses_one_forward_and_reaches_model_through_denormalization(self):
  from fukuwarai_tcp_loss import forward_with_tcp_loss
  class Model(torch.nn.Module):
   def __init__(self):super().__init__();self.q=torch.nn.Parameter(torch.tensor([.1,.2,-.3,.4,.1745]));self.calls=0
   def forward(self,b):self.calls+=1;return self.q[None,None,:].expand(1,2,5),None
  class Denorm(torch.nn.Module):
   @torch.no_grad()
   def forward(self,b):return {'action':b['action']*.5}
  class Policy:
   def __init__(self):self.model=Model();self.unnormalize_outputs=Denorm()
   def forward(self,b):out=self.model(b)[0];return out.sum()*0,{}
  policy=Policy();batch={'action':torch.zeros((1,2,5)),'action_is_pad':torch.tensor([[False,True]])}
  loss,_=forward_with_tcp_loss(policy,batch,10,10);loss.backward()
  self.assertEqual(policy.model.calls,1);self.assertGreater(policy.model.q.grad[:4].abs().sum(),0)
