"""Differentiable authored joint-to-TCP geometry for supervised action loss.

Inputs are joint vectors; targets are teacher actions, never object poses.
"""
import torch
from fukuwarai_tool_cues import static_geometry


def tool_xyz_course_tensor(state):
    if state.shape[-1]!=5:raise ValueError('Expected five course-order joints')
    robot,_=static_geometry();raw=state[..., [0,1,3,2]]
    values=dict(zip(robot['arm_raw_order'],raw.unbind(-1)))
    values['crane_plus_joint_hand']=state[...,4]
    shape=state.shape[:-1];eye=torch.eye(4,dtype=state.dtype,device=state.device)
    base=eye.expand(*shape,4,4).clone();base[...,2,3]=robot['base_z_m'];frames={'world':base}
    for joint in robot['joints']:
        rotation=eye.expand(*shape,4,4).clone()
        if joint['axis'] is not None:
            axis=torch.as_tensor(joint['axis'],dtype=state.dtype,device=state.device)
            x,y,z=axis.unbind();zero=x*0
            skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)
            angle=values[joint['name']][...,None,None]
            r=torch.eye(3,dtype=state.dtype,device=state.device)+torch.sin(angle)*skew+(1-torch.cos(angle))*(skew@skew)
            rotation[...,:3,:3]=r
        origin=torch.as_tensor(joint['origin'],dtype=state.dtype,device=state.device)
        frames[joint['child']]=frames[joint['parent']]@origin@rotation
    tcp=torch.as_tensor([*robot['tcp_link4_m'],1.],dtype=state.dtype,device=state.device)
    return (frames['crane_plus_link4']@tcp)[...,:3]
