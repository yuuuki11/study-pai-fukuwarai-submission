"""Masked supervised TCP loss from teacher action joints, not object poses."""
import torch
import torch.nn.functional as F
from fukuwarai_torch_kinematics import tool_xyz_course_tensor


def masked_tcp_loss(predicted, target, is_pad, board_z):
    if predicted.shape!=target.shape or predicted.shape[:-1]!=is_pad.shape:
        raise ValueError('Action/mask shapes differ')
    xyz=tool_xyz_course_tensor(predicted)
    teacher=tool_xyz_course_tensor(target.detach())
    valid=(~is_pad.bool()) & (teacher[...,2]<=board_z+.015)
    per_slot=F.smooth_l1_loss(xyz,teacher,beta=.002,reduction='none').sum(-1)
    return (per_slot*valid).sum()/valid.sum().clamp(min=1)


def forward_with_tcp_loss(policy,batch,weight,board_z):
    if weight<=0:return policy.forward(batch)
    captured=[]
    def capture(module,inputs,output):captured.append(output[0])
    handle=policy.model.register_forward_hook(capture)
    try:loss,details=policy.forward(batch)
    finally:handle.remove()
    if len(captured)!=1:raise RuntimeError('Expected exactly one ACT forward')
    # Pinned LeRobot inference Unnormalize.forward is @torch.no_grad.
    # Reuse its exact affine transform without that inference-only decorator.
    transform=getattr(policy.unnormalize_outputs.forward, '__wrapped__', None)
    if transform is None:raise RuntimeError('Pinned differentiable unnormalization contract changed')
    raw=transform(policy.unnormalize_outputs, {'action':captured[0]})['action']
    if not raw.requires_grad:raise RuntimeError('TCP predictions detached from ACT graph')
    auxiliary=masked_tcp_loss(raw,batch['action'],batch['action_is_pad'],board_z)
    if not auxiliary.requires_grad:raise RuntimeError('TCP loss detached')
    details=dict(details,tcp_loss_m=float(auxiliary.detach()),tcp_loss_weight=weight)
    return loss+weight*auxiliary,details
