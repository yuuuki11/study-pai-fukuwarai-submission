"""Shared endpoint-preserving joint-path contract, independent of ROS/model inputs."""
import numpy as np


def prepare_path(start, waypoints, bounds, minimum_z=None, board_z=None, max_speed=1.):
    previous=np.asarray(start,float);source_time=0.;time=0.;result=[]
    if previous.ndim!=1 or not np.isfinite(previous).all() or not 0<max_speed<=1.:
        raise ValueError('Invalid path initial state or speed')
    for at,target in waypoints:
        target=np.asarray(target,float)
        if target.shape!=previous.shape or not np.isfinite(target).all() or not np.isfinite(at) or at<=source_time:
            raise ValueError('Invalid path point/time')
        if any(not lo<=q<=hi for q,(lo,hi) in zip(target,bounds)) or len(bounds)!=len(target):
            raise ValueError('Path exceeds joint bounds')
        if minimum_z is not None:
            count=max(5,int(np.ceil(np.max(abs(target-previous))/.02))+1)
            if any(min(minimum_z(previous+alpha*(target-previous)).values())<board_z+.0005 for alpha in np.linspace(0,1,count)):
                raise RuntimeError('Safety stop: shared joint path intersects board')
        duration=max(at-source_time,float(np.max(abs(target-previous)))/max_speed)
        time+=duration;result.append((time,target.tolist()))
        previous=target;source_time=at
    if not result:raise ValueError('Empty joint path')
    return result
