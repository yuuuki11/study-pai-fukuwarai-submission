"""Static board safety projection, no object pose or task-goal information."""
import numpy as np


def project_waypoints(kine, waypoints, board_z):
    corrected=[];report=[]
    for seconds,values in waypoints:
        q=np.asarray(values,float)
        if q.shape!=(4,) or not np.isfinite(q).all():raise ValueError('Invalid arm waypoint')
        minimum=min(kine.minimum_z(q).values())
        if minimum < board_z+.0005:
            xyz=np.asarray(kine.tool(q),float);lift=board_z+.003-minimum
            if not 0<lift<=.010:raise RuntimeError('Safety stop: required path lift exceeds10mm')
            target=xyz.copy();target[2]+=lift
            try:new=np.asarray(kine.solve(target,q),float)
            except ValueError as exc:raise RuntimeError('Safety stop: path projection IK failed') from exc
            if new.shape!=(4,) or not np.isfinite(new).all() or max(abs(new-q))>.1:
                raise RuntimeError('Safety stop: path projection joint correction exceeds0.1rad')
            xy=float(np.linalg.norm(kine.tool(new)[:2]-xyz[:2]))
            if xy>1e-5 or min(kine.minimum_z(new).values())<board_z+.0005:
                raise RuntimeError('Safety stop: path projection failed XY/clearance check')
            report.append({'time_s':seconds,'lift_m':float(lift),'max_joint_correction_rad':float(max(abs(new-q))),
                           'tcp_xy_shift_m':xy,'requested_min_z_m':float(minimum)})
            q=new
        corrected.append((seconds,q.tolist()))
    # Caller must still verify joint bounds, full interpolated path and rate.
    return corrected,report
