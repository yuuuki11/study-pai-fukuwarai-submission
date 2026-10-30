"""Optional joint-only endpoint readiness; never uses object state."""
import numpy as np

class EndpointTrackingGate:
    def __init__(self, tolerance_rad=.02, maximum_wait_sim_s=1.):
        if not 0 < tolerance_rad <= .02 or not 0 < maximum_wait_sim_s <= 1.:
            raise ValueError('Invalid endpoint tracking guard')
        self.tolerance=tolerance_rad;self.maximum_wait=maximum_wait_sim_s
        self.target=None;self.deadline=None
    def arm(self,target,deadline):
        target=np.asarray(target,float)
        if target.shape!=(5,) or not np.isfinite(target).all() or not np.isfinite(deadline):
            raise ValueError('Invalid endpoint')
        self.target=target.copy();self.deadline=float(deadline)
    def ready(self,current,sim_s):
        if self.target is None:return True
        current=np.asarray(current,float)
        if current.shape!=(5,) or not np.isfinite(current).all() or not np.isfinite(sim_s):raise ValueError('Invalid telemetry')
        if sim_s < self.deadline:return False
        error=float(np.max(np.abs(current[:4]-self.target[:4])))
        if error<=self.tolerance:return True
        if sim_s-self.deadline>=self.maximum_wait:raise RuntimeError('Endpoint tracking timeout')
        return False
