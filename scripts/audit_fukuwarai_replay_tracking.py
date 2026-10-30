#!/usr/bin/env python3
"""Read-only achieved-joint path alignment audit; never a task-success scorer."""
import argparse,json
from pathlib import Path
import numpy as np
from fukuwarai_tool_cues import tool_xyz_course


def audit(source,replay,stride=5):
    def load(p):
        rows=[json.loads(x) for x in (p/'frames.jsonl').read_text().splitlines()][::stride]
        q=np.asarray([r['observation.state'] for r in rows],float)
        xyz=np.asarray([tool_xyz_course(x[[0,1,3,2,4]]) for x in q])
        return q,xyz
    a,ax=load(source);b,bx=load(replay);n,m=len(a),len(b)
    costs=np.full((n+1,m+1),np.inf);costs[0,0]=0;parents={}
    for i in range(1,n+1):
        for j in range(1,m+1):
            options=[(costs[i-1,j-1],(i-1,j-1)),(costs[i-1,j],(i-1,j)),(costs[i,j-1],(i,j-1))]
            value,prev=min(options,key=lambda z:z[0]);costs[i,j]=value+np.linalg.norm(a[i-1,:4]-b[j-1,:4]);parents[i,j]=prev
    pairs=[];i,j=n,m
    while i and j:
        pairs.append((i-1,j-1));i,j=parents[i,j]
    pairs.reverse();ia,ib=np.array(pairs).T
    dq=abs(a[ia,:4]-b[ib,:4]);dx=np.linalg.norm(ax[ia]-bx[ib],axis=1)
    board=json.loads(Path('config/fukuwarai.json').read_text())['board_surface_z_m'];contact=ax[ia,2]<=board+.015
    def stats(mask):
        return {'aligned_pairs':int(mask.sum()),'arm_joint_mae_rad':float(dq[mask].mean()),'tcp_error_median_mm':float(np.median(dx[mask])*1000),'tcp_error_p95_mm':float(np.quantile(dx[mask],.95)*1000),'replay_tool_height_bias_mm':float(np.median((bx[ib,2]-ax[ia,2])[mask])*1000)} if mask.any() else None
    return {'source':str(source),'replay':str(replay),'stride':stride,'method':'unrestricted monotone DTW on achieved arm joints, diagnostic alignment only; not time/physics equivalence or a success score','all':stats(np.ones(len(pairs),bool)),'source_near_board':stats(contact),'source_min_tool_z_m':float(ax[:,2].min()),'replay_min_tool_z_m':float(bx[:,2].min()),'model_success':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('Preserve existing audit; choose a new output')
    a.output.write_text(json.dumps(audit(a.source,a.replay),indent=2)+'\n')
