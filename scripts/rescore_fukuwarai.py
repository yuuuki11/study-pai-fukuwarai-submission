#!/usr/bin/env python3
"""Read-only ROS-bag rescore; write derived v2 reports, never alter old results."""
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped
from collect_expert_demo import kinematics, ARM
from fukuwarai_success import MouthScorer, HoldScorer

ROOT = Path(__file__).resolve().parents[1]


def rescore(directory, kine):
    result = json.loads((directory/'result.json').read_text())
    scene_file = directory/'scene_config.json'
    scene = json.loads((scene_file if scene_file.exists() else ROOT/'config/fukuwarai.json').read_text())
    scorer = MouthScorer(scene); hold = HoldScorer(scorer)
    poses = []
    for db in sorted((directory/'rosbag').glob('*.db3')):
        with sqlite3.connect(db.resolve().as_uri()+'?mode=ro', uri=True) as conn:
            topic = conn.execute("select id from topics where name='/expert/mouth_pose'").fetchone()
            if not topic:
                continue
            for data, in conn.execute('select data from messages where topic_id=? order by timestamp',topic):
                m = deserialize_message(data, PoseStamped)
                t = m.header.stamp.sec+m.header.stamp.nanosec/1e9
                poses.append((t, [m.pose.position.x,m.pose.position.y,m.pose.position.z],
                              [m.pose.orientation.x,m.pose.orientation.y,m.pose.orientation.z,m.pose.orientation.w]))
    poses.sort(key=lambda v:v[0]); times = np.array([x[0] for x in poses])
    rows=[]; missing=0; max_hold=0.; passed=False
    for line in (directory/'observations.jsonl').read_text().splitlines():
        r=json.loads(line);t=r['sim_s']
        if not len(times):
            missing+=1;hold.since=None;continue
        i=int(np.searchsorted(times,t));i=min([max(0,i-1),min(i,len(times)-1)],key=lambda j:abs(times[j]-t))
        if abs(times[i]-t) > .05:
            missing+=1;hold.since=None;continue
        q=dict(zip(r['joint_names'],r['joint_positions']))
        m=hold.update(t,poses[i][1],poses[i][2],float(kine.tool([q[n] for n in ARM])[2]))
        max_hold=max(max_hold,m['held_sim_s']);passed=passed or m['success']
        rows.append(dict(sim_s=t,**m))
    if not rows:
        status='unscorable'
    else:
        status='pass' if passed else ('incomplete_telemetry' if missing else 'fail')
    report={'original_result_preserved':True,'original_success':result.get('success'),
            'seed':result.get('seed'),'rules':scorer.rules,'status':status,'success':passed,
            'scored_samples':len(rows),'missing_pose_samples':missing,'maximum_hold_sim_s':max_hold,
            'minimum_center_error_m':min((r['center_error_m'] for r in rows),default=None),
            'final_metrics':rows[-1] if rows else None,
            'note':'Post-hoc score at recorded observation times with nearest bag pose <=50ms. Missing poses cannot establish a pass.'}
    target=directory/'rescore_mouth_outline_v2.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    return dict(path=str(directory),**report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();kine=kinematics();reports=[]
    for directory in args.directory:
        reports.append(rescore(directory,kine))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'success_version':'mouth_outline_v2','episodes':reports,
        'successes':sum(x['success'] for x in reports),'total':len(reports)},indent=2)+'\n')
    print(json.dumps([{'seed':r['seed'],'status':r['status'],'original_success':r['original_success'],
                      'final_metrics':r['final_metrics']} for r in reports],indent=2))

if __name__=='__main__':main()
