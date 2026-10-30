"""Replay original ROS trajectories as a diagnostic, never a learned policy.

Messages retain waypoints and durations. Dispatch is quantized to the evaluator's
100ms simulation clock; lateness is recorded rather than called exact replay.
"""
import hashlib
import sqlite3
import numpy as np
from fukuwarai_teacher_replay import TeacherReplay


class TrajectoryReplay(TeacherReplay):
    def __init__(self, episode, seed):
        super().__init__(episode, seed)
        from rclpy.serialization import deserialize_message
        from trajectory_msgs.msg import JointTrajectory
        from collect_expert_demo import ARM_TOPIC, HAND_TOPIC
        import json
        first = json.loads((self.episode/'frames.jsonl').read_text().splitlines()[0])['sim_time_ns']
        bag = self.episode/'rosbag/rosbag_0.db3'
        self.messages = []
        with sqlite3.connect(f'file:{bag}?mode=ro', uri=True) as db:
            topics = dict(db.execute('select id,name from topics'))
            for tid, ns, data in db.execute('select topic_id,timestamp,data from messages order by timestamp,id'):
                if topics[tid] in (ARM_TOPIC, HAND_TOPIC) and ns >= first and ns <= first + round(self.times[-1]*1e9):
                    msg = deserialize_message(data, JointTrajectory)
                    self.messages.append(((ns-first)/1e9, topics[tid], msg))
        if not self.messages:
            raise ValueError('No original trajectory commands in capture window')
        self.cursor = 0
        self.metadata.update(execution='original_ros_trajectories_100ms_dispatch',
                             bag_sha256=hashlib.sha256(bag.read_bytes()).hexdigest(),
                             original_commands=len(self.messages),
                             frame_action_reconstruction=False,
                             per_tick_target_clipping=False,
                             safety='joint bounds and <=0.02rad sampled trajectory board clearance; original durations retained')

    def execute_tick(self, node, kine, config, elapsed):
        from collect_expert_demo import ARM, HAND, ARM_TOPIC
        sent = []
        while self.cursor < len(self.messages) and self.messages[self.cursor][0] <= elapsed + 1e-8:
            at, topic, msg = self.messages[self.cursor]
            names = msg.joint_names
            if names not in (ARM, [HAND]):
                raise RuntimeError('Unexpected recorded trajectory joint order')
            current = np.array([node.joints[n] for n in names])
            previous_time = 0.
            waypoints = []
            for point in msg.points:
                target = np.asarray(point.positions, dtype=float)
                seconds = point.time_from_start.sec + point.time_from_start.nanosec/1e9
                if target.shape != current.shape or not np.isfinite(target).all() or seconds <= previous_time:
                    raise RuntimeError('Invalid recorded trajectory')
                for n, q in zip(names, target):
                    lo, hi = kine.bounds[n]
                    if not lo <= q <= hi:
                        raise RuntimeError('Recorded trajectory exceeds joint bounds')
                if names == ARM:
                    for alpha in np.linspace(0, 1, max(5, int(np.ceil(np.max(np.abs(target-current))/.02))+1)):
                        if min(kine.minimum_z(current+alpha*(target-current)).values()) < config['board_surface_z_m']+.0005:
                            raise RuntimeError('Safety stop: recorded trajectory intersects board')
                waypoints.append((seconds, target.tolist()))
                current, previous_time = target, seconds
            node.recorder.plan(int(node.sim*1e9), names, [node.joints[n] for n in names], waypoints)
            (node.arm if topic == ARM_TOPIC else node.hand).publish(msg)
            node.record(topic, msg)
            sent.append({'scheduled_s':at,'dispatch_s':elapsed,'lateness_s':elapsed-at,'points':len(msg.points)})
            self.cursor += 1
        return sent


class ChunkedReferenceReplay(TrajectoryReplay):
    """Diagnostic continuous chunks built from a saved achieved-joint reference.

    Slew-limit consecutive planned waypoints to0.1rad/100ms. Unlike the legacy
    executor this does not restart interpolation from observed joints every tick.
    This is an explicit changed execution contract, not a claim of identical safety.
    """
    def __init__(self, episode, seed, time_scale=3., chunk_size=8, preserve_path=False, observed_next_state=True):
        TeacherReplay.__init__(self, episode, seed, time_scale, observed_next_state=observed_next_state)
        if not 2 <= chunk_size <= 16:
            raise ValueError('Diagnostic chunk_size must be within [2,16]')
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from collect_expert_demo import ARM, HAND, ARM_TOPIC, HAND_TOPIC
        self.messages=[]
        previous=self.initial_state.copy()
        count=int(np.ceil(self.times[-1]*self.time_scale/.1))+1
        reference=[]
        for index in range(count):
            desired=self.action(index*.1)[[0,1,3,2,4]]
            if preserve_path:
                # Stretch segments, never clip away their endpoints/geometry.
                steps=max(1,int(np.ceil(np.max(np.abs(desired-previous))/.1)))
                origin=previous.copy()
                reference.extend(origin+(desired-origin)*j/steps for j in range(1,steps+1))
                previous=desired.copy()
            else:
                previous=previous+np.clip(desired-previous,-.1,.1)
                reference.append(previous.copy())
        count=len(reference)
        for begin in range(0,count,chunk_size):
            points=[]
            for index in range(begin,min(begin+chunk_size,count)):
                points.append(((index-begin+1)*.1,reference[index].copy()))
            for names,topic,indices in [(ARM,ARM_TOPIC,slice(0,4)),([HAND],HAND_TOPIC,slice(4,5))]:
                msg=JointTrajectory(joint_names=names)
                for seconds,q in points:
                    pt=JointTrajectoryPoint(positions=q[indices].tolist())
                    ns=round(seconds*1e9);pt.time_from_start.sec=ns//1000000000;pt.time_from_start.nanosec=ns%1000000000
                    msg.points.append(pt)
                self.messages.append((begin*.1,topic,msg))
        self.cursor=0
        self.metadata['source_reference_duration_s']=self.times[-1]*self.time_scale
        self.times=[0.,count*.1/self.time_scale]
        self.metadata.update(duration_sim_s=count*.1,preserve_reference_path=preserve_path,execution='continuous_achieved_reference_chunks',chunk_size=chunk_size,
            command_messages=len(self.messages),frame_action_reconstruction=False,
            safety='consecutive planned waypoint delta<=0.1rad/100ms; joint bounds and sampled board clearance before chunk publish; no observed-state per-tick reanchoring')
