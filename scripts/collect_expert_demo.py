#!/usr/bin/env python3
"""Collect one state-based simulator expert rollout, never a screen-operated demo.
The mouth is reset before recording; during recording only arm commands move it.
"""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from rclpy.parameter import Parameter
import rosbag2_py
from sensor_msgs.msg import JointState, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ament_index_python.packages import get_package_share_directory
from crane_plus_description.robot_description_loader import RobotDescriptionLoader
from launch import LaunchContext
from crane_kinematics import ARM, HAND, CraneKinematics
from synced_recorder import SyncedRecorder

ROOT = Path(__file__).resolve().parents[1]
ARM_TOPIC = '/crane_plus_arm_controller/joint_trajectory'
HAND_TOPIC = '/crane_plus_gripper_controller/joint_trajectory'


class PoseStream:
    def __init__(self):
        self.latest = queue.Queue(maxsize=1)
        self.error = None
        self.process = subprocess.Popen(['ign', 'topic', '-e', '--json-output', '-t',
                                         '/world/default/dynamic_pose/info'],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        for line in self.process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            for pose in message.get('pose', []):
                if pose.get('name') == 'fukuwarai_mouth':
                    item = (time.monotonic(), message.get('header', {}).get('stamp', {}), pose)
                    try:
                        self.latest.get_nowait()
                    except queue.Empty:
                        pass
                    self.latest.put(item)
                    break
        self.error = 'Ignition pose stream ended'

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.thread.join(timeout=1)


class Expert(Node):
    def __init__(self, stream, config):
        super().__init__('fukuwarai_state_expert', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.stream, self.config = stream, config
        self.joints, self.mouth, self.last_pose = {}, None, 0
        self.mouth_quaternion = None
        self.sim = 0.
        self.mouth_id, self.pose_sim = None, 0.
        self.writer, self.samples = None, None
        self.phase = 'setup'
        self.last_sample = -1.
        self.recorder = SyncedRecorder(ARM + [HAND], config.get('observation_camera', {}).get('fps', 10))
        self.latest_image = None
        self.calculations = ThreadPoolExecutor(max_workers=1)
        self.arm = self.create_publisher(JointTrajectory, ARM_TOPIC, 10)
        self.hand = self.create_publisher(JointTrajectory, HAND_TOPIC, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/expert/mouth_pose', 10)
        self.phase_pub = self.create_publisher(String, '/expert/task_state', 10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.create_subscription(Clock, '/clock', self.on_clock, 10)
        # Keep the newest camera observation after resets; a deep sensor queue
        # can otherwise leave a previous trial's image at the front.
        self.create_subscription(CompressedImage, '/fukuwarai/overhead/image/compressed', self.on_image,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))

    def compute(self, function, *args):
        pending = self.calculations.submit(function, *args)
        while not pending.done():
            self.spin()
        return pending.result()

    def on_image(self, msg):
        self.latest_image = msg
        self.recorder.image(msg)

    def record(self, topic, msg):
        if self.writer:
            self.writer.write(topic, serialize_message(msg), int(self.sim * 1e9))

    def on_clock(self, msg):
        self.sim = msg.clock.sec + msg.clock.nanosec / 1e9
        self.record('/clock', msg)

    def on_joints(self, msg):
        self.joints.update(zip(msg.name, msg.position))
        self.recorder.state(msg)
        self.record('/joint_states', msg)

    def spin(self):
        rclpy.spin_once(self, timeout_sec=.015)
        try:
            wall, stamp, pose = self.stream.latest.get_nowait()
            self.last_pose = wall
            self.mouth_id = pose.get('id')
            self.pose_sim = int(stamp.get('sec', 0)) + int(stamp.get('nsec', 0)) / 1e9
            p = pose.get('position', {})
            o = pose.get('orientation', {})
            self.mouth_quaternion = [float(o.get(k, 1. if k == 'w' else 0.)) for k in ('x', 'y', 'z', 'w')]
            self.mouth = np.array([p.get(k, 0.) for k in ('x', 'y', 'z')])
            msg = PoseStamped()
            msg.header.frame_id = 'world'
            msg.header.stamp.sec = int(stamp.get('sec', 0))
            msg.header.stamp.nanosec = int(stamp.get('nsec', 0))
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, self.mouth)
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z = (float(o.get(k, 0.)) for k in ('x', 'y', 'z'))
            msg.pose.orientation.w = float(o.get('w', 1.))
            self.pose_pub.publish(msg)
            self.record('/expert/mouth_pose', msg)
        except queue.Empty:
            pass
        self.recorder.flush(self.phase)
        if self.samples and self.sim - self.last_sample >= .1 and self.mouth is not None:
            self.last_sample = self.sim
            self.samples.write(json.dumps({'sim_s': self.sim, 'phase': self.phase,
                               'joint_names': ARM + [HAND], 'joint_positions': [self.joints.get(n) for n in ARM + [HAND]],
                               'mouth_xyz_m': self.mouth.tolist(), 'mouth_quaternion_xyzw': self.mouth_quaternion, 'goal_xy_m': self.config['target_xy_m'],
                               'error_m': float(np.linalg.norm(self.mouth[:2] - self.config['target_xy_m']))}) + '\n')
            self.samples.flush()

    def wait(self, duration):
        end = self.sim + duration
        deadline = time.monotonic() + max(20., duration * 40)
        while self.sim < end:
            self.spin()
            if time.monotonic() > deadline:
                raise RuntimeError('Simulation clock stalled or too slow')
            if self.stream.error or time.monotonic() - self.last_pose > 8:
                raise RuntimeError('Mouth telemetry stale')

    def issue_shared_path(self, names, waypoints, kine=None):
        from fukuwarai_joint_path import prepare_path
        if kine is None:
            if not hasattr(self, '_path_kinematics'):
                self._path_kinematics = kinematics()
            kine = self._path_kinematics
        self.last_path_projection = []
        if names == ARM and getattr(self, 'shared_path_projection', False):
            from fukuwarai_path_projection import project_waypoints
            waypoints, self.last_path_projection = project_waypoints(kine, waypoints, self.config['board_surface_z_m'])
        points = prepare_path([self.joints[n] for n in names], waypoints,
                              [kine.bounds[n] for n in names],
                              kine.minimum_z if names == ARM else None,
                              self.config['board_surface_z_m'])
        msg = JointTrajectory(joint_names=names)
        for seconds, q in points:
            point = JointTrajectoryPoint(positions=q)
            ns = round(seconds * 1e9)
            point.time_from_start.sec, point.time_from_start.nanosec = divmod(ns, 1000000000)
            msg.points.append(point)
        pub = self.arm if names == ARM else self.hand
        self.wait_for_controller(pub)
        self.recorder.plan(int(self.sim * 1e9), names, [self.joints[n] for n in names], points)
        pub.publish(msg)
        self.record(ARM_TOPIC if names == ARM else HAND_TOPIC, msg)
        if not hasattr(self, 'last_shared_path_endpoints'):
            self.last_shared_path_endpoints = {}
        self.last_shared_path_endpoints[tuple(names)] = list(points[-1][1])
        return points[-1][0]

    def command(self, q, seconds, hand=False):
        names = [HAND] if hand else ARM
        msg = JointTrajectory()
        msg.joint_names = names
        pt = JointTrajectoryPoint()
        pt.positions = list(map(float, q))
        pt.time_from_start.sec = int(seconds)
        pt.time_from_start.nanosec = int((seconds % 1) * 1e9)
        msg.points = [pt]
        pub = self.hand if hand else self.arm
        self.wait_for_controller(pub)
        if getattr(self, 'shared_joint_path', False):
            seconds = self.issue_shared_path(names, [(seconds, list(q))])
        else:
            pub.publish(msg)
            self.record(HAND_TOPIC if hand else ARM_TOPIC, msg)
            self.recorder.plan(int(self.sim * 1e9), names, [self.joints[n] for n in names], [(seconds, list(q))])
        self.wait(seconds + .04)
        error = max(abs(self.joints[n] - target) for n, target in zip(names, q))
        settle_deadline = self.sim + 1.5
        while error > .04 and self.sim < settle_deadline:
            self.wait(.05)
            error = max(abs(self.joints[n] - target) for n, target in zip(names, q))
        if error > .09:
            raise RuntimeError(f'Joint tracking error {error:.3f} rad')

    def wait_for_controller(self, publisher, timeout=5.0):
        """Allow ROS discovery to finish before a trajectory is sent.

        A short-lived zero-subscriber race after world resets used to turn an
        otherwise valid episode into a zero-frame recording. Spinning here
        only waits for the controller's existing subscription; it does not
        send speculative commands or change the trajectory.
        """
        deadline = time.monotonic() + timeout
        while not publisher.get_subscription_count():
            self.spin()
            if time.monotonic() >= deadline:
                raise RuntimeError('Controller is not subscribed')

    def stage(self, name):
        self.phase = name
        self.recorder.phase = name
        msg = String(data=json.dumps({'phase': name, 'controller': 'simulator_state_expert'}))
        self.phase_pub.publish(msg)
        self.record('/expert/task_state', msg)
        print(f'{self.sim:.3f}: {name}', flush=True)

    def start_bag(self, out):
        self.writer = rosbag2_py.SequentialWriter()
        self.writer.open(rosbag2_py.StorageOptions(uri=str(out / 'rosbag'), storage_id='sqlite3'),
                         rosbag2_py.ConverterOptions('', ''))
        for name, kind in [('/clock', 'rosgraph_msgs/msg/Clock'), ('/joint_states', 'sensor_msgs/msg/JointState'),
                           (ARM_TOPIC, 'trajectory_msgs/msg/JointTrajectory'), (HAND_TOPIC, 'trajectory_msgs/msg/JointTrajectory'),
                           ('/expert/mouth_pose', 'geometry_msgs/msg/PoseStamped'), ('/expert/task_state', 'std_msgs/msg/String')]:
            self.writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=kind, serialization_format='cdr'))
        self.writer.create_topic(rosbag2_py.TopicMetadata(name='/fukuwarai/overhead/image/compressed',
                                 type='sensor_msgs/msg/CompressedImage', serialization_format='cdr'))
        self.samples = (out / 'observations.jsonl').open('w')
        self.recorder.start(out, self.writer, int(self.sim * 1e9), [self.joints[n] for n in ARM + [HAND]])

    def close_bag(self):
        # Close the image window first. Otherwise every spin can enqueue a new
        # final image before its following joint sample arrives. Never fabricate
        # that sample or relax synchronization; unresolved frames still reject.
        self.recorder.stop_images()
        deadline = time.monotonic() + .5
        while self.recorder.frames and rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.01)
            self.recorder.flush(self.phase)
        audit = self.recorder.finish()
        if self.samples:
            self.samples.close()
            self.samples = None
        self.writer = None  # SequentialWriter closes and writes metadata on destruction.
        return audit

    def cartesian_path(self, kine, xy, heights, seconds):
        previous = [self.joints[n] for n in ARM]
        trajectory = JointTrajectory(joint_names=ARM)
        waypoints = []
        for index, height in enumerate(heights):
            q = self.compute(kine.solve, np.r_[xy, height], previous)
            if min(kine.minimum_z(q).values()) < self.config['board_surface_z_m'] + .001:
                raise RuntimeError('Cartesian path would intersect the board')
            previous = q
            t = seconds * (index + 1) / len(heights)
            point = JointTrajectoryPoint(positions=list(map(float, q)))
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t % 1) * 1e9)
            trajectory.points.append(point)
            waypoints.append((t, list(map(float, q))))
        if getattr(self, 'shared_joint_path', False):
            seconds = self.issue_shared_path(ARM, waypoints, kine)
        else:
            self.recorder.plan(int(self.sim * 1e9), ARM, [self.joints[n] for n in ARM], waypoints)
            self.arm.publish(trajectory)
            self.record(ARM_TOPIC, trajectory)
        self.wait(seconds + .1)
        settle_deadline = self.sim + 1.
        while max(abs(self.joints[n] - p) for n, p in zip(ARM, previous)) > .04 and self.sim < settle_deadline:
            self.wait(.05)
        if max(abs(self.joints[n] - p) for n, p in zip(ARM, previous)) > .09:
            raise RuntimeError('Cartesian path joint tracking error')


def kinematics():
    loader = RobotDescriptionLoader()
    loader.use_gazebo, loader.use_ignition = 'true', 'true'
    loader.gz_control_config_package = 'crane_plus_control'
    loader.gz_control_config_file_path = 'config/crane_plus_controllers.yaml'
    description = loader.load().perform(LaunchContext())
    return CraneKinematics(description, get_package_share_directory('crane_plus_description'))


def run(args):
    config = json.loads((ROOT / 'config/fukuwarai.json').read_text())
    out = Path(args.output_root) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    rclpy.init()
    stream = PoseStream()
    node = Expert(stream, config)
    result = {'source': 'simulator_state_expert', 'screen_observed': False, 'seed': args.seed,
              'success': False, 'recording_camera': True, 'evaluation': 'position_only',
              'geometry_scale': config.get('geometry_scale', 1),
              'scene_config_sha256': hashlib.sha256((ROOT / 'config/fukuwarai.json').read_bytes()).hexdigest(),
              'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'goal_xy_m': config['target_xy_m'], 'tolerance_m': config['position_tolerance_m'],
              'hold_time_sim_s': config['hold_time_sim_s'], 'mouth_teleported_during_episode': False}
    try:
        deadline = time.monotonic() + 20
        while not (all(n in node.joints for n in ARM + [HAND]) and node.mouth is not None and node.sim > 0
                   and node.latest_image is not None):
            node.spin()
            if time.monotonic() > deadline:
                raise RuntimeError('No joint/clock/mouth/camera telemetry')
        kine = kinematics()
        if node.count_publishers(ARM_TOPIC) > 1 or node.count_publishers(HAND_TOPIC) > 1:
            raise RuntimeError('Another teleoperation/controller publisher is active; stop it first')
        node.stage('home_before_recording')
        q = np.array([node.joints[n] for n in ARM])
        node.command(np.zeros(4), max(1.5, float(np.max(np.abs(q))) / .8))
        node.command([.1745], .5, hand=True)
        previous_id = node.mouth_id
        subprocess.run(['python3', str(ROOT / 'scripts/reset_mouth.py'), '--seed', str(args.seed)], check=True)
        deadline = time.monotonic() + 25
        while node.mouth_id == previous_id:
            node.spin()
            if time.monotonic() > deadline:
                raise RuntimeError('New mouth entity not observed after reset')
        node.wait(1.)
        initial = node.mouth.copy()
        result['initial_mouth_xyz_m'] = initial.tolist()
        goal = np.array(config['target_xy_m'])
        direction = (goal - initial[:2]) / np.linalg.norm(goal - initial[:2])
        tool = initial[:2] - direction * .045
        height = config['board_surface_z_m'] + .003
        q_above = kine.solve(np.r_[tool, height + .06])
        q_low = kine.solve(np.r_[tool, height], q_above)
        # Preflight the approach and contact poses before recording/moving.
        for candidate in (q_above, q_low):
            if min(kine.minimum_z(candidate).values()) < config['board_surface_z_m'] + .001:
                raise RuntimeError('Planned arm would intersect the board')
        node.start_bag(out)
        node.stage('approach_above')
        node.command(q_above, max(2., float(np.max(np.abs(q_above))) / 1.))
        node.stage('lower_behind_mouth')
        node.cartesian_path(kine, tool, np.linspace(height + .06, height, 13)[1:], 1.8)
        node.stage('push')
        contact_start = node.mouth.copy()
        held = 0.
        first_inside = None
        for step in range(65):
            error = float(np.linalg.norm(node.mouth[:2] - goal))
            if float(np.dot(node.mouth[:2] - goal, direction)) > .012:
                raise RuntimeError('Push passed goal without meeting tolerance; stop rather than continue sweeping')
            if error <= config['position_tolerance_m']:
                if first_inside is None:
                    first_inside = node.sim
                held = node.sim - first_inside
                if held >= config['hold_time_sim_s']:
                    result['success'] = True
                    break
                node.wait(.15)
                continue
            first_inside = None
            tool += direction * .002
            q = node.compute(kine.solve, np.r_[tool, height], [node.joints[n] for n in ARM])
            if min(kine.minimum_z(q).values()) < config['board_surface_z_m'] + .001:
                raise RuntimeError('Planned push would intersect the board')
            node.command(q, .14)
            if step % 4 == 0:
                print(f'push {step}: mouth={node.mouth[:2].round(5).tolist()} error={error:.4f}', flush=True)
            if np.linalg.norm(node.mouth[:2] - initial[:2]) > .15 or abs(node.mouth[2] - initial[2]) > .025:
                raise RuntimeError('Mouth left task surface')
        result['push_displacement_m'] = float(np.linalg.norm(node.mouth[:2] - contact_start[:2]))
        if result['success']:
            # Score again after withdrawing, so contact does not merely hold it in place.
            node.stage('withdraw_up')
            node.cartesian_path(kine, tool, np.linspace(height, height + .06, 13)[1:], 1.8)
            node.stage('verify_released')
            start = node.sim
            while node.sim - start < config['hold_time_sim_s']:
                node.wait(.1)
                if np.linalg.norm(node.mouth[:2] - goal) > config['position_tolerance_m']:
                    result['success'] = False
                    break
            held = node.sim - start if result['success'] else 0.
        result['held_after_withdrawal_sim_s'] = held
        result['final_mouth_xyz_m'] = node.mouth.tolist()
        result['final_error_m'] = float(np.linalg.norm(node.mouth[:2] - goal))
        node.stage('success' if result['success'] else 'failure')
        node.wait(.3)
    except (Exception, KeyboardInterrupt) as exc:
        result['success'] = False
        result['error'] = str(exc) or 'Interrupted by operator'
        print(f'FAILED: {exc}', flush=True)
    finally:
        if node.mouth is not None:
            result['final_mouth_xyz_m'] = node.mouth.tolist()
            result['final_error_m'] = float(np.linalg.norm(node.mouth[:2] - config['target_xy_m']))
        # Hold current measured joints on every exit; never leave a pending sweep.
        if rclpy.ok() and all(n in node.joints for n in ARM) and node.arm.get_subscription_count():
            hold = JointTrajectory(joint_names=ARM)
            pt = JointTrajectoryPoint(positions=[float(node.joints[n]) for n in ARM])
            pt.time_from_start.nanosec = 200_000_000
            hold.points = [pt]
            node.arm.publish(hold)
            node.record(ARM_TOPIC, hold)
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=.02)
        audit = node.close_bag()
        result['synchronization'] = audit
        result['physical_success'] = result['success']
        result['accepted_for_training'] = bool(result['success'] and audit['quality_pass'])
        stream.close()
        node.calculations.shutdown(wait=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        (out / 'scene_config.json').write_text(json.dumps(config, indent=2) + '\n')
        print(json.dumps({'output': str(out), **result}), flush=True)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-root', default=str(ROOT / 'data/expert_images'))
    raise SystemExit(run(parser.parse_args()))
