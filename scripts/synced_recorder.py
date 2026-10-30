"""Fixed-camera frames synchronized in simulation time with robot states/actions.
No simulator object coordinates are included in model inputs.
"""
from bisect import bisect_right
from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class SyncedRecorder:
    def __init__(self, joint_names, fps=10):
        self.names, self.fps = joint_names, fps
        self.states = deque(maxlen=1000)
        self.frames = deque(maxlen=100)
        self.plans = []
        self.out = self.rows = self.writer = None
        self.phase = 'setup'
        self.frames_received = self.frames_written = self.frames_rejected = 0
        self.offsets = []
        self.times = []
        self.accept_images = False
        self.rejection_reasons = {}
        self.pending_frames_at_stop = None
        self.images_after_capture_stop = 0

    def reject(self, reason):
        self.frames_rejected += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    def stop_images(self):
        """End the capture window while still accepting joint states to drain it."""
        if self.pending_frames_at_stop is None:
            self.pending_frames_at_stop = len(self.frames)
        self.accept_images = False

    def state(self, msg):
        state = dict(zip(msg.name, msg.position))
        if all(n in state for n in self.names):
            self.states.append((stamp_ns(msg.header.stamp), np.array([state[n] for n in self.names])))

    def plan(self, timestamp_ns, names, start, waypoints):
        # Append an explicit plan, preserving the command that was active at each frame.
        self.plans.append({'time': timestamp_ns, 'names': names,
                           'start': np.array(start), 'points': waypoints})

    def image(self, msg):
        if self.out:
            if not self.accept_images:
                self.images_after_capture_stop += 1
                return
            # A camera frame queued before recording began is stale, not a
            # sample. Exclude it before counting received frames.
            if stamp_ns(msg.header.stamp) < self.begin:
                return
            self.frames_received += 1
            if len(self.frames) == self.frames.maxlen:
                self.reject('image_queue_overflow')
            self.frames.append(msg)

    def start(self, out, writer, sim_ns, current_state):
        self.out = Path(out)
        (self.out / 'images').mkdir()
        self.rows = (self.out / 'frames.jsonl').open('w')
        self.writer = writer
        self.begin = sim_ns
        self.accept_images = True
        self.initial_state = np.array(current_state, dtype=float)

    def action(self, timestamp_ns):
        output = self.initial_state.copy()
        # Commands are joint targets interpolated at the next camera tick.
        for plan in self.plans:
            if plan['time'] > timestamp_ns:
                break
            elapsed = (timestamp_ns - plan['time']) / 1e9
            a_time, a_value = 0., plan['start']
            value = a_value
            for b_time, b_value in plan['points']:
                if elapsed <= b_time:
                    alpha = np.clip((elapsed - a_time) / max(1e-9, b_time - a_time), 0., 1.)
                    value = a_value + alpha * (np.array(b_value) - a_value)
                    break
                a_time, a_value, value = b_time, np.array(b_value), np.array(b_value)
            for name, target in zip(plan['names'], value):
                output[self.names.index(name)] = target
        return output

    def flush(self, phase, force=False):
        if not self.out or not self.states:
            return
        while self.frames:
            msg = self.frames[0]
            ns = stamp_ns(msg.header.stamp)
            if ns < self.begin:
                self.frames.popleft()
                continue
            history = list(self.states)
            times = [x[0] for x in history]
            index = bisect_right(times, ns)
            if index == len(history) and not force:
                break
            self.frames.popleft()
            if index == 0 or index == len(history):
                self.reject('missing_state_bracket')
                continue
            before, after = history[index - 1], history[index]
            gap = after[0] - before[0]
            offset = min(ns - before[0], after[0] - ns) / 1e6
            if gap > 100_000_000 or offset > 30.:
                self.reject('state_gap_or_offset')
                continue
            if 'jpeg' not in msg.format.lower():
                raise RuntimeError('Expected JPEG image transport: ' + msg.format)
            jpg = bytes(msg.data)
            decoded = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape != (480, 640, 3):
                self.reject('invalid_image')
                continue
            path = f'images/{self.frames_written:06d}.jpg'
            (self.out / path).write_bytes(jpg)
            state = before[1] + (ns - before[0]) / max(1, gap) * (after[1] - before[1])
            action = self.action(ns + round(1e9 / self.fps))
            row = {'frame_index': self.frames_written, 'timestamp': (ns - self.begin) / 1e9,
                   'sim_time_ns': ns, 'image': path, 'observation.state': state.tolist(),
                   'action': action.tolist(), 'phase': phase,
                   'nearest_state_offset_ms': offset, 'state_bracket_ms': gap / 1e6}
            self.rows.write(json.dumps(row) + '\n')
            self.rows.flush()
            compressed = CompressedImage()
            compressed.header = msg.header
            compressed.format = 'jpeg'
            compressed.data = jpg
            from rclpy.serialization import serialize_message
            self.writer.write('/fukuwarai/overhead/image/compressed', serialize_message(compressed), ns)
            self.offsets.append(offset)
            self.times.append(ns)
            self.frames_written += 1

    def finish(self):
        self.stop_images()
        self.flush(self.phase, force=True)
        if self.rows:
            self.rows.close()
        intervals = np.diff(self.times) / 1e9
        audit = {'frames_received': self.frames_received, 'frames_written': self.frames_written,
                 'frames_rejected': self.frames_rejected, 'fps_requested': self.fps,
                 'rejection_reasons': dict(self.rejection_reasons),
                 'pending_frames_at_capture_stop': self.pending_frames_at_stop,
                 'images_after_capture_stop': self.images_after_capture_stop,
                 'median_frame_interval_sim_s': float(np.median(intervals)) if len(intervals) else None,
                 'maximum_frame_interval_sim_s': float(max(intervals)) if len(intervals) else None,
                 'maximum_nearest_state_offset_ms': float(max(self.offsets)) if self.offsets else None,
                 'input_features': ['observation.images.overhead', 'observation.state'],
                 'state_joint_names': self.names, 'action_joint_names': self.names,
                 'action_semantics': 'linear interpolation of issued joint trajectory at image time + 1/fps',
                 'oracle_coordinates_in_model_input': False}
        audit['quality_pass'] = bool(self.fps == 10 and self.frames_written >= 100 and
                                     self.frames_received == self.frames_written and
                                     self.frames_rejected == 0 and len(intervals) and
                                     max(intervals) <= .11 and max(self.offsets) <= 30.)
        if self.out:
            (self.out / 'sync_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        self.out = self.writer = None
        return audit
