"""Recorded-command diagnostic provider; never a learned policy."""
import bisect
import hashlib
import json
from pathlib import Path
import numpy as np


class TeacherReplay:
    def __init__(self, episode, seed, time_scale=1., observed_next_state=False):
        if not np.isfinite(time_scale) or not 1. <= time_scale <= 5.:
            raise ValueError('Diagnostic replay time scale must be within [1, 5]')
        self.time_scale = float(time_scale)
        self.episode = Path(episode)
        result = json.loads((self.episode / 'result.json').read_text())
        if result.get('seed') != seed or not result.get('accepted_for_training'):
            raise ValueError('Replay requires an accepted teacher with the exact source seed')
        rows = [json.loads(x) for x in (self.episode / 'frames.jsonl').read_text().splitlines()]
        self.times = [(r['sim_time_ns'] - rows[0]['sim_time_ns']) / 1e9 for r in rows]
        self.actions = np.asarray([r['action'] for r in rows], dtype=float)
        if observed_next_state:
            states = np.asarray([r['observation.state'] for r in rows], dtype=float)
            self.actions = np.concatenate([states[1:], states[-1:]], axis=0)
        if (len(rows) < 2 or np.any(np.diff(self.times) <= 0)
                or self.actions.shape != (len(rows), 5) or not np.isfinite(self.actions).all()):
            raise ValueError('Invalid replay time/action sequence')
        self.initial_state = np.asarray(rows[0]['observation.state'], dtype=float)
        if self.initial_state.shape != (5,) or not np.isfinite(self.initial_state).all():
            raise ValueError('Invalid recorded initial joints')
        self.initial_mouth = result.get('initial_mouth_xyz_m')
        self.metadata = {'source_episode': self.episode.name, 'seed': seed,
                         'frames_sha256': hashlib.sha256((self.episode / 'frames.jsonl').read_bytes()).hexdigest(),
                         'frames': len(rows), 'duration_sim_s': self.times[-1] * self.time_scale,
                         'time_scale': self.time_scale,
                         'target_source': 'next_recorded_observed_joints' if observed_next_state else 'recorded_interpolated_command',
                         'future_state_in_policy_input': False,
                         'selection': 'last recorded target at elapsed simulation time, final target held two seconds',
                         'learned_model': False, 'model_success': False}

    def action(self, elapsed_sim_s):
        source_time = round(elapsed_sim_s / self.time_scale, 9)  # source timestamps have ns resolution
        index = max(0, bisect.bisect_right(self.times, source_time) - 1)
        return self.actions[index][[0, 1, 3, 2, 4]].copy()  # raw -> course

    def finished(self, elapsed_sim_s):
        return elapsed_sim_s >= self.times[-1] * self.time_scale + 2.
