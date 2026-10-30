"""Causal joint motion cues; no target actions or future observations."""
import torch


def velocity_cue(current, previous):
    if current.shape != previous.shape or current.shape[-1] != 5:
        raise ValueError('Joint history expects matching five-joint observations')
    # Recorded and rollout observations are nominally 10Hz. Clip and scale
    # radians/second to keep the ENV identity-normalized token bounded.
    return ((current - previous) * 10. / 4.).clamp(-1., 1.)


class JointHistoryStream:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last = None

    def previous(self, current):
        return current.clone() if self.last is None else self.last.clone()

    def commit(self, current):
        self.last = current.clone()


class CausalJointHistory(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
        self.meta, self.hf_dataset = base.meta, base.hf_dataset
        self.episode_data_index = base.episode_data_index
        column = self.hf_dataset['observation.state']
        self.states = (column.to(dtype=torch.float32) if torch.is_tensor(column)
                       else torch.stack([torch.as_tensor(q, dtype=torch.float32) for q in column]))
        self.previous_indices = torch.arange(len(base)).sub(1).clamp(min=0)
        for start in self.episode_data_index['from']:
            self.previous_indices[int(start)] = int(start)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        # Use the immediately previous raw frame, even when that frame was
        # excluded from loss; never concatenate across masks or episodes.
        item['observation.previous_state'] = self.states[self.previous_indices[index]].clone()
        return item
