"""Read-only teacher targets bounded like the rollout joint-step limiter.

Labels remain absolute joint positions. Future labels use their own recorded
observations, never the current observation or another episode's state.
This transforms supervision only; future states are never policy inputs.
"""
import torch


class BoundedActionTargets(torch.utils.data.Dataset):
    limit_rad = .1

    def __init__(self, base):
        self.base = base
        self.meta, self.hf_dataset = base.meta, base.hf_dataset
        self.episode_data_index = base.episode_data_index
        def column(name):
            values = self.hf_dataset[name]
            return (values.to(dtype=torch.float32) if torch.is_tensor(values)
                    else torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in values]))
        states, actions = column('observation.state'), column('action')
        if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != 5:
            raise ValueError('Expected matching five-joint state/action columns')
        if not torch.isfinite(states).all() or not torch.isfinite(actions).all():
            raise ValueError('Nonfinite teacher state/action')
        # Rollout caps all five joints, including the hand.
        self.targets = states + (actions - states).clamp(-self.limit_rad, self.limit_rad)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        ep = int(item['episode_index'])
        end = int(self.episode_data_index['to'][ep])
        count = item['action'].shape[0]
        future = torch.arange(index, index + count).clamp(max=end - 1)
        item['action'] = self.targets[future].clone()
        return item

    def train_stats(self, episode_ids):
        selected = torch.cat([self.targets[int(self.episode_data_index['from'][ep]):
                                           int(self.episode_data_index['to'][ep])]
                              for ep in episode_ids])
        return {'mean': selected.mean(0).numpy(),
                'std': selected.std(0, unbiased=False).clamp_min(1e-6).numpy(),
                'min': selected.amin(0).numpy(), 'max': selected.amax(0).numpy(),
                'count': torch.tensor([len(selected)]).numpy()}
