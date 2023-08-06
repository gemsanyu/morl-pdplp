import torch 
from torch.distributions.utils import probs_to_logits

class Categorical:
    def __init__(self, probs_shape): 
        # NOTE: probs_shape is supposed to be 
        #       the shape of probs that will be 
        #       produced by policy network
        if len(probs_shape) < 1: 
            raise ValueError("`probs_shape` must be at least 1.")
        self.probs_dim = len(probs_shape) 
        self.probs_shape = probs_shape
        self._num_events = probs_shape[-1]
        self._batch_shape = probs_shape[:-1] if self.probs_dim > 1 else torch.Size()
        self._event_shape=torch.Size()

    def set_probs_(self, probs):
        self.probs = probs
        self.logits = probs_to_logits(self.probs)

    def set_probs(self, probs):
        self.probs = probs / probs.sum(-1, keepdim=True) 
        self.logits = probs_to_logits(self.probs)


    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape(sample_shape + self._batch_shape + self._event_shape)

    def log_prob(self, value):
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)
