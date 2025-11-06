from typing import final

import mlx.core as mx


@final
class Categorical:
    """Categorical distribution for sampling actions from logits"""

    logits: mx.array
    log_probs: mx.array
    probs: mx.array

    def __init__(self, logits: mx.array):
        """Initialize categorical distribution from logits"""
        self.logits = logits
        self.log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        self.probs = mx.exp(self.log_probs)

    def sample(self) -> mx.array:
        """Sample from the categorical distribution"""
        return mx.random.categorical(self.logits, axis=-1)

    def log_prob(self, value: mx.array) -> mx.array:
        """Compute log probability of given values"""
        if value.ndim == 0:
            # Single sample case
            return self.log_probs[int(value)]
        else:
            # Batched case - gather log probs at specified indices
            batch_size = value.shape[0]
            batch_indices = mx.arange(batch_size)
            return self.log_probs[batch_indices, value]

    def entropy(self) -> mx.array:
        """Compute distribution entropy: -sum(p * log(p))"""
        safe_probs = mx.clip(self.probs, 1e-10, 1.0)
        return -(self.probs * mx.log(safe_probs)).sum(axis=-1)
