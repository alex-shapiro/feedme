from dataclasses import dataclass
from typing import final

import mlx.core as mx


@final
class TrajectoryBuffer:
    def __init__(
        self,
        capacity: int,
        gamma: float = 0.99,
        lamda: float = 0.95,
    ):
        # capacity
        self.capacity = capacity
        # environment observations
        self.obs = mx.zeros([capacity, 30, 2], dtype=mx.float32)
        # predicted actions
        self.actions = mx.zeros([capacity], dtype=mx.int8)
        # action log probabilities
        self.logps = mx.zeros(capacity, dtype=mx.float32)
        # predicted env state values
        self.values = mx.zeros(capacity, dtype=mx.float32)
        # action rewards
        self.rewards = mx.zeros(capacity, dtype=mx.float32)
        # action advantages
        self.advantages = mx.zeros(capacity, dtype=mx.float32)
        # discounted cumulative future rewards for the state
        self.returns = mx.zeros(capacity, dtype=mx.float32)
        # discount factor
        self.gamma = gamma
        # value estimation discount
        self.lamda = lamda
        # index for the next insert
        self.next_index = 0
        # index for the start of the current episode
        self.episode_start_index = 0
        # total insert length
        self.inserts = 0

    def push(
        self,
        obs: mx.array,
        action: int,
        action_mask: mx.array,
        logp: float,
        value: float,
        reward: float,
    ):
        self.obs[self.next_index] = obs
        self.actions[self.next_index] = action
        self.logps[self.next_index] = logp
        self.values[self.next_index] = value
        self.rewards[self.next_index] = reward
        self.next_index = (self.next_index + 1) % self.capacity
        self.inserts += 1

    def push_episode_end(
        self,
        value: float,
        is_truncated: bool,
        is_self_play: bool = True,
    ):
        end = self.capacity if self.next_index == 0 else self.next_index
        range = slice(self.episode_start_index, end)
        ep_rewards = self.rewards[range]
        ep_values = self.values[range]

        # TD error
        bootstrap_value = value if is_truncated else 0.0
        next_values = mx.concatenate([ep_values[1:], mx.array([bootstrap_value])])

        # For self-play, use negative gamma because rewards alternate perspective
        # For non-self-play, use positive gamma because we only track our own rewards
        gamma_sign = -self.gamma if is_self_play else self.gamma
        deltas = ep_rewards + gamma_sign * next_values - ep_values

        # GAE-Lambda advantage
        self.advantages[range] = cumulative_sum(deltas, gamma_sign * self.lamda)
        # Return
        if is_truncated:
            ep_rewards = mx.concatenate([ep_rewards, mx.array(bootstrap_value)])
            self.returns[range] = cumulative_sum(ep_rewards, gamma_sign)[:-1]
        else:
            self.returns[range] = cumulative_sum(ep_rewards, gamma_sign)

        # Move the episode pointer
        self.episode_start_index = self.next_index

    def get_batch(self) -> "TrajectoryBatch":
        """Sample N random elements from the trajectory buffer"""
        advantages = self.advantages
        advantage_mean = mx.mean(advantages)
        advantage_std = mx.std(advantages)
        # Add small eps to prevent division by zero
        advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)
        return TrajectoryBatch(
            obs=self.obs,
            actions=self.actions,
            returns=self.returns,
            advantages=advantages,
            logps=self.logps,
        )

    def __len__(self):
        return min(self.inserts, self.capacity)


@dataclass
class TrajectoryBatch:
    obs: mx.array
    actions: mx.array
    advantages: mx.array
    logps: mx.array
    returns: mx.array


def cumulative_sum(x: mx.array, gamma: float) -> mx.array:
    """
    Returns the discounted cumulative sum of vector elements
    Example: cs([1,2,3], 0.95) => [5.59325, 4.835, 3]
    """
    result = mx.zeros_like(x)
    result[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        result[i] = x[i] + gamma * result[i + 1]
    return result
