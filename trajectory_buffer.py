from dataclasses import dataclass
from typing import final

import mlx.core as mx


@final
class TrajectoryBuffer:
    def __init__(
        self,
        capacity: int,
        obs_space_shape: tuple[int, int],
        gamma: float = 0.99,
        lamda: float = 0.95,
    ):
        # capacity
        self.capacity = capacity
        # index for the next insert
        self.next_index = 0
        # environment observations
        self.obs = mx.zeros([capacity, *obs_space_shape], dtype=mx.float32)
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
        # index for the start of the current episode
        self.episode_start_index = 0

    def push(
        self,
        obs: mx.array,
        action: int,
        logp: float,
        value: float,
        reward: float,
    ):
        self.obs[self.next_index] = obs
        self.actions[self.next_index] = action
        self.logps[self.next_index] = logp
        self.values[self.next_index] = value
        self.rewards[self.next_index] = reward
        self.next_index += 1

    def push_episode_end(
        self,
        value: float,
        truncated: bool,
    ):
        end = self.capacity if self.next_index == 0 else self.next_index
        range = slice(self.episode_start_index, end)
        ep_rewards = self.rewards[range]
        ep_values = self.values[range]

        # TD error
        bootstrap_value = value if truncated else 0.0
        next_values = mx.concatenate([ep_values[1:], mx.array([bootstrap_value])])
        deltas = ep_rewards + self.gamma * next_values - ep_values

        # GAE-Lambda advantage
        self.advantages[range] = cumulative_sum(deltas, self.gamma * self.lamda)
        # Return
        if truncated:
            ep_rewards = mx.concatenate([ep_rewards, mx.array([bootstrap_value])])
            self.returns[range] = cumulative_sum(ep_rewards, self.gamma)[:-1]
        else:
            self.returns[range] = cumulative_sum(ep_rewards, self.gamma)

        # Move the episode pointer
        self.episode_start_index = self.next_index

    def get_batch(self) -> "TrajectoryBatch":
        """Sample N random elements from the trajectory buffer"""
        assert self.next_index == self.capacity
        advantages = self.advantages
        advantage_mean = mx.mean(advantages)
        advantage_std = mx.std(advantages)
        # Add small eps to prevent division by zero
        advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)
        self.next_index = 0
        self.episode_start_index = 0
        return TrajectoryBatch(
            obs=self.obs,
            actions=self.actions,
            returns=self.returns,
            advantages=advantages,
            logps=self.logps,
        )

    def __len__(self):
        return self.next_index


@dataclass
class TrajectoryBatch:
    obs: mx.array
    actions: mx.array
    advantages: mx.array
    logps: mx.array
    returns: mx.array

    def concat(self, other: "TrajectoryBatch") -> "TrajectoryBatch":
        return TrajectoryBatch(
            obs=mx.concat([self.obs, other.obs]),
            actions=mx.concat([self.actions, other.actions]),
            advantages=mx.concat([self.advantages, other.advantages]),
            logps=mx.concat([self.logps, other.logps]),
            returns=mx.concat([self.returns, other.returns]),
        )


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
