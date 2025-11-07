from typing import final, override

import mlx.core as mx
from mlx import nn

from categorical import Categorical


class FeedmeNet(nn.Module):
    pass


@final
class PolicyNet(nn.Module):
    def __init__(self, history_length: int = 5):
        # input shape: [B, T, 2], but we only use last history_length steps
        super().__init__()
        self.history_length = history_length
        # Flattened: history_length * 2 features
        input_size = history_length * 2
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 3)

    @override
    def __call__(
        self,
        obs: mx.array,
        actions: mx.array | None,
    ) -> tuple[Categorical, mx.array | None]:
        policy = self.policy(obs)
        logps = None if actions is None else policy.log_prob(actions)
        return policy, logps

    def policy(self, obs: mx.array) -> Categorical:
        # Extract recent history: [B, T, 2] -> [B, history_length, 2]
        x = obs[:, -self.history_length :, :]
        # Flatten: [B, history_length, 2] -> [B, history_length * 2]
        x = mx.flatten(x, start_axis=1)
        x = self.linear1(x)
        x = nn.leaky_relu(x)
        x = self.linear2(x)
        x = nn.leaky_relu(x)
        x = self.linear3(x)
        x = nn.leaky_relu(x)
        logits = self.linear4(x)
        return Categorical(logits)


@final
class ValueNet(nn.Module):
    def __init__(self, history_length: int = 5):
        # input shape: [B, T, 2], but we only use last history_length steps
        super().__init__()
        self.history_length = history_length
        # Flattened: history_length * 2 features
        input_size = history_length * 2
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 1)

    @override
    def __call__(self, obs: mx.array) -> mx.array:
        # Extract recent history: [B, T, 2] -> [B, history_length, 2]
        x = obs[:, -self.history_length :, :]
        # Flatten: [B, history_length, 2] -> [B, history_length * 2]
        x = mx.flatten(x, start_axis=1)
        x = self.linear1(x)
        x = nn.leaky_relu(x)
        x = self.linear2(x)
        x = nn.leaky_relu(x)
        x = self.linear3(x)
        x = nn.leaky_relu(x)
        return self.linear4(x)


class EaterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_net = PolicyNet()
        self.v_net = ValueNet()

    def step(self, obs: mx.array) -> tuple[int, float, float]:
        # Add batch dimension if missing
        if obs.ndim == 2:
            obs = mx.expand_dims(obs, axis=0)
        policy = self.p_net.policy(obs)
        action = policy.sample()
        logp = policy.log_prob(action)
        value = self.v_net(obs)
        return (int(action.item()), float(logp.item()), float(value.item()))

    def value(self, obs: mx.array) -> float:
        # Add batch dimension if missing
        if obs.ndim == 2:
            obs = mx.expand_dims(obs, axis=0)
        return float(self.v_net(obs))
