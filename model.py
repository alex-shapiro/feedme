from typing import final, override

import mlx.core as mx
from mlx import nn

from categorical import Categorical


class FeedmeNet(nn.Module):
    pass


@final
class PolicyNet(nn.Module):
    def __init__(self):
        # input shape: [B, 30, 2]
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        # Flattened 32 * 30 = 960
        self.linear1 = nn.Linear(960, 64)
        self.linear2 = nn.Linear(64, 3)

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
        x = self.conv1(obs)
        x = nn.leaky_relu(x)
        x = self.conv2(x)
        x = nn.leaky_relu(x)
        x = mx.flatten(x, start_axis=1)
        x = self.linear1(x)
        x = nn.leaky_relu(x)
        logits = self.linear2(x)
        return Categorical(logits)


@final
class ValueNet(nn.Module):
    def __init__(self):
        # input shape: [B, 30, 2]
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        # Flattened 32 * 30 = 960
        self.linear1 = nn.Linear(960, 64)
        self.linear2 = nn.Linear(64, 1)

    @override
    def __call__(self, obs: mx.array) -> mx.array:
        x = self.conv1(obs)
        x = nn.leaky_relu(x)
        x = self.conv2(x)
        x = nn.leaky_relu(x)
        x = mx.flatten(x, start_axis=1)
        x = self.linear1(x)
        x = nn.leaky_relu(x)
        return self.linear2(x)


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
