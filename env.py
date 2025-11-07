from enum import Enum
from typing import final, override

import mlx.core as mx


class Action(Enum):
    FeedSelf = 0
    FeedOther = 1
    OpenMouth = 2

    @override
    def __repr__(self) -> str:
        match self:
            case Action.FeedSelf:
                return "FeedSelf"
            case Action.FeedOther:
                return "FeedOther"
            case Action.OpenMouth:
                return "OpenMouth"


@final
class FeedMeEnv:
    def __init__(self):
        self.timeout = 100
        self.reset()

    def obs_space_shape(self) -> tuple[int, int]:
        return self.obs.shape  # pyright: ignore[reportReturnType]

    def action_space_n(self) -> int:
        return len(Action)

    def reset(self) -> tuple[mx.array, mx.array]:
        self.t = 0
        self.obs = mx.zeros([self.timeout, 2], dtype=mx.float32) - 1.0
        return self.observation()

    def observation(self) -> tuple[mx.array, mx.array]:
        a = self.obs
        b = self.obs[:, [1, 0]]
        return a, b

    def step(
        self,
        action_a: int,
        action_b: int,
    ) -> tuple[tuple[mx.array, mx.array], tuple[int, int], bool]:
        assert self.t < self.timeout
        self.obs[self.t, 0] = float(action_a)
        self.obs[self.t, 1] = float(action_b)

        reward_a = self.get_reward(action_a, action_b)
        reward_b = self.get_reward(action_b, action_a)
        self.t += 1
        reward = (reward_a, reward_b)
        done = self.t == self.timeout
        return self.observation(), reward, done

    def get_reward(self, action_a: int, action_b: int) -> int:
        assert action_a >= 0 and action_a <= 2
        assert action_b >= 0 and action_b <= 2
        a = Action(action_a)
        b = Action(action_b)
        match a:
            case Action.FeedSelf:
                return 1
            case Action.FeedOther:
                return 0
            case Action.OpenMouth:
                return 10 if b == Action.FeedOther else 0
