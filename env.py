from enum import Enum
from typing import final

import mlx.core as mx


class Action(Enum):
    FeedSelf = 0
    FeedOther = 1
    OpenMouth = 2
    Signal = 3


@final
class FeedMeEnv:
    def __init__(self):
        self.timeout = 30
        self.reset()

    def reset(self):
        self.t = 0
        self.state = mx.zeros([self.timeout, 2], dtype=mx.int8) - 1

    def step(
        self,
        action_a: int,
        action_b: int,
    ) -> tuple[mx.array, tuple[int, int], bool]:
        assert self.t < self.timeout
        self.state[self.t, 0] = action_a
        self.state[self.t, 0] = action_b

        reward_a = self.get_reward(action_a, action_b)
        reward_b = self.get_reward(action_b, action_a)
        self.t += 1
        reward = (reward_a, reward_b)
        done = self.t == self.timeout
        return self.state, reward, done

    def get_reward(self, action_a: int, action_b: int) -> int:
        assert action_a >= 0 and action_a <= 3
        assert action_b >= 0 and action_b <= 3
        a = Action(action_a)
        b = Action(action_b)
        match a:
            case Action.FeedSelf:
                return 1
            case Action.FeedOther:
                return 0
            case Action.OpenMouth:
                return 10 if b == Action.FeedOther else 0
            case Action.Signal:
                return 0
