from enum import Enum


class Action(Enum):
    FeedSelf = 0
    FeedOther = 1
    OpenMouth = 2
    Signal = 3


class Env:
    def __init__(self):
        self.t = 0
        self.timeout = 30

    def reset(self):
        self.t = 0

    def step(self, action_a: int, action_b: int) -> tuple[tuple[int, int], bool]:
        assert self.t < self.timeout
        reward_a = self.get_reward(action_a, action_b)
        reward_b = self.get_reward(action_b, action_a)
        self.t += 1
        reward = (reward_a, reward_b)
        done = self.t == self.timeout
        return reward, done

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
