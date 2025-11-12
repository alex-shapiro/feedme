import random

import mlx.core as mx

from env import Action


class TitForTatAgent:
    """
    Deterministic agent with a tit-for-tat strategy
    - Start with cooperation (FeedOther on turn 0)
    - If opponent defects (does not pick FeedOther after OpenMouth), defect back
    """

    def __init__(self, tolerance: int = 2, randomness: float = 0.01):
        """
        Args:
            start_with_feed: If True, starts by feeding (turn 0 = FeedOther)
                           If False, starts by opening mouth (turn 0 = OpenMouth)
            tolerance_defections: Number of defections to tolerate before retaliating
        """
        self.tolerance = tolerance
        self.defection_count = 0
        self.randomness = randomness

    def reset(self):
        """Reset the agent's state for a new episode"""
        self.defection_count = 0

    def step(self, obs: mx.array) -> int:
        """Select action based on observation history"""
        t = count_valid_timesteps(obs)

        # First turn: start with cooperation pattern
        if t == 0:
            return Action.FeedOther.value

        # Check if opponent defected last turn
        my_action = Action(int(obs[t - 1, 0]))
        opponent_action = Action(int(obs[t - 1, 1]))

        if opponent_action == Action.FeedSelf:
            # If the opponent did not defect last time, reduce their count
            self.defection_count = max(self.defection_count - 1, 0)
        elif my_action == Action.FeedOther:
            # If the opponent defected last time and I did not, increase their count
            self.defection_count = min(self.defection_count + 1, 2)

        # If opponent defected more than tolerance, punish by always doing FeedSelf
        if self.defection_count > self.tolerance:
            return Action.FeedSelf.value

        if random.random() <= self.randomness:
            return Action.OpenMouth.value
        else:
            return Action.FeedOther.value


def count_valid_timesteps(obs: mx.array) -> int:
    """Count number of valid (non -1) timesteps in observation"""
    t = 0
    for i in range(obs.shape[0]):
        if float(obs[i, 0]) == -1.0:
            break
        t += 1
    return t
