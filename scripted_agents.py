"""Scripted agents for curriculum learning"""

import mlx.core as mx

from env import Action


class TitForTatAgent:
    """
    Implements a tit-for-tat strategy:
    - Start with cooperation (FeedOther on turn 0)
    - On odd turns: OpenMouth (expecting to be fed)
    - On even turns: FeedOther (feeding the opponent)
    - If opponent defects (doesn't feed when I have OpenMouth), defect back
    """

    def __init__(self, start_with_feed: bool = True):
        """
        Args:
            start_with_feed: If True, starts by feeding (turn 0 = FeedOther)
                           If False, starts by opening mouth (turn 0 = OpenMouth)
        """
        self.start_with_feed = start_with_feed
        self.defected = False  # Track if we've detected defection

    def reset(self):
        """Reset the agent's state for a new episode"""
        self.defected = False

    def step(self, obs: mx.array) -> int:
        """
        Select action based on observation history.

        Args:
            obs: Observation array [T, 2] where obs[i] = [my_action, opponent_action]

        Returns:
            Action index (0=FeedSelf, 1=FeedOther, 2=OpenMouth)
        """
        # Find current timestep (first row with -1 values)
        t = 0
        for i in range(obs.shape[0]):
            if float(obs[i, 0]) == -1.0:
                break
            t = i + 1

        # First turn: start with cooperation pattern
        if t == 0:
            return (
                Action.FeedOther.value
                if self.start_with_feed
                else Action.OpenMouth.value
            )

        # Check if opponent defected last turn
        last_my_action = int(obs[t - 1, 0])
        last_opponent_action = int(obs[t - 1, 1])

        # If I had OpenMouth last turn and opponent didn't FeedOther, they defected
        if (
            last_my_action == Action.OpenMouth.value
            and last_opponent_action != Action.FeedOther.value
        ):
            self.defected = True

        # If opponent defected, punish by always doing FeedSelf
        if self.defected:
            return Action.FeedSelf.value

        # Otherwise, continue tit-for-tat pattern
        # Alternate: FeedOther on even turns (0, 2, 4...), OpenMouth on odd turns (1, 3, 5...)
        if self.start_with_feed:
            return Action.FeedOther.value if t % 2 == 0 else Action.OpenMouth.value
        else:
            return Action.OpenMouth.value if t % 2 == 0 else Action.FeedOther.value


class AlwaysCooperateAgent:
    """Agent that always cooperates by alternating FeedOther and OpenMouth"""

    def __init__(self, start_with_feed: bool = True):
        self.start_with_feed = start_with_feed

    def reset(self):
        pass

    def step(self, obs: mx.array) -> int:
        # Find current timestep
        t = 0
        for i in range(obs.shape[0]):
            if float(obs[i, 0]) == -1.0:
                break
            t = i + 1

        if self.start_with_feed:
            return Action.FeedOther.value if t % 2 == 0 else Action.OpenMouth.value
        else:
            return Action.OpenMouth.value if t % 2 == 0 else Action.FeedOther.value


class AlwaysDefectAgent:
    """Agent that always defects (FeedSelf)"""

    def reset(self):
        pass

    def step(self, obs: mx.array) -> int:
        return Action.FeedSelf.value
