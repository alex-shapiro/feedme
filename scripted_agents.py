import mlx.core as mx

from env import Action


class TitForTatAgent:
    """
    Deterministic agent with a tit-for-tat strategy
    - Start with cooperation (FeedOther on turn 0)
    - On odd turns: OpenMouth (expecting to be fed)
    - On even turns: FeedOther (feeding the opponent)
    - If opponent defects (does not pick FeedOther after OpenMouth), defect back
    """

    start_with_feed: bool
    defection_count: int
    tolerance_defections: int

    def __init__(self, start_with_feed: bool = True, tolerance_defections: int = 2):
        """
        Args:
            start_with_feed: If True, starts by feeding (turn 0 = FeedOther)
                           If False, starts by opening mouth (turn 0 = OpenMouth)
            tolerance_defections: Number of defections to tolerate before retaliating
        """
        self.start_with_feed = start_with_feed
        self.tolerance_defections = tolerance_defections
        self.defection_count = 0  # Track number of opponent defections

    def reset(self):
        """Reset the agent's state for a new episode"""
        self.defection_count = 0

    def step(self, obs: mx.array) -> int:
        """
        Select action based on observation history.

        Args:
            obs: Observation array [T, 3] where obs[i] = [my_action, opponent_action, is_episode_start]

        Returns:
            Action index (0=FeedSelf, 1=FeedOther, 2=OpenMouth)
        """
        t = count_valid_timesteps(obs)

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
            self.defection_count += 1

        # If opponent defected more than tolerance, punish by always doing FeedSelf
        if self.defection_count > self.tolerance_defections:
            return Action.FeedSelf.value

        # Otherwise, continue tit-for-tat pattern
        # Alternate: FeedOther on even turns (0, 2, 4...), OpenMouth on odd turns (1, 3, 5...)
        if self.start_with_feed:
            return Action.FeedOther.value if t % 2 == 0 else Action.OpenMouth.value
        else:
            return Action.OpenMouth.value if t % 2 == 0 else Action.FeedOther.value


def count_valid_timesteps(obs: mx.array) -> int:
    """Count number of valid (non -1) timesteps in observation"""
    t = 0
    for i in range(obs.shape[0]):
        if float(obs[i, 0]) == -1.0:
            break
        t += 1
    return t
