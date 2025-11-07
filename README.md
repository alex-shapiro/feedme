# FeedMe

There are 2 actors. Each turn, A and B actors decide on their actions independently and then the environment steps once:

- FeedSelf (reward = 1)
- FeedOther
- OpenMouth (reward = 10 iff the other agent chooses FeedOther)
- Signal

Goal is to train agents that perform tit-for-tat behavior

## Dev Log

- V1: no noticeable training over time
- V2: training leads both agents to keep their mouth open at all times (0 reward)
- V3: remove Signal action and calculate policy loss wrt policy entropy (slight boost to value over greedy strategy, but it stops after rewards ~36)
- V4: move to a pure MLP model architecture and increase episode length from 30 to 100 (no improvement)
- V5: use stochastic episode duration to avoid inductive defection
- V6: switch to a transformer model architecture
