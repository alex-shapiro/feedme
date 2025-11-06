# FeedMe

There are 2 actors. Each turn, A and B actors decide on their actions independently and then the environment steps once:

- FeedSelf (reward = 1)
- FeedOther
- OpenMouth (reward = 10 iff the other agent chooses FeedOther)
- Signal

Goal is to train agents that perform tit-for-tat behavior
