from agent import FeedMeAgent

if __name__ == "__main__":
    agent = FeedMeAgent(n_epochs=100)
    agent.load_latest_model()
    agent.evaluate(n_episodes=1)
