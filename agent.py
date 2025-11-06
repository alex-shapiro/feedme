from dataclasses import dataclass
from typing import Any, final

import mlx.core as mx
from mlx.optimizers import AdamW

from env import Action, FeedMeEnv
from model import EaterNet
from trajectory_buffer import TrajectoryBatch, TrajectoryBuffer

type Gradients = dict[str, Any]


@final
class FeedMeAgent:
    def __init__(
        self,
        n_epochs: int,
        n_steps_per_epoch: int = 512,
        n_policy_training_iters: int = 80,
        n_value_training_iters: int = 80,
        gamma: float = 0.99,
        lamda: float = 0.99,
        clip_ratio: float = 0.2,
        policy_lr: float = 1e-3,
        value_lr: float = 1e-3,
        target_kl: float = 0.05,
    ):
        super().__init__()
        self.n_epochs = n_epochs
        self.n_steps_per_epoch = n_steps_per_epoch
        self.n_policy_training_iters = n_policy_training_iters
        self.n_value_training_iters = n_value_training_iters
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

        # simulation env
        self.env = FeedMeEnv()

        # model
        self.model = EaterNet()
        self.policy_optimizer = AdamW(learning_rate=policy_lr)
        self.value_optimizer = AdamW(learning_rate=value_lr)

        # trajectory buffer
        self.trajectories_a = TrajectoryBuffer(
            capacity=n_steps_per_epoch,
            gamma=gamma,
            lamda=lamda,
        )
        self.trajectories_b = TrajectoryBuffer(
            capacity=n_steps_per_epoch,
            gamma=gamma,
            lamda=lamda,
        )

    def train(self):
        for epoch in range(1, self.n_epochs + 1):
            print(f"\nEpoch {epoch}")

            # build rollouts
            (obs_a, obs_b) = self.env.reset()
            final_step = self.n_steps_per_epoch - 1
            for t in range(self.n_steps_per_epoch):
                action_a, logp_a, value_a = self.model.step(obs_a)
                action_b, logp_b, value_b = self.model.step(obs_b)
                (next_obs_a, next_obs_b), (reward_a, reward_b), done = self.env.step(
                    action_a,
                    action_b,
                )
                self.trajectories_a.push(
                    obs=obs_a,
                    action=action_a,
                    logp=logp_a,
                    value=value_a,
                    reward=reward_a,
                )
                self.trajectories_b.push(
                    obs=obs_b,
                    action=action_b,
                    logp=logp_b,
                    value=value_b,
                    reward=reward_b,
                )
                obs_a = next_obs_a
                obs_b = next_obs_b
                truncated = t == final_step
                if done or truncated:
                    value_a = self.model.value(obs_a) if truncated else 0.0
                    value_b = self.model.value(obs_b) if truncated else 0.0
                    self.trajectories_a.push_episode_end(value_a, truncated=truncated)
                    self.trajectories_b.push_episode_end(value_b, truncated=truncated)
                    obs_a, obs_b = self.env.reset()

            self.update()

            if epoch % 10 == 0:
                self.evaluate(n_episodes=100)

    def update(self):
        batch_a = self.trajectories_a.get_batch()
        batch_b = self.trajectories_b.get_batch()
        batch = batch_a.concat(batch_b)
        assert len(batch.obs) == 1024

        policy_loss_val = 0.0
        policy_grad_norm = 0.0
        for i in range(self.n_policy_training_iters):
            loss, policy_info, grads = self.policy_loss(batch)
            self.policy_optimizer.update(self.model.p_net, grads)
            if policy_info.approximate_kl > 1.5 * self.target_kl:
                print(
                    f"stopping early at iter {i} for reaching max KL (value ~{policy_info.approximate_kl:.4f})"
                )
                break

        for i in range(self.n_value_training_iters):
            loss, grads = self.value_loss(batch)
            self.value_optimizer.update(self.model.v_net, grads)
            mx.eval(self.model.v_net.parameters())

    def policy_loss(
        self, batch: TrajectoryBatch
    ) -> tuple[float, "PolicyInfo", Gradients]:
        pass

    def value_loss(self, batch: TrajectoryBatch) -> tuple[float, Gradients]:
        pass

    def evaluate(self, n_episodes: int):
        ep_rewards_a = []
        ep_rewards_b = []
        for i in range(n_episodes):
            ra = 0.0
            rb = 0.0
            obs_a, obs_b = self.env.reset()
            done = False
            while not done:
                action_a = int(self.model.p_net.policy(obs_a).sample().item())
                action_b = int(self.model.p_net.policy(obs_b).sample().item())
                (obs_a, obs_b), (reward_a, reward_b), done = self.env.step(
                    action_a, action_b
                )
                ra += reward_a
                rb += reward_b
                if i == 0:
                    print(f"A: {Action(action_a)}, B: {Action(action_b)}")
            ep_rewards_a.append(ra)
            ep_rewards_b.append(rb)
        ep_rewards_a = mx.array(ep_rewards_a)
        ep_rewards_b = mx.array(ep_rewards_b)
        print(f"A reward: mean {mx.mean(ep_rewards_a):.4f} +/- {mx.std(ep_rewards_a)}")
        print(f"B reward: mean {mx.mean(ep_rewards_b):.4f} +/- {mx.std(ep_rewards_b)}")


@dataclass
class PolicyInfo:
    approximate_kl: float
    mean_entropy: float
    clipped_fraction: float


if __name__ == "__main__":
    agent = FeedMeAgent(n_epochs=100)
    agent.train()
    agent.evaluate(n_episodes=100)
