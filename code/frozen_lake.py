import gym
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

HIDDEN_SIZE = 128
BATCH_SIZE = 200
PERCENTILE = 70
GAMMA = 0.90


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert (env.observation_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (env.observation_space.n,), dtype=np.float32
        )

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple("Episode", field_names=["reward", "steps"])
EpisodeStep = namedtuple("EpisodeStep", field_names=["observation", "act"])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.FloatTensor([obs]).to(device)
        act_probs_v = sm(net(obs_v))
        action = torch.multinomial(act_probs_v, 1).item()

        next_obs, rewa, is_done, _ = env.step(action)

        episode_steps.append(EpisodeStep(observation=obs, act=action))

        if is_done:
            batch.append(
                Episode(
                    reward=rewa * (GAMMA ** len(episode_steps)), steps=episode_steps
                )
            )
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = [episode.reward for episode in batch]

    reward_bound = np.percentile(rewards, percentile)

    train_obs = []
    train_act = []
    elite_batch = []

    for episode in batch:
        if episode.reward > reward_bound:
            train_obs.extend([step.observation for step in episode.steps])
            train_act.extend([step.act for step in episode.steps])
            elite_batch.append(episode)

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)

    return elite_batch, train_obs_v, train_act_v, reward_bound


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    env = DiscreteOneHotWrapper(env)

    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions).to(device)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-frozen_lake")

    full_batch = []

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):

        reward_mean = float(np.mean([episode.reward for episode in batch]))

        full_batch, obs_v, acts_v, reward_b = filter_batch(
            full_batch + batch, PERCENTILE
        )

        if not full_batch:
            continue

        full_batch = full_batch[-500:]

        action_scores_v = net(obs_v.to(device))

        loss_v = objective(action_scores_v, acts_v.to(device))

        optimizer.zero_grad()
        loss_v.backward()

        optimizer.step()
        print(
            "{:d} loss={:.3f}, reward_mean={:.3f}, reward_bound={:.3f}, batch {:d}".format(
                iter_no, loss_v.item(), reward_mean, reward_b, len(full_batch)
            )
        )
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        if reward_mean > 0.7:
            print("Solved!")
            break
    writer.close()

