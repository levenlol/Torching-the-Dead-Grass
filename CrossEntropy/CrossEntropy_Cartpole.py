HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70 # keeping the best 30% episodes

from collections import namedtuple
import torch.nn as nn
import torch
import numpy as np
import gym
import os
import torch.utils.tensorboard as ts

class DiscreteOneHotObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotObservationWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n,), dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
            # dont use Softmax, CrossEntropy from pytorch that combing the Exp of softmax and the Log of cross entropy in a more numerical stable way.
        )

    def forward(self, x):
        return self.net(x)

    
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

video_recorder = None

def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]

        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)

        episode_reward += reward
        episode_steps.append(EpisodeStep(obs, action))

       #video_recorder.capture_frame()

        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        
        obs = next_obs

def filter_batch(batch, percentile):
    rewards = list(map(lambda s : s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    #env = DiscreteOneHotObservationWrapper(gym.make("FrozenLake-v1"))
    #path_project = os.path.abspath(os.path.join(__file__, ".."))
    #path_of_video_with_name = os.path.join(path_project, "videotest.mp4")
    #video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, path=path_of_video_with_name, enabled=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)
    writer = ts.SummaryWriter()

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)

        # calculate cross entropy between the network output and the actions that the agent took.
        loss_v = objective(action_scores_v, acts_v) 
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        if(reward_m > 199):
            print("Solved.")
            break
    writer.close()
    #video_recorder.close()