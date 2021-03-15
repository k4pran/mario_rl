import random

import gym
import gym_super_mario_bros
import torch
import numpy as np
from agent_base import Agent
from agent_augments.memory import ReplayMemory
from skimage.transform import resize
from skimage.color import rgb2gray
from torchsummary import summary

from model import DQN

EVAL_SAVE_FORMAT = "./checkpoint/eval_net-episode-{}--score-{}"
TARG_SAVE_FORMAT = "./checkpoint/targ_net-episode-{}--score-{}"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print("Using device: {}".format(device))


class AgentQ(Agent, ReplayMemory):

    def __init__(self,
                 action_space,
                 height,
                 width,
                 frames=4,
                 batch_size=64,
                 epsilon=1,
                 epsilon_min=0.02,
                 epsilon_decay=1e-6,
                 gamma=0.9,
                 learning_rate=0.0002,
                 save_freq=1000,
                 training=True,
                 load_agent=False,
                 eval_agent_path="checkpoint/eval_net-episode-7000--score-10.0",
                 targ_agent_path="checkpoint/targ_net-episode-7000--score-10.0"):
        ReplayMemory.__init__(self, capacity=10000)

        self.action_space = action_space
        self.frames = frames
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.iteration_count = 0
        self.training = training

        if not training:
            epsilon = epsilon_min

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.replace_every = 1000

        self.eval_net = DQN(device, action_space, batch_size, frames, height, width, learning_rate)
        self.targ_net = DQN(device, action_space, batch_size, frames, height, width, learning_rate)

        if load_agent:
            self.load_model(eval_agent_path, targ_agent_path)

        summary(self.eval_net, (frames, height, width))
        summary(self.targ_net, (frames, height, width))

    @staticmethod
    def flatten(x):
        flattened_count = 1
        for dim in x.shape[1:]:
            flattened_count *= dim
        return x.view(-1, flattened_count)

    def update_target_network(self):
        if self.iteration_count % self.replace_every == 0:
            self.targ_net.load_state_dict(self.eval_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay \
            if self.epsilon > self.epsilon_min else self.epsilon_min

    def act(self, state) -> int:
        if random.random() > self.epsilon:
            state = torch.tensor(state, device=device)
            return self.eval_net(state.unsqueeze(0)).argmax(1).item()
        else:
            return random.randrange(self.action_space)

    def before(self):
        pass

    def after(self, episode, score):
        self.iteration_count += 1
        if episode % self.save_freq == 0:
            self.save_model(episode, score)
        self.decay_epsilon()

    def learn(self, *args, **kwargs):
        self.memorise(kwargs.get('state'),
                      kwargs.get('action'),
                      kwargs.get('reward'),
                      kwargs.get('next_state'),
                      kwargs.get('done'))

        if not self.training:
            return

        if self.get_mem_count() >= self.batch_size:
            sample = self.sample(self.batch_size)
            states = torch.stack([i[0] for i in sample])
            actions = torch.tensor([i[1] for i in sample], device=device)
            rewards = torch.tensor([i[2] for i in sample], dtype=torch.float32, device=device)
            next_states = torch.stack([i[3] for i in sample])
            dones = torch.tensor([i[4] for i in sample], dtype=torch.uint8, device=device)

            self.update_target_network()

            current_q_vals = self.eval_net(states)
            next_q_vals = self.targ_net(next_states)
            q_target = current_q_vals.clone().detach()
            q_target[torch.arange(states.size()[0]), actions] = rewards + (self.gamma * next_q_vals.max(dim=1)[0]) * (
                        1 - dones)

            self.eval_net.optimizer.zero_grad()
            loss = self.eval_net.loss(current_q_vals, q_target)
            loss.backward()
            self.eval_net.optimizer.step()

    def memorise(self, state, action, reward, next_state, done):
        state = torch.tensor(state, device=device)
        next_state = torch.tensor(next_state, device=device)

        self.store(
            state,
            action,
            reward,
            next_state,
            done)

    def get_epsilon(self):
        return self.epsilon

    def load_model(self, eval_path, targ_path):
        self.eval_net.load_state_dict(torch.load(eval_path))
        self.targ_net.load_state_dict(torch.load(targ_path))

    def save_model(self, episode, score):
        torch.save(self.eval_net.state_dict(), EVAL_SAVE_FORMAT.format(episode, score))
        torch.save(self.targ_net.state_dict(), TARG_SAVE_FORMAT.format(episode, score))


class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        t_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info

    def reset(self):
        self._obs_buffer = []
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(80, 80, 1), dtype=np.uint8)

    def observation(self, obs):
        return PreProcessFrame.process(obs)

    @staticmethod
    def process(frame):
        frame = frame[40:, :, :]
        frame = rgb2gray(frame.astype(np.float32))
        frame = resize(frame, (80, 80))

        return np.expand_dims(frame, -1).astype(np.float32)


class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(self.observation_space.shape[-1],
                                                       self.observation_space.shape[0],
                                                       self.observation_space.shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(n_steps, axis=0),
            env.observation_space.high.repeat(n_steps, axis=0),
            dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name):
    env = gym_super_mario_bros.make(env_name)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)
    return ScaleFrame(env)
