import random

import torch
from agent_frame import AgentBase

from agent_augments.memory import ReplayMemory
from torchsummary import summary

from model import DQN

EVAL_SAVE_FORMAT = "./checkpoint/eval_net-episode-{}--score-{}--last_reward-{}"
TARG_SAVE_FORMAT = "./checkpoint/targ_net-episode-{}--score-{}--last_reward-{}"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print("Using device: {}".format(device))


class AgentQ(AgentBase, ReplayMemory):

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
                 eval_agent_path="saved_models/LVL1-Complete/eval_net",
                 targ_agent_path="saved_models/LVL1-Complete/targ_net"):
        ReplayMemory.__init__(self, capacity=10000)

        self.action_space = action_space
        self.frames = frames
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.iteration_count = 0
        self.training = training
        self.checkpoints_recorded = []

        if not training:
            epsilon = epsilon_min

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.replace_every = 1000
        self.episode = 0

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

    def before(self, *args, **kwargs):
        pass

    def after(self, *args, **kwargs):
        self.iteration_count += 1
        if self.episode % self.save_freq == 0 and \
                self.episode not in self.checkpoints_recorded:
            self.checkpoints_recorded.append(self.episode)
            self.save_model(self.episode, kwargs['score'], kwargs['reward'])
        self.decay_epsilon()

    def learn(self, *args, **kwargs):
        self.memorise(kwargs.get('state'),
                      kwargs.get('action'),
                      kwargs.get('reward'),
                      kwargs.get('next_state'),
                      kwargs.get('done'))

        if kwargs['done']:
            self.episode += 1

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

    def save_model(self, episode, score, last_reward):
        torch.save(self.eval_net.state_dict(), EVAL_SAVE_FORMAT.format(episode, score, last_reward))
        torch.save(self.targ_net.state_dict(), TARG_SAVE_FORMAT.format(episode, score, last_reward))
