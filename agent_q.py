import random
from collections import deque

import torch
import numpy as np
from torch.nn.modules import Module, Linear, Conv3d
import torch.nn.functional as fn
from torchvision.transforms import transforms

from agent_base import Agent
from agent_exceptions import UnknownTensorType
from agent_augments.memory import ReplayMemory

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")



def prepare_2d_shape(state: torch.tensor):
    return state.permute(2, 0, 1)


def prepare_3d_shape(stacked_2d_state: torch.Tensor):
    return stacked_2d_state.permute(1, 0, 2, 3)


class AgentQ(Agent, Module, ReplayMemory):

    def __init__(self,
                 state_space,
                 action_space,
                 channels=1,
                 image_dims=(90, 90),
                 batch_size=1,
                 nb_motion_frames=4,
                 epsilon=0.3,
                 epsilon_min=0.2,
                 epsilon_decay=0.99,
                 gamma=0.9,
                 learning_rate=0.001):
        Module.__init__(self)
        ReplayMemory.__init__(self)
        self.state_space = state_space
        self.action_space = action_space
        self.channels = channels
        self.image_dims = image_dims
        self.batch_size = batch_size
        self.nb_motion_frames = nb_motion_frames
        self.current_motion_frames = deque(maxlen=nb_motion_frames)

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.in_layer = Conv3d(channels, 32, (1, 8, 8))
        self.hidden_conv_1 = Conv3d(32, 64, (1, 4, 4))
        self.hidden_conv_2 = Conv3d(64, 128, (1, 3, 3))
        self.hidden_fc1 = Linear(128 * 4 * 78 * 78, 64)
        self.hidden_fc2 = Linear(64, 32)
        self.output = Linear(32, action_space)

        self.transform = self.get_transform()

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate)

    def before(self, *args, **kwargs):
        pass

    def after(self, *args, **kwargs):
        self.decay_epsilon()

    def decay_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state) -> int:
        if random.random() > self.epsilon and len(self.current_motion_frames) == self.current_motion_frames.maxlen:
            state = prepare_2d_shape(self.as_tensor(state))
            state = self.transform(state)
            updated_motion_frames = [motion_frame[0] for motion_frame in list(self.current_motion_frames)[1:]] + [state]
            motion_states = self.stack_frames([motion_frame for motion_frame in updated_motion_frames])
            return self(prepare_3d_shape(motion_states).unsqueeze(0)).argmax(1).item()
        else:
            return random.randrange(self.action_space)

    def learn(self, *args, **kwargs):
        self.memorise(kwargs.get('state'),
                      kwargs.get('action'),
                      kwargs.get('reward'),
                      kwargs.get('next_state'),
                      kwargs.get('done'))

        if self.get_mem_count() >= self.batch_size:
            sample = self.sample(self.batch_size)
            states = torch.stack([i[0] for i in sample])
            actions = torch.tensor([i[1] for i in sample])
            rewards = torch.tensor([i[2] for i in sample])
            next_states = torch.stack([i[3] for i in sample])
            dones = torch.tensor([i[4] for i in sample])

            current_q_vals = self(states)[torch.arange(states.size()[0]), actions]
            max_next_q_vals = self(next_states).max(dim=1)[0]
            q_target = torch.tensor(rewards + (self.gamma * max_next_q_vals)) * ~dones

            self.optimizer.zero_grad()
            loss = fn.smooth_l1_loss(current_q_vals, q_target)
            loss.backward()

            self.optimizer.step()

            return loss.item()

    def memorise(self, state, action, reward, next_state, done):
        state = prepare_2d_shape(self.as_tensor(state))
        state = self.transform(state)

        next_state = prepare_2d_shape(self.as_tensor(next_state))
        next_state = self.transform(next_state)

        self.current_motion_frames.append((state, next_state))

        if len(self.current_motion_frames) == self.current_motion_frames.maxlen:
            stacked_states = self.stack_frames([motion_frame[0] for motion_frame in self.current_motion_frames])
            stacked_next_states = self.stack_frames([motion_frame[1] for motion_frame in self.current_motion_frames])

            self.store(
                prepare_3d_shape(stacked_states),
                action,
                reward,
                prepare_3d_shape(stacked_next_states),
                done)

    def forward(self, state):
        in_out = self.in_layer(state)
        in_out = self.hidden_conv_1(in_out)
        in_out = self.hidden_conv_2(in_out)
        in_out = fn.relu(self.hidden_fc1(self.flatten(in_out)))
        in_out = fn.relu(self.hidden_fc2(in_out))
        return self.output(in_out)

    def as_tensor(self, unconverted_tensor):
        if isinstance(unconverted_tensor, np.ndarray):
            return torch.from_numpy(unconverted_tensor.copy())
        elif isinstance(unconverted_tensor, torch.Tensor):
            return unconverted_tensor
        else:
            raise UnknownTensorType("Tensor type is not supported. Supported types: ndarray | torch.tensor")

    def stack_frames(self, frames):
        return torch.stack(tuple(frames))

    def get_transform(self):
        transform_pipeline = [transforms.ToPILImage()]
        if self.channels == 1:
            transform_pipeline.append(transforms.Grayscale(num_output_channels=1))
        transform_pipeline.append(transforms.Resize(self.image_dims))
        transform_pipeline.append(transforms.ToTensor())
        return transforms.Compose(transform_pipeline)

    def prepare_single_state(self, state: np.ndarray):
        state = self.as_tensor(state)
        state = prepare_2d_shape(state)
        state = self.transform(state)
        return state.permute(2, 0, 1)

    def flatten(self, x):
        flattened_count = 1
        for dim in x.shape[1:]:
            flattened_count *= dim
        return x.view(-1, flattened_count)
