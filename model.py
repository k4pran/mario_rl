import torch
from torch.nn.modules import Module, Linear, Conv2d
import torch.nn.functional as fn


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


# class DQN(Module):
#
#     def __init__(self,
#                  device,
#                  action_space,
#                  batch_size,
#                  channels=1,
#                  height=80,
#                  width=80,
#                  learning_rate=0.001):
#         super(DQN, self).__init__()
#
#         kernel_size = 3
#         stride = 2
#         self.in_layer = Conv2d(channels, batch_size, kernel_size, stride)
#         height, width = conv_output_shape((height, width), kernel_size, stride)
#
#         kernel_size = 5
#         stride = 2
#         self.hidden_conv_1 = Conv2d(batch_size, 64, kernel_size, stride)
#         height, width = conv_output_shape((height, width), kernel_size, stride)
#
#         kernel_size = 3
#         stride = 1
#         self.hidden_conv_2 = Conv2d(64, 64, kernel_size, stride)
#
#         height, width = conv_output_shape((height, width), kernel_size, stride)
#         self.hidden_fc1 = Linear(batch_size * height * width, 512)
#         self.output = Linear(512, action_space)
#         self.loss = torch.nn.MSELoss()
#
#         self.optimizer = torch.optim.Adam(
#             self.parameters(), lr=learning_rate)
#
#         self.to(device)
#
#     def forward(self, state):
#         in_out = fn.relu(self.in_layer(state))
#         in_out = fn.relu(self.hidden_conv_1(in_out))
#         in_out = fn.relu(self.hidden_conv_2(in_out))
#         in_out = torch.flatten(in_out, 1)
#         in_out = fn.relu(self.hidden_fc1(in_out))
#         return self.output(in_out)

class DQN(Module):

    def __init__(self,
                 device,
                 action_space,
                 batch_size,
                 channels=1,
                 height=80,
                 width=80,
                 learning_rate=0.001):
        super(DQN, self).__init__()

        kernel_size = 8
        stride = 4
        self.in_layer = Conv2d(channels, batch_size, kernel_size, stride)
        height, width = conv_output_shape((height, width), kernel_size, stride)

        kernel_size = 4
        stride = 2
        self.hidden_conv_1 = Conv2d(batch_size, 64, kernel_size, stride)
        height, width = conv_output_shape((height, width), kernel_size, stride)

        kernel_size = 3
        stride = 1
        self.hidden_conv_2 = Conv2d(64, 64, kernel_size, stride)

        height, width = conv_output_shape((height, width), kernel_size, stride)
        self.hidden_fc1 = Linear(batch_size * height * width, 512)
        self.output = Linear(512, action_space)
        self.loss = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate)

        self.to(device)

    def forward(self, state):
        in_out = fn.relu(self.in_layer(state))
        in_out = fn.relu(self.hidden_conv_1(in_out))
        in_out = fn.relu(self.hidden_conv_2(in_out))
        in_out = torch.flatten(in_out, 1)
        in_out = fn.relu(self.hidden_fc1(in_out))
        return self.output(in_out)