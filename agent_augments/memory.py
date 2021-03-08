import random
from collections import deque


class ReplayMemory:

    def __init__(self, capacity=100000):
        self.memory = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if batch_size >= len(self.memory):
            return [i for i in self.memory]
        return random.sample(self.memory, batch_size)

    def get_mem_count(self):
        return len(self.memory)