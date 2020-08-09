import random

from agent_base import Agent


class AgentRandom(Agent):

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def before(self, *args, **kwargs):
        pass

    def after(self, *args, **kwargs):
        pass

    def act(self) -> int:
        return random.randrange(self.action_space)

    def learn(self, *args, **kwargs):
        pass
