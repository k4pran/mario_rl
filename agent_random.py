import random

from scaffo import AgentBase


class AgentRandom(AgentBase):

    def __init__(self, state_space, action_space, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_space = state_space
        self.action_space = action_space

    def before(self, *args, **kwargs):
        pass

    def after(self, *args, **kwargs):
        pass

    def act(self, _) -> int:
        return random.randrange(self.action_space)

    def learn(self, *args, **kwargs):
        pass
