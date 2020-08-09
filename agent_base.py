from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def before(self, *args, **kwargs):
        pass

    @abstractmethod
    def after(self, *args, **kwargs):
        pass

    @abstractmethod
    def act(self) -> int:
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        pass
