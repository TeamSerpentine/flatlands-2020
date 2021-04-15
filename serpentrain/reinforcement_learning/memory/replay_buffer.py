from abc import ABC, abstractmethod


class ReplayBuffer(ABC):
    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        """
        Add a sample
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        """
        Pick a batch of random samples
        """
        raise NotImplementedError
