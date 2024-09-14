#from offlinerllib.buffer.base import Buffer
from abc import ABC, abstractmethod

class Buffer(ABC):
    @abstractmethod
    def random_batch(self, batch_size):
        raise NotImplementedError