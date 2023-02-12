import numpy as np
from abc import ABC, abstractmethod

class Module(ABC):
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @property
    def parameters(self):
        return self.__dict__

class WsAndBs:
    def __init__(self, out_size, in_size=1, type="Zero") -> None:
        self.value = self.initialize_weights((out_size, in_size), type)
        self.grad = np.zeros_like(self.value)
    
    def initialize_weights(self, shape: tuple, type: str="Zero") -> np.ndarray:
        '''
            Initialize weights of shape: (shape) with type: type strategy
        '''
        if type == "Zero":
            return np.zeros(shape)
        elif type == "Xavier":
            raise NotImplementedError()
        elif type == "Random":
            return np.random.rand(shape)
        else:
            raise KeyError("Incorrect option for weight initialization strategy")
        
    def zero_grad(self):
        self.grad = np.zeros_like(self.grad)