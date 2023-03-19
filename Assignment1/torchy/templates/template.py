from abc import ABC, abstractmethod
from typing import TypedDict, Dict, List, Tuple

import numpy as np

class AutoDiff(ABC):

    def __init__(self) -> None:
        self.cache: dict = {}

    @abstractmethod
    def forward(self, *args) -> None:
        pass

    @abstractmethod
    def backward(self, *args) -> None:
        pass

    @abstractmethod
    def __call__(self, *args) -> None:
        pass

class Weights(TypedDict):
    w: np.ndarray
    b: np.ndarray
    dw: np.ndarray
    db: np.ndarray

class Layer(AutoDiff):
    weights: Weights = {}

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args) -> None:
        pass

    @abstractmethod
    def backward(self, *args) -> None:
        pass

    @abstractmethod
    def init_weights(self, *args) -> None:
        pass

class Loss(AutoDiff):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.value: np.float32 = 0.0
        self.reg_value: float = 0.0

    @abstractmethod
    def forward(self, *args) -> None:
        pass

    @abstractmethod
    def diff(self, *args) -> None:
        pass

    def regularize_loss(self) -> None:
        if self.regularize == "l1":
            for layer in self.model.parameters:
                self.reg_value += np.sum(np.abs(layer["w"]))
        elif self.regularize == "l2":
            for layer in self.model.parameters:
                self.reg_value += np.sum(np.square(layer["w"]))
        self.value += self.alpha * self.reg_value
        self.reg_value = 0.0

    def backward(self) -> None:
        y_hat = self.diff()
        y_hat = self.model.output_activation.backward(y_hat)
        L = len(self.model.layers)
        for i, layer in enumerate(self.model.layers[::-1]):
            layer.backward(y_hat)
            if self.regularize == "l2":
                layer.weights['dw']+= self.alpha * layer.weights['w']
            elif self.regularize == "l1":
                layer.weights['dw']+= self.alpha * np.sign(layer.weights['w'] + 1e-8)
            if L - i - 1 >= 1:
                l__h_prev = np.dot(y_hat, layer.weights['w'])
                y_hat = self.model.activation.backward(l__h_prev)

    def __call__(self, *args) -> None:
        self.forward(*args)
        return self
    
class Activation(AutoDiff):

    def __init__(self) -> None:
        super().__init__()
        self.cache['x'] = []

    @abstractmethod
    def forward(self, *args) -> None:
        pass

    @abstractmethod
    def backward(self, *args) -> None:
        pass

class Optimizer(ABC):
    def __init__(self) -> None:
        self.params = {}

    @abstractmethod
    def step(self, *args) -> None:
        pass

    @abstractmethod
    def zero_grad(self, *args) -> None:
        pass

class Module(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        y = self.layers[-1](x)
        o = self.output_activation(y)
        return o
    
    def __call__(self, x) -> np.ndarray:
        return self.forward(x)
    
    @property
    def parameters(self) -> List[Weights]:
        return [layer.weights for layer in self.layers]