from torchy.templates import Module
from torchy.layers import Linear
from torchy.activations import ReLU, Sigmoid, Tanh, Softmax
from typing import List

class NeuralNetwork(Module):
    def __init__(self, n_features: int, n_classes: int, hidden_sizes: List[int], hidden_activation: str, init_strategy: str="he") -> None:
        super().__init__()
        self.layers = []
        for i, size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(Linear(n_features, size, init_strategy))
            else:
                self.layers.append(Linear(hidden_sizes[i-1], size, init_strategy))
        self.layers.append(Linear(hidden_sizes[-1], n_classes, init_strategy))
        self.activation = {
            'relu': ReLU(),
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
        }.get(hidden_activation, ReLU())
        self.output_activation = Softmax()
        