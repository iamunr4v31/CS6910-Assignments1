from thee import BaseModule
from thee.layers import Linear
from thee.activations import ReLU, Softmax

class Model(BaseModule):
    def __init__(self) -> None:
        self.layers = [
            Linear(28*28, 1024), 
            Linear(1024, 512), 
            Linear(512, 256), 
            Linear(256, 10)
            ]
        self.relu = ReLU()
        self.softmax = Softmax()
        self.call_stack = []
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            self.call_stack.append(self.relu)
        x = self.softmax(self.layers[-1](x))
        self.call_stack.append(self.softmax)

        return x