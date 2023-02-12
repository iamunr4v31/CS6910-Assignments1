import numpy as np

class Softmax:
    def __init__(self) -> None:
        pass
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def diff(self, x):
        z = self(x)
        return - np.outer(z, z) + np.diag(z.flatten())