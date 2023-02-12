import numpy as np

class Tanh:
    def __init__(self) -> None:
        pass
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def diff(self, x) -> np.ndarray:
        return 1 - self(x) ** 2