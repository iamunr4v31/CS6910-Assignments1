import numpy as np

class ReLU:
    def __init__(self) -> None:
        pass

    def __call__(self, x) -> np.ndarray:
        baseline = np.zeros_like(x)
        return np.maximum(x, baseline)
    
    def diff(self, x) -> np.ndarray:
        return (x > 0).astype(int)