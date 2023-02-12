import numpy as np

class Sigmoid:
    def __init__(self, scaler: int=1) -> None:
        self.scaler = scaler
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        sig = 1 / (1 + np.exp(-x))
        return self.scaler * sig
    
    def diff(self, x) -> np.ndarray:
        return self(x) * (1 - self(x))