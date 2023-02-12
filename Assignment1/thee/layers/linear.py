import numpy as np
from thee.utils import WsAndBs

class Linear:
    def __init__(self, in_size: int, out_size: int, type="Zero"):
        '''
            in_size: int -> Number of input units
            out_size: int -> Number of output units
            type: str -> "Zero" initialization or "Xavier" Initialization or "Random" Initialization
        '''
        self.Weights = WsAndBs(in_size, out_size, type)
        self.bias = WsAndBs(out_size, 1, type)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''
            x -> pass the numpy ndarray into the linear layer (out_size, x.shape[1])
        '''
        return self.Weights.value.T @ x + self.bias.value