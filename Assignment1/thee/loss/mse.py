import numpy as np

class MSELoss:
    def __init__():
        pass

    def __call__(self, y_pred, y_hat):
        return np.mean((y_pred - y_hat) ** 2)
    
    def diff(self, y_pred, y_hat):
        return  - np.mean((y_pred - y_hat))