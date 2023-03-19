from torchy.templates import Loss, Module
import numpy as np

class MSE(Loss):
    def __init__(self, model: Module, regularize: str="none", alpha: float=5e-3) -> None:
        super().__init__(model)
        self.regularize = regularize
        self.alpha = alpha

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.float32:
        self.cache['y_pred'] = y_pred
        self.cache['y_true'] = y_true
        # self.value = np.squeeze(np.mean(np.square(y_pred - y_true)))
        self.value = np.squeeze(np.sum(np.square(y_pred - y_true)) / y_pred.shape[0])
        self.regularize_loss()
        return self
    
    def diff(self) -> np.ndarray:
        return 2 * (self.cache['y_pred'] - self.cache['y_true'])

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.float32:
        return self.forward(y_pred, y_true)

class CrossEntropy(Loss):
    def __init__(self, model: Module, regularize: str="none", alpha: float=5e-3) -> None:
        super().__init__(model)
        self.regularize = regularize
        self.alpha = alpha

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.float32:
        self.cache['y_pred'] = y_pred
        self.cache['y_true'] = y_true
        self.value = np.squeeze(-np.sum(y_true * np.log(y_pred + 1e-12)))
        self.regularize_loss()
        return self

    def diff(self) -> np.ndarray:
        return self.cache['y_pred'] - self.cache['y_true']

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.float32:
        return self.forward(y_pred, y_true)