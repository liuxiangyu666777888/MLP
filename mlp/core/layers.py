import numpy as np


class Layer:
    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dx: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros((1, out_features))
        self.dw = np.zeros_like(self.weight)
        self.db = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return np.dot(x, self.weight) + self.bias

    def backward(self, dx: np.ndarray) -> np.ndarray:
        x = self.cache
        self.dw[...] = np.dot(x.T, dx)
        self.db[...] = np.sum(dx, axis=0, keepdims=True)
        return np.dot(dx, self.weight.T)


class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dx: np.ndarray) -> np.ndarray:
        x = self.cache
        return np.where(x > 0, dx, 0)


class Sigmoid(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50, 50)
        out = 1.0 / (1.0 + np.exp(-x))
        self.cache = out
        return out

    def backward(self, dx: np.ndarray) -> np.ndarray:
        out = self.cache
        return dx * out * (1.0 - out)
