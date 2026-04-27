from typing import Dict, List

import numpy as np

from core.layers import Linear, ReLU, Sigmoid

class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, activation: str = 'relu'):
        self.layers = []

        self.layers.append(Linear(input_dim, hidden_dim))
        activation = activation.lower()
        if activation == 'relu':
            self.layers.append(ReLU())
        elif activation == 'sigmoid':
            self.layers.append(Sigmoid())
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.layers.append(Linear(hidden_dim, num_classes))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dx: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dx = layer.backward(dx)
        return dx

    def parameters(self) -> List[Dict[str, np.ndarray]]:
        params = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                params.append(
                    {
                        'weight': layer.weight,
                        'bias': layer.bias,
                        'dw': layer.dw,
                        'db': layer.db,
                    }
                )
        return params

    def get_params(self) -> List[Dict[str, np.ndarray]]:
        return self.parameters()
