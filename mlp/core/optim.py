import numpy as np

class SGD:
    def __init__(self, model_params, lr: float = 0.01, weight_decay: float = 0.0, learning_rate=None):
        self.model_params = model_params
        if learning_rate is not None:
            lr = learning_rate
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for param in self.model_params:
            grad_w = param['dw']
            if self.weight_decay > 0:
                grad_w = grad_w + self.weight_decay * param['weight']
            param['weight'] -= self.lr * grad_w
            param['bias'] -= self.lr * param['db']
