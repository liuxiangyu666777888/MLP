import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.cache = None

    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> float:
        return self.forward(logits, targets)

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        m = logits.shape[0]
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.cache = (softmax, targets)

        loss = -np.mean(np.log(softmax[np.arange(m), targets] + 1e-15))
        return float(loss)

    def backward(self) -> np.ndarray:
        softmax, targets = self.cache
        m = softmax.shape[0]
        grad = softmax.copy()
        grad[np.arange(m), targets] -= 1
        grad /= m
        return grad
