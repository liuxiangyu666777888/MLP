import os
import gzip
import numpy as np
from typing import Generator, Tuple


def load_fashion_mnist(path: str, kind: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    labels_path = os.path.join(path, f"{kind}-labels-idx1-ubyte.gz")
    images_path = os.path.join(path, f"{kind}-images-idx3-ubyte.gz")

    if not os.path.exists(labels_path) or not os.path.exists(images_path):
        raise FileNotFoundError(
            "Fashion-MNIST data files not found. "
            f"Expected: {labels_path} and {images_path}"
        )

    with gzip.open(labels_path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    images = images.astype(np.float32) / 255.0
    return images, labels


def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    m = X.shape[0]
    indices = np.arange(m)
    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, m, batch_size):
        batch_index = indices[i:i + batch_size]
        yield X[batch_index], y[batch_index]
