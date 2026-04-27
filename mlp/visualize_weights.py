import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")
FIGURE_PATH = os.path.join(BASE_DIR, "figures")


def normalize_img(img):
    img_min = np.min(img)
    img_max = np.max(img)

    if img_max - img_min < 1e-10:
        return np.zeros_like(img)

    return (img - img_min) / (img_max - img_min)


def visualize_first_layer_weights(model_path=None, num_weights=25, grid_size=None):
    if model_path is None:
        model_path = os.path.join(MODEL_PATH, "best_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    os.makedirs(FIGURE_PATH, exist_ok=True)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    first_layer = model.layers[0]
    weights = first_layer.weight

    if weights.shape[0] != 784:
        raise ValueError(f"First layer weights shape have 784 rows, but got {weights.shape[0]} rows")

    hidden_weights = weights.shape[1]
    num_weights = min(num_weights, hidden_weights)
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(num_weights)))
    elif grid_size * grid_size < num_weights:
        raise ValueError("grid_size is too small for the requested number of weights")

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = np.array(axes).ravel()

    for i in range(grid_size * grid_size):
        axes[i].axis("off")

        if i >= num_weights:
            continue

        weight_img = weights[:, i].reshape(28, 28)
        weight_img = normalize_img(weight_img)
        axes[i].imshow(weight_img, cmap="gray")
        axes[i].set_title(f"Weight {i}")

    plt.suptitle("First Layer Weights")
    plt.tight_layout()
    save_path = os.path.join(FIGURE_PATH, "first_layer_weights.png")
    plt.savefig(save_path)
    plt.close()

    print(f"First-layer weight figure saved to: {save_path}")


if __name__ == "__main__":
    visualize_first_layer_weights()
