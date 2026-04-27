import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils.data_loader import load_fashion_mnist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")
FIGURE_PATH = os.path.join(BASE_DIR, "figures")
DATA_PATH = os.path.join(BASE_DIR, "data")

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

def error_analysis(model_path=None,num_errors=9):
    if model_path is None:
        model_path = os.path.join(MODEL_PATH,"best_model.pkl")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    os.makedirs(FIGURE_PATH, exist_ok=True)
    
    X_test, y_test = load_fashion_mnist(DATA_PATH,kind="t10k")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    logits = model.forward(X_test)
    y_pred = np.argmax(logits, axis=1)
    
    error_indices = np.where(y_pred != y_test)[0]
    print(f"Number of errors: {len(error_indices)}")
    
    if len(error_indices) == 0:
        print("No errors found")
        return
    
    num_errors = min(num_errors, len(error_indices))
    selected_indices = error_indices[:num_errors]
    
    grid_size = int(np.ceil(np.sqrt(num_errors)))
    fig,axes = plt.subplots(grid_size,grid_size,figsize=(10,10))
    axes = np.array(axes).reshape(-1)
    
    for ax in axes:
        ax.axis("off")

    for i, idx in enumerate(selected_indices):
        image = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        pred_label = y_pred[idx]

        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(
            f"True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_label]}",
            fontsize=8,
        )
        axes[i].axis("off")

        print(
            f"Sample index: {idx}, "
            f"True: {CLASS_NAMES[true_label]}, "
            f"Pred: {CLASS_NAMES[pred_label]}"
        )

    plt.suptitle("Misclassified Test Samples", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(FIGURE_PATH, "error_analysis.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Error analysis figure saved to: {save_path}")


if __name__ == "__main__":
    error_analysis(num_errors=9)
        
