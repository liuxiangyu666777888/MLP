import os
import pickle

from utils.data_loader import load_fashion_mnist
from utils.metrics import accuracy_score, get_confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models")


def test(model_path=None):
    if model_path is None:
        model_path = os.path.join(MODEL_PATH, "best_model.pkl")

    X_test, y_test = load_fashion_mnist(DATA_PATH, kind="t10k")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logits = model.forward(X_test)
    acc = accuracy_score(y_test, logits)
    cm = get_confusion_matrix(y_test, logits)

    print(f"Test accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    test()
