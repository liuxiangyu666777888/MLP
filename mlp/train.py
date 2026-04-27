import os
import pickle

import numpy as np

from core.loss import CrossEntropyLoss
from core.model import MLP
from core.optim import SGD
from utils.data_loader import create_batches, load_fashion_mnist
from utils.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models")


def train(config=None):
    default_config = {
        "epochs": 10,
        "batch_size": 128,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "hidden_dim": 128,
        "activation": "relu",
        "val_ratio": 0.2,
        "seed": 42,
        "lr_decay_every": 5,
        "lr_decay_gamma": 0.5,
        "save_model": True,
        "model_name": "best_model.pkl",
        "save_history": True,
        "history_name": "history.pkl",
    }
    final_config = default_config.copy()
    if config:
        final_config.update(config)

    epochs = final_config["epochs"]
    batch_size = final_config["batch_size"]
    learning_rate = final_config["learning_rate"]
    weight_decay = final_config["weight_decay"]
    hidden_dim = final_config["hidden_dim"]
    activation = final_config["activation"]
    val_ratio = final_config["val_ratio"]
    seed = final_config["seed"]
    lr_decay_every = final_config["lr_decay_every"]
    lr_decay_gamma = final_config["lr_decay_gamma"]
    save_model = final_config["save_model"]
    model_name = final_config["model_name"]
    save_history = final_config["save_history"]
    history_name = final_config["history_name"]

    os.makedirs(MODEL_PATH, exist_ok=True)
    np.random.seed(seed)

    X_full, y_full = load_fashion_mnist(DATA_PATH, kind="train")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X_full))
    split_index = int((1 - val_ratio) * len(X_full))

    train_idx = indices[:split_index]
    val_idx = indices[split_index:]

    X_train = X_full[train_idx]
    y_train = y_full[train_idx]
    X_val = X_full[val_idx]
    y_val = y_full[val_idx]

    model = MLP(
        input_dim=784,
        hidden_dim=hidden_dim,
        num_classes=10,
        activation=activation,
    )

    criterion = CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = float("-inf")
    best_epoch = 0
    best_model_path = None
    history_path = None

    for epoch in range(epochs):
        if epoch > 0 and lr_decay_every > 0 and epoch % lr_decay_every == 0:
            optimizer.lr *= lr_decay_gamma
            print(f"Learning rate decay to {optimizer.lr:.4f}")

        train_loss_sum = 0.0
        num_batches = 0

        for X_batch, y_batch in create_batches(X_train, y_train, batch_size, shuffle=True):
            logits = model.forward(X_batch)
            loss = criterion.forward(logits, y_batch)
            grad_loss = criterion.backward()
            model.backward(grad_loss)
            optimizer.step()

            train_loss_sum += loss
            num_batches += 1

        avg_train_loss = train_loss_sum / num_batches

        val_logits = model.forward(X_val)
        val_loss = criterion.forward(val_logits, y_val)
        val_acc = accuracy_score(y_val, val_logits)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            if save_model:
                best_model_path = os.path.join(MODEL_PATH, model_name)
                with open(best_model_path, "wb") as f:
                    pickle.dump(model, f)

        if save_history:
            history_path = os.path.join(MODEL_PATH, history_name)
            payload = {
                "config": final_config,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "history": history,
            }
            with open(history_path, "wb") as f:
                pickle.dump(payload, f)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

    return {
        "config": final_config,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "history": history,
        "best_model_path": best_model_path,
        "history_path": history_path,
    }


if __name__ == "__main__":
    result = train()
    print(result)
