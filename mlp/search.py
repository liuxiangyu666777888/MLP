import csv
import itertools
import os
import shutil

from train import MODEL_PATH, train


def format_value(value):
    return str(value).replace(".", "p")


def make_model_name(config):
    return (
        f"model_hd{config['hidden_dim']}_"
        f"lr{format_value(config['learning_rate'])}_"
        f"wd{format_value(config['weight_decay'])}_"
        f"act{config['activation']}.pkl"
    )


def grid_search():
    os.makedirs(MODEL_PATH, exist_ok=True)

    search_space = {
        "learning_rate": [0.1, 0.01, 0.001],
        "hidden_dim": [64, 128, 256],
        "weight_decay": [0.0, 1e-4, 1e-3],
        "activation": ["relu", "sigmoid"],
    }

    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]
    combinations = list(itertools.product(*values))
    print(f"Total combinations: {len(combinations)}")

    results = []
    best_result = None
    fieldnames = [
        "learning_rate",
        "hidden_dim",
        "weight_decay",
        "activation",
        "best_val_acc",
        "best_epoch",
        "model_path",
    ]

    for run_id, combination in enumerate(combinations, start=1):
        config = dict(zip(keys, combination))
        config.update(
            {
                "epochs": 10,
                "batch_size": 128,
                "val_ratio": 0.2,
                "seed": 42,
                "lr_decay_every": 5,
                "lr_decay_gamma": 0.5,
                "save_model": True,
                "model_name": make_model_name(config),
                "save_history": False,
            }
        )

        print()
        print(f"[{run_id}/{len(combinations)}] Running config:")
        print(config)

        result = train(config)
        row = {
            "learning_rate": config["learning_rate"],
            "hidden_dim": config["hidden_dim"],
            "weight_decay": config["weight_decay"],
            "activation": config["activation"],
            "best_val_acc": result["best_val_acc"],
            "best_epoch": result["best_epoch"],
            "model_path": result["best_model_path"],
        }
        results.append(row)

        if best_result is None or row["best_val_acc"] > best_result["best_val_acc"]:
            best_result = row.copy()
            print(f"New best result: {best_result}")

    csv_path = os.path.join(MODEL_PATH, "search.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    best_model_final_path = os.path.join(MODEL_PATH, "best_model.pkl")
    if best_result is not None and best_result["model_path"] is not None:
        if os.path.abspath(best_result["model_path"]) != os.path.abspath(best_model_final_path):
            shutil.copyfile(best_result["model_path"], best_model_final_path)

    print(f"Search results saved to: {csv_path}")
    print(f"Best model saved to: {best_model_final_path}")
    print(f"Best result: {best_result}")

    return best_result, results


if __name__ == "__main__":
    grid_search()
