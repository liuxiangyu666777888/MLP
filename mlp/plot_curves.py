import os
import pickle
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")
FIGURE_PATH = os.path.join(BASE_DIR, "figures")

def plot_training_curves(history_path=None):
    if history_path is None:
        history_path = os.path.join(MODEL_PATH,"history.pkl")
        
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    os.makedirs(FIGURE_PATH, exist_ok=True)
    
    with open(history_path, "rb") as f:
        payload = pickle.load(f)
        
    history = payload["history"]
    
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_acc = history["val_acc"]
    
    epochs = range(1,len(train_loss)+1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs,train_loss,marker="o",label="Train Loss")
    plt.plot(epochs,val_loss,marker="o",label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(FIGURE_PATH,"loss.png")
    plt.savefig(loss_path)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs,val_acc,marker="o",color="green",label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path = os.path.join(FIGURE_PATH,"acc.png")
    plt.savefig(acc_path)
    plt.close()
    
    
    print(f"Loss curve saved to: {loss_path}")
    print(f"Validation accuracy curve saved to: {acc_path}")
    
if __name__ == "__main__":
    plot_training_curves()
