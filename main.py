### 📁 main.py

from scripts.train import train_model
from scripts.evaluate import evaluate_model

if __name__ == '__main__':
    print("🚀 Training Started...")
    train_model(epochs=30)

    print("\n📈 Evaluating Model...")
    evaluate_model()