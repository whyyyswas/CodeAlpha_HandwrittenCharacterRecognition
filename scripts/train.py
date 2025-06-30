### 📁 scripts/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from scripts.model import ImprovedCNNClassifier as CNNClassifier
from scripts.dataset import get_loaders
from tqdm import tqdm
import os

def train_model(epochs=20, lr=0.001, save_path='models/char_cnn.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier().to(device)
    train_loader, _ = get_loaders()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"✅ Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"🧠 Model saved at {save_path}")