### 📁 scripts/predict.py

import torch
from torchvision import transforms
from PIL import Image
from scripts.model import ImprovedCNNClassifier as CNNClassifier
import os


def load_class_mapping(mapping_file='data/emnist-balanced-mapping.txt'):
    mapping = {}
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    with open(mapping_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                idx, unicode_val = line.strip().split()
                mapping[int(idx)] = chr(int(unicode_val))
    return mapping


def predict(image_path, model_path='models/char_cnn.pth'):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.Pad(2),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(image.to(device))
        predicted_index = torch.argmax(output, dim=1).item()

    mapping = load_class_mapping()
    predicted_char = mapping.get(predicted_index, '?')

    print(f"🌤️ Predicted class index: {predicted_index}")
    print(f"📌 Predicted character: {predicted_char}")