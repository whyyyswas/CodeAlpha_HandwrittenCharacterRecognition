# ========== utils/parse_iam.py ==========
import os

def parse_iam_labels(ascii_path):
    samples = []
    with open(ascii_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            file_id = parts[0]
            label = ' '.join(parts[8:])
            img_path = os.path.join("data/iam_dataset", "formsA-D", file_id.replace("-", "/") + ".png")
            if os.path.exists(img_path):
                samples.append((img_path, label))
    return samples