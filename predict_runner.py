### 📁 predict_runner.py

import sys
from scripts.predict import predict

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_runner.py path/to/image.png")
        sys.exit(1)

    image_path = sys.argv[1]
    predict(image_path)