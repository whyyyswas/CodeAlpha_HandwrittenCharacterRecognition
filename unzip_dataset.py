# ========== unzip_dataset.py ==========
# One-time unzip script
import zipfile

zip_path = 'data/iam_dataset.zip'
extract_path = 'data/iam_dataset'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("✅ Dataset extracted.")
