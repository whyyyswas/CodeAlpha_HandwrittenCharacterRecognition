# ========== utils/ocr_infer.py ==========
import easyocr
reader = easyocr.Reader(['en'])

def predict_text(image_path):
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)