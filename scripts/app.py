import os
import sys
from flask import Flask, request, render_template, send_from_directory
from flask_compress import Compress

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ocr_infer import predict_text

app = Flask(__name__, static_folder="../static", template_folder="../templates")
Compress(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(save_path, 'wb') as f:
                for chunk in file.stream:
                    f.write(chunk)

            result_text = predict_text(save_path)
            return render_template('result.html', text=result_text, image=file.filename)
    return render_template('index.html')

@app.route('/evaluate')
def evaluate():
    from scripts import evaluate as ev  # Triggers evaluation when route is called
    return render_template("evaluate.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)