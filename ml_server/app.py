import os
import json
import logging
import sqlite3
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

app = Flask(__name__)
CORS(app, supports_credentials=True)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_weights_path = "model.weights.h5"
model_config_path = "config.json"

with open(model_config_path, "r") as f:
    model_config = tf.keras.models.model_from_json(f.read())
model_config.load_weights(model_weights_path)
model = model_config

def preprocess_image(img):
    img = np.array(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] != 3:
        raise ValueError("Invalid image format.")
    channels = cv2.split(img)
    processed_channels = []
    for channel in channels:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        channel = cv2.filter2D(channel, -1, kernel)
        channel = cv2.GaussianBlur(channel, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(3, 3))
        channel = clahe.apply(channel)
        processed_channels.append(channel)
    img = cv2.merge(processed_channels)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def create_reports_db():
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            report_type TEXT NOT NULL,
            input_image BLOB NOT NULL,
            report_data BLOB NOT NULL,
            prediction TEXT NOT NULL,
            probabilities TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    conn.close()

create_reports_db()

disease_classes = [
    "Cashew anthracnose", "Cashew gumosis", "Cashew healthy", "Cashew leaf miner",
    "Cashew red rust", "Cassava bacterial blight", "Cassava brown spot", "Cassava green mite",
    "Cassava healthy", "Cassava mosaic", "Corn___Common_Rust", "Corn___Gray_Leaf_Spot",
    "Corn___Healthy", "Corn___Northern_Leaf_Blight", "Maize fall armyworm", "Maize grasshopper",
    "Maize healthy", "Maize leaf beetle", "Maize leaf blight", "Maize leaf spot", "Maize streak virus",
    "Potato___Early_Blight", "Potato___Healthy", "Potato___Late_Blight", "Rice___Brown_Spot",
    "Rice___Healthy", "Rice___Leaf_Blast", "Rice___Neck_Blast", "Sugarcane_Bacterial Blight",
    "Sugarcane_Healthy", "Sugarcane_Red Rot", "Tomato healthy", "Tomato leaf blight",
    "Tomato leaf curl", "Tomato septoria leaf spot", "Tomato verticillium wilt", "Wheat___Brown_Rust",
    "Wheat___Healthy", "Wheat___Yellow_Rust"
]

def generate_and_store_report(user_id, username, disease_name, probabilities, img):
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    page_width, page_height = letter

    c.setFont("Helvetica-Bold", 110)
    c.setFillColorRGB(0.9, 0.9, 0.9)
    c.saveState()
    c.rotate(45)
    c.drawString(page_width / 4, page_height / 10, "Crop Connect")
    c.restoreState()

    title_height = 60
    title_y = page_height - title_height
    c.setFillColorRGB(2 / 255, 101 / 255, 2 / 255)
    c.rect(0, title_y, page_width, title_height, fill=True, stroke=False)

    c.setFillColorRGB(245 / 255, 222 / 255, 179 / 255)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(page_width / 2, title_y + (title_height - 24) / 2, "Crop Disease Prediction Report")

    c.setFont("Helvetica", 12)
    c.setFillColorRGB(0, 0, 0)
    y_offset = page_height - 100
    c.drawString(50, y_offset, f"Username: {username}")
    c.drawString(50, y_offset - 20, f"Prediction: {disease_name}")

    top_diseases = sorted(zip(disease_classes, probabilities), key=lambda x: x[1], reverse=True)[:10]
    table_data = [["Disease", "Probability"]] + [[d, f"{p:.2%}"] for d, p in top_diseases]

    table = Table(table_data, colWidths=[150, 100])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    table.wrapOn(c, 50, 50)
    table.drawOn(c, 50, page_height - 350)

    input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "input_temp.jpg")
    cv2.imwrite(input_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    c.drawImage(input_image_path, page_width - 250, page_height - 350, width=200, height=200)

    c.setFont("Helvetica", 10)
    c.drawString(50, 30, f"Date: {datetime.now().strftime('%B %d, %Y')}")
    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    pdf_data = pdf_buffer.read()

    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO reports (user_id, report_type, input_image, report_data, prediction, probabilities)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        "Crop Disease",
        sqlite3.Binary(img.tobytes()),
        sqlite3.Binary(pdf_data),
        disease_name,
        str(probabilities)
    ))
    conn.commit()
    conn.close()
    return True, pdf_data, cursor.lastrowid

@app.route('/crop-disease-predict', methods=['POST'])
def crop_disease_predict():
    if not request.files:
        return jsonify({"error": "No image received"}), 400

    file = request.files.get('image')
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or missing image file"}), 400

    user_details = request.form.get("user_details")
    if not user_details:
        return jsonify({"error": "User details missing"}), 400

    user_details = json.loads(user_details)
    if not user_details.get("_id"):
        return jsonify({"error": "User ID is missing"}), 400

    img = Image.open(file.stream)
    processed_image = preprocess_image(img)
    prediction = model.predict(processed_image)
    predicted_class_index = int(np.argmax(prediction, axis=1)[0])
    probabilities = prediction[0].tolist()
    disease_name = disease_classes[predicted_class_index]

    success, pdf_data, report_id = generate_and_store_report(
        user_id=user_details["_id"],
        username=user_details["name"],
        disease_name=disease_name,
        probabilities=probabilities,
        img=np.array(img)
    )

    if success:
        return jsonify({
            "success": True,
            "prediction": disease_name,
            "report_id": report_id
        })
    else:
        return jsonify({"error": pdf_data}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
