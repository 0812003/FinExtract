from flask import Flask, request, render_template, send_file
import pytesseract  # OCR for image text extraction
import pandas as pd
import os
import cv2
import numpy as np
import re
from pdf2image import convert_from_path

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Path to Tesseract OCR executable (update if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def fix_encoding(text):
    """
    Fix encoding issues like 'JSPMâ€™s' to 'JSPM's'.
    """
    try:
        return text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return text.encode('unicode_escape').decode('utf-8')

def extract_structured_data_from_image(image):
    """
    Extract structured tables and unstructured text from an image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    table_data = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = gray[y:y+h, x:x+w]
        cell_text = pytesseract.image_to_string(roi, config='--psm 6').strip()
        if cell_text:
            table_data.append((x, y, cell_text))
    
    # Sort extracted data by position and split into separate columns using spaces or tabs
    table_data.sort(key=lambda x: (x[1], x[0]))
    extracted_texts = [row[2] for row in table_data]
    structured_rows = [re.split(r'\s{2,}|\t', row) for row in extracted_texts]
    df = pd.DataFrame(structured_rows)
    return df

def extract_text_from_pdf(pdf_path):
    """
    Convert PDF to images and extract structured data from each page.
    """
    images = convert_from_path(pdf_path, dpi=300)
    all_data = []
    for i, image in enumerate(images):
        image_np = np.array(image)
        df = extract_structured_data_from_image(image_np)
        all_data.append(df)
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        csv_output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.basename(pdf_path)}_structured_data.csv")
        final_df.to_csv(csv_output_path, index=False, encoding='utf-8')
        return csv_output_path
    return None

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    csv_path = extract_text_from_pdf(file_path)
    if csv_path:
        return send_file(csv_path, as_attachment=True)
    return "Failed to extract structured data from PDF"

if __name__ == '__main__':
    app.run(debug=True)
