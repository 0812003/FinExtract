import cv2
import pytesseract
import re
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pypdf import PdfReader
from pdf2image import convert_from_path
from io import BytesIO
from PIL import Image
import os

# Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update path if needed

nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess images for better OCR
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return processed

# Extract text from an image
def extract_text_from_image(image_path, lang="eng+mar"):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    processed_image = preprocess_image(image)
    text = pytesseract.image_to_string(processed_image, lang=lang, config='--psm 6')  # Better text extraction
    return text.strip()

# Extract text from scanned PDFs using OCR
def extract_text_from_scanned_pdf(pdf_path, lang="eng+mar"):
    text = ""
    images = convert_from_path(pdf_path, dpi=300)  
    for image in images:
        img_bytes = BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        img_pil = Image.open(img_bytes)
        image_cv = np.array(img_pil)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        processed_image = preprocess_image(image_cv)
        page_text = pytesseract.image_to_string(processed_image, lang=lang, config='--psm 6')
        text += page_text + "\n"
    return text.strip()

# Extract text from text-based PDFs
def extract_text_from_text_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text.strip()

# Extract text from a file (PDF or image)
def extract_text(file_path):
    text = ""
    if file_path.lower().endswith(".pdf"):
        text = extract_text_from_text_pdf(file_path)
        if not text:
            text = extract_text_from_scanned_pdf(file_path)
    elif file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        text = extract_text_from_image(file_path)
    return text.strip()

# Extract headers from text with high precision
def extract_headers(text):
    lines = text.split("\n")
    headers = []

    for line in lines:
        line = line.strip()
        if re.match(r'^[A-Z][A-Z\s\d\W]+$', line) and len(line.split()) <= 10:
            headers.append(line)

    return list(set(headers))  # Remove duplicates

# Detect column layouts based on tabular structures
def detect_column_layouts(text):
    lines = text.split("\n")
    columns = []
    for line in lines:
        if "\t" in line or re.search(r'(\s{3,})', line):  # Checks for tab or large spaces
            columns.append(line)

    return columns

# Preprocess extracted text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Extract features for clustering
def extract_features(text):
    num_words = len(text)
    unique_words = set(text)
    num_unique_words = len(unique_words)
    total_length = sum(len(word) for word in text)
    avg_word_length = total_length / num_words if num_words > 0 else 0
    type_token_ratio = num_unique_words / num_words if num_words > 0 else 0
    return [num_words, num_unique_words, avg_word_length, type_token_ratio]

# Cluster documents based on text content
def cluster_documents(file_paths):
    features = []
    for file_path in file_paths:
        text = extract_text(file_path)
        if not text:
            continue

        preprocessed_text = preprocess_text(text)
        extracted_features = extract_features(preprocessed_text)
        extracted_features.append(text)
        features.append(extracted_features)

    if not features:
        return []

    df = pd.DataFrame(features, columns=['num_words', 'num_unique_words', 'avg_word_length', 'type_token_ratio', 'text'])
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['text'])

    silhouette_scores = []
    for k in range(2, min(10, len(file_paths))):  
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(tfidf_matrix)
            silhouette_avg = silhouette_score(tfidf_matrix, labels)
            silhouette_scores.append(silhouette_avg)
        except ValueError:
            silhouette_scores.append(-1)

    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2 if silhouette_scores else 2
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(tfidf_matrix)

    return labels

# Classify document structure with headers & columns
def classify_document_structure(file_path):
    text = extract_text(file_path)
    headers = extract_headers(text)
    columns = detect_column_layouts(text)

    classification = {
        'headers': headers,
        'column_layouts': columns,
        'file_name': file_path
    }
    return classification

# Example usage
if __name__ == "__main__":
    file_paths = ['cashflow.jpg', 'balance sheet.jpg', 'income.jpg','comprehensive income.jpg', 'Shareholder Equity.jpg']  # Replace with actual files
    cluster_labels = cluster_documents(file_paths)

    for file_path, label in zip(file_paths, cluster_labels):
        print(f"File: {file_path}, Cluster: {label}")
        print(classify_document_structure(file_path))
