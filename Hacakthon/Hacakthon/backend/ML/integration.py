from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cloudinary
import cloudinary.uploader
import pymongo
from dotenv import load_dotenv
from pdf2image import convert_from_path
import logging
# from extract_tables import extract_tables_from_pdf  # Your ML function
from extract_tables import EnhancedTableExtractor  # Your ML function

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB Connection
MONGO_URI = os.getenv("MONGODB_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["pdf_extraction"]
collection = db["extracted_files"]

# Cloudinary Config
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

UPLOAD_FOLDER = os.path.join("backend", "ML", "uploads")
OUTPUT_FOLDER = os.path.join("backend", "ML", "output")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

poppler_path = r"C:\Users\soham\Downloads\poppler-24.08.0\Library\bin"  # Change this path if necessary

# images = convert_from_path("your_file.pdf", poppler_path=poppler_path)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract tables
    extractor = EnhancedTableExtractor()
    
    try:
        extractor.process_pdf(file_path, OUTPUT_FOLDER) 
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")  
        return jsonify({"error": "Error processing PDF"}), 500 

    extracted_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".csv")]
    if not extracted_files:
        return jsonify({"error": "No tables extracted"}), 400

    first_csv_file = extracted_files[0]
    csv_path = os.path.join(OUTPUT_FOLDER, first_csv_file)

    upload_result = cloudinary.uploader.upload(csv_path, resource_type="raw")
    cloudinary_url = upload_result.get("secure_url")

    file_data = {
        "filename": first_csv_file,
        "cloudinary_url": cloudinary_url
    }
    collection.insert_one(file_data)

    return jsonify({"message": "File uploaded successfully", "file_url": cloudinary_url})

if __name__ == "__main__":
    app.run(port=5001, debug=True)
