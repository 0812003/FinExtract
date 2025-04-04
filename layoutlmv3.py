import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# Path to Tesseract OCR executable (update if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize processor and model
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-large")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-large")

def extract_text_and_boxes(image_path):
    """Extract text and bounding box coordinates using OCR."""
    image = Image.open(image_path)

    # OCR with pytesseract to get text and bounding boxes
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    words, boxes = [], []
    for i in range(len(ocr_data["text"])):
        if ocr_data["text"][i].strip():
            words.append(ocr_data["text"][i])
            x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
            boxes.append([x, y, x + w, y + h])  # Convert to (x1, y1, x2, y2)
    
    return words, boxes, image

def perform_inference(image_path):
    """Perform LayoutLMv3 inference on the image."""
    # Extract words and boxes using OCR
    words, boxes, image = extract_text_and_boxes(image_path)

    # Preprocess image and text data for the model
    encoding = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True)

    # Perform inference (get model outputs)
    with torch.no_grad():
        outputs = model(**encoding)

    # Get predictions (for token classification, e.g., NER)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # Decode predictions to labels (if you have a specific set of labels)
    predicted_labels = predictions[0].cpu().numpy()

    return predicted_labels

# Example usage:
image_path = r"F:\HACK-SPHERE\Data\1.png"  # Replace with your image path
predictions = perform_inference(image_path)
print(predictions)
