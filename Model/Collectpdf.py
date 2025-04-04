import pdfplumber
import os

def extract_text_from_pdfs(pdf_folder):
    """Extract text from all PDFs in a folder."""
    texts = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        texts.append(text)
    return texts

# Load data
pdf_folder = r"F:\HACK-SPHERE\Data"
documents = extract_text_from_pdfs(pdf_folder)
print(f"Extracted {len(documents)} documents.")
print(documents[0])