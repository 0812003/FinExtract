# import fitz  # PyMuPDF for PDF handling
# import pdfplumber  # For extracting tables from text-based PDFs
# import pytesseract  # OCR for scanned PDFs
# from pdf2image import convert_from_path  # Convert PDF pages to images for OCR
# import pandas as pd
# import os
# import cv2
# import numpy as np

# # Path to Tesseract OCR executable (update if necessary)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# def fix_encoding(text):
#     """
#     Fix encoding issues like 'JSPMâ€™s' to 'JSPM's'.
#     """
#     try:
#         return text.encode('utf-8').decode('utf-8')
#     except UnicodeDecodeError:
#         return text.encode('unicode_escape').decode('utf-8')

# def extract_tables_from_pdf(pdf_path, output_folder="output"):
#     """
#     Extract structured tables first, then extract unstructured table-like text.
#     """
#     os.makedirs(output_folder, exist_ok=True)
#     pdf_document = fitz.open(pdf_path)
#     is_text_based = any(page.get_text() for page in pdf_document)
    
#     if is_text_based:
#         print("Text-based PDF detected. Extracting structured tables...")
#         with pdfplumber.open(pdf_path) as pdf:
#             for i, page in enumerate(pdf.pages):
#                 tables = page.extract_tables()
                
#                 # First iteration: Extract structured tables
#                 for j, table in enumerate(tables):
#                     table = [[fix_encoding(cell) if isinstance(cell, str) else cell for cell in row] for row in table]
#                     df = pd.DataFrame(table[1:], columns=table[0])
#                     output_path = os.path.join(output_folder, f"page_{i+1}_table_{j+1}.csv")
#                     df.to_csv(output_path, index=False, encoding='utf-8')
#                     print(f"Table {j+1} from Page {i+1} saved to {output_path}")
                
#                 # Second iteration: Extract non-tabular structured data
#                 if not tables:
#                     text = fix_encoding(page.extract_text())
#                     if text:
#                         text_output = os.path.join(output_folder, f"page_{i+1}_structured_text.csv")
#                         with open(text_output, "w", encoding="utf-8") as f:
#                             f.write(text)
#                         print(f"Structured text from Page {i+1} saved to {text_output}")
#     else:
#         print("Scanned PDF detected. Performing OCR for structured text and tables...")
#         images = convert_from_path(pdf_path)
        
#         for i, image in enumerate(images):
#             custom_oem_psm_config = r'--oem 3 --psm 6'
#             text = pytesseract.image_to_string(image, config=custom_oem_psm_config)
#             text = fix_encoding(text)
            
#             # Save OCR text
#             text_output = os.path.join(output_folder, f"page_{i+1}_ocr_text.csv")
#             with open(text_output, "w", encoding="utf-8") as f:
#                 f.write(text)
#             print(f"OCR extracted text from Page {i+1} saved to {text_output}")
            
#             # Extract structured table-like data using contours
#             gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
#             _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
#             contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             table_data = []
#             for cnt in contours:
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 roi = gray[y:y+h, x:x+w]
#                 cell_text = pytesseract.image_to_string(roi, config='--psm 6').strip()
#                 if cell_text:
#                     table_data.append((x, y, cell_text))
            
#             # Sort extracted data by position and save
#             table_data.sort(key=lambda x: (x[1], x[0]))
#             df = pd.DataFrame(table_data, columns=["X", "Y", "Text"])
#             ocr_table_output = os.path.join(output_folder, f"page_{i+1}_ocr_table.csv")
#             df.to_csv(ocr_table_output, index=False, encoding='utf-8')
#             print(f"OCR table data from Page {i+1} saved to {ocr_table_output}")
    
#     pdf_document.close()
#     print("Table extraction complete!")

# # Example usage
# pdf_path = r"F:\HACK-SPHERE\Data\Sample (1).pdf"  # Replace with your PDF path
# extract_tables_from_pdf(pdf_path)

import fitz  # PyMuPDF for PDF handling
import pdfplumber  # Extract tables from text-based PDFs
import pytesseract  # OCR for scanned PDFs
from pdf2image import convert_from_path  # Convert PDF pages to images for OCR
import pandas as pd
import os
import cv2
import numpy as np

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def fix_encoding(text):
    """Fix encoding issues in extracted text."""
    replacements = {
        "â€™": "'", "â€œ": '"', "â€": '"',
        "â€“": "-", "â€˜": "'", "â€": "-"
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text.strip()

def split_headers(header_row):
    """Attempt to clean and split merged header rows intelligently."""
    if isinstance(header_row, list) and len(header_row) == 1:
        possible_delimiters = [' ', '|', ',', ';']
        for delimiter in possible_delimiters:
            if delimiter in header_row[0]:
                return header_row[0].split(delimiter)
    return header_row

def extract_tables_from_pdf(pdf_path, output_folder="output"):
    """Extract tables and text from a PDF."""
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        pdf_document = fitz.open(pdf_path)
        is_text_based = any(page.get_text() for page in pdf_document)
        
        if is_text_based:
            print("Text-based PDF detected. Extracting structured tables...")
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    # Extract structured tables
                    for j, table in enumerate(tables):
                        if table:
                            header = split_headers(table[0])  # Fix merged headers
                            df = pd.DataFrame(table[1:], columns=header)
                            output_path = os.path.join(output_folder, f"page_{i+1}_table_{j+1}.csv")
                            df.to_csv(output_path, index=False, encoding='utf-8')
                            print(f"Table {j+1} from Page {i+1} saved to {output_path}")
                    
                    # Extract text if no tables are found
                    if not tables:
                        text = fix_encoding(page.extract_text())
                        if text:
                            text_output = os.path.join(output_folder, f"page_{i+1}_text.csv")
                            with open(text_output, "w", encoding="utf-8") as f:
                                f.write(text)
                            print(f"Extracted text from Page {i+1} saved to {text_output}")
        else:
            print("Scanned PDF detected. Performing OCR...")
            images = convert_from_path(pdf_path)
            
            for i, image in enumerate(images):
                custom_oem_psm_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(image, config=custom_oem_psm_config)
                text = fix_encoding(text)
                
                # Save OCR text
                text_output = os.path.join(output_folder, f"page_{i+1}_ocr_text.csv")
                with open(text_output, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"OCR extracted text from Page {i+1} saved to {text_output}")
                
                # Extract tables using contours
                gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                table_data = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi = gray[y:y+h, x:x+w]
                    cell_text = pytesseract.image_to_string(roi, config='--psm 6').strip()
                    if cell_text:
                        table_data.append((x, y, cell_text))
                
                # Sort extracted data by position and save
                table_data.sort(key=lambda x: (x[1], x[0]))
                df = pd.DataFrame(table_data, columns=["X", "Y", "Text"])
                ocr_table_output = os.path.join(output_folder, f"page_{i+1}_ocr_table.csv")
                df.to_csv(ocr_table_output, index=False, encoding='utf-8')
                print(f"OCR table data from Page {i+1} saved to {ocr_table_output}")
        
        pdf_document.close()
        print("Table extraction complete!")
    
    except Exception as e:
        print(f"Error: {e}")

# Example usage
pdf_path = r"F:\HACK-SPHERE\Data\Sample (1).pdf"
extract_tables_from_pdf(pdf_path)
