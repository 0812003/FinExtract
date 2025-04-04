# import os
# import cv2
# import pytesseract
# import camelot
# import pandas as pd
# from pdf2image import convert_from_path
# from pdfminer.high_level import extract_text

# # Configure Tesseract path (modify if necessary)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# def extract_text_from_pdf(pdf_path):
#     """ Extracts text from a PDF file using PDFMiner (for text-based PDFs). """
#     text = extract_text(pdf_path)
#     return text

# def extract_text_from_images(pdf_path):
#     """ Extracts text from scanned PDF pages using OCR (Tesseract). """
#     pages = convert_from_path(pdf_path)
#     extracted_text = []
    
#     for i, page in enumerate(pages):
#         text = pytesseract.image_to_string(page)
#         extracted_text.append(text)
    
#     return "\n".join(extracted_text)

# def extract_tables_from_pdf(pdf_path):
#     """ Extracts table data from a PDF file using Camelot. """
#     tables = camelot.read_pdf(pdf_path, flavor="stream")  # Use 'lattice' if the tables have borders
    
#     if tables.n > 0:
#         return tables[0].df  # Returning the first detected table as a DataFrame
#     else:
#         return None

# def process_pdf(pdf_path, output_csv="output.csv"):
#     """ Extracts text and tables from a PDF and saves structured data to CSV. """
#     print("Extracting text...")
#     extracted_text = extract_text_from_pdf(pdf_path) or extract_text_from_images(pdf_path)

#     print("Extracting tables...")
#     table_df = extract_tables_from_pdf(pdf_path)
    
#     # Save extracted text to a .txt file
#     with open("extracted_text.txt", "w", encoding="utf-8") as file:
#         file.write(extracted_text)

#     # Save extracted table to CSV if available
#     if table_df is not None:
#         table_df.to_csv(output_csv, index=False)
#         print(f"Table extracted and saved to {output_csv}")
#     else:
#         print("No tables detected.")

#     return output_csv

# if __name__ == "__main__":
#     pdf_path =r"f:\HACK-SPHERE\Data\Sample (5).pdf"  # Change this to your PDF file path
#     output_file = process_pdf(pdf_path, "extracted_data.csv")
#     print(f"Extraction complete. Download the file: {output_file}")


import os
import cv2
import pytesseract
import camelot
import pandas as pd
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
import fitz  # PyMuPDF for PDF handling
import pdfplumber  # For extracting tables from text-based PDFs

# Configure Tesseract path (modify if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def fix_encoding(text):
    """Fixes encoding issues."""
    try:
        return text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return text.encode('unicode_escape').decode('utf-8')

def extract_text_from_pdf(pdf_path):
    """ Extracts text from a PDF file using PDFMiner (for text-based PDFs). """
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:  # Catch potential errors during text extraction
        print(f"Error during text extraction (PDFMiner): {e}")
        return None

def extract_text_from_images(pdf_path):
    """ Extracts text from scanned PDF pages using OCR (Tesseract). """
    try:
        pages = convert_from_path(pdf_path)
        extracted_text = []
        for page in pages:
            text = pytesseract.image_to_string(page)
            extracted_text.append(text)
        return "\n".join(extracted_text)
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None


def extract_tables_from_pdf_camelot(pdf_path, output_csv="output.csv"):
    """ Extracts table data from a PDF file using Camelot. """
    try:
        tables = camelot.read_pdf(pdf_path, flavor="stream")  # Use 'lattice' if the tables have borders
        if tables.n > 0:
            table_df = tables[0].df  # Returning the first detected table as a DataFrame
            table_df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"Table extracted and saved to {output_csv}")
            return True # Table found and saved
        else:
            print("No tables detected using Camelot.")
            return False # No Table found
    except Exception as e:
        print(f"Error during table extraction (Camelot): {e}")
        return False


def extract_tables_from_pdf_pdfplumber(pdf_path, output_folder="output"):
    """Extracts tables using pdfplumber."""
    try:
        os.makedirs(output_folder, exist_ok=True)
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for j, table in enumerate(tables):
                    table = [[fix_encoding(cell) if isinstance(cell, str) else cell for cell in row] for row in table]
                    df = pd.DataFrame(table[1:], columns=table[0])
                    output_path = os.path.join(output_folder, f"page_{i + 1}_table_{j + 1}.csv")
                    df.to_csv(output_path, index=False, encoding='utf-8')  # Ensure UTF-8 encoding
                    print(f"Table {j + 1} from Page {i + 1} saved to {output_path}")
        return True  # Tables found and saved
    except Exception as e:
        print(f"Error during table extraction (pdfplumber): {e}")
        return False

def process_pdf(pdf_path, output_csv="output.csv", output_folder="output"):
    """Main function to process the PDF."""
    print("Checking for tables...")
    if extract_tables_from_pdf_camelot(pdf_path,output_csv) or extract_tables_from_pdf_pdfplumber(pdf_path, output_folder): #Prioritize camelot then pdfplumber
        return # If tables are found and extracted, exit

    print("No readily identifiable tables found. Extracting text...")
    extracted_text = extract_text_from_pdf(pdf_path) or extract_text_from_images(pdf_path)

    if extracted_text:
        with open("extracted_text.txt", "w", encoding="utf-8") as file:
            file.write(extracted_text)
        print("Text extracted and saved to extracted_text.txt")
    else:
        print("No text could be extracted.")


if __name__ == "__main__":
    pdf_path = r"F:\HACK-SPHERE\Data\Sample (16).pdf"  # Change this to your PDF file path
    process_pdf(pdf_path, "extracted_data.csv", "extracted_tables")
    print("PDF processing complete.")