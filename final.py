import os
import cv2
import pytesseract
import camelot
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from io import StringIO
import logging
import warnings
from PyPDF2 import PdfReader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_with_pdfminer(pdf_path, page_numbers=None):
    try:
        resource_manager = PDFResourceManager()
        output_string = StringIO()
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            all_texts=True
        )
        
        converter = TextConverter(resource_manager, output_string, laparams=laparams)
        with open(pdf_path, 'rb') as pdf_file:
            for page in PDFPage.get_pages(pdf_file, page_numbers):
                try:
                    interpreter = PDFPageInterpreter(resource_manager, converter)
                    interpreter.process_page(page)
                except Exception as e:
                    logging.warning(f"Error processing page: {e}")
                    continue
                    
        text = output_string.getvalue()
        converter.close()
        output_string.close()
        return text
    except Exception as e:
        logging.error(f"PDFMiner extraction failed: {e}")
        return None

def extract_text_with_pypdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logging.error(f"PyPDF2 extraction failed: {e}")
        return None

def extract_text_from_images(pdf_path):
    try:
        pages = convert_from_path(pdf_path, dpi=300)
        extracted_text = []
        
        for i, page in enumerate(pages):
            img_np = np.array(page)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            text = pytesseract.image_to_string(threshold, config=custom_config)
            extracted_text.append(text)
        
        return "\n".join(extracted_text)
    except Exception as e:
        logging.error(f"OCR extraction failed: {e}")
        return None

def extract_tables_with_camelot(pdf_path):
    all_tables = []
    try:
        tables_stream = camelot.read_pdf(
            pdf_path, 
            flavor='stream',
            edge_tol=500,
            row_tol=10,
            strip_text='\n'
        )
        
        tables_lattice = camelot.read_pdf(
            pdf_path,
            flavor='lattice',
            line_scale=40
        )
        
        for tables in [tables_stream, tables_lattice]:
            for table in tables:
                if table.parsing_report['accuracy'] > 80:
                    df = table.df
                    df = df.replace('\n', ' ', regex=True)
                    df = df.replace('^\s*$', pd.NA, regex=True)
                    df = df.dropna(how='all')
                    all_tables.append(df)
        
        if all_tables:
            final_df = pd.concat(all_tables, ignore_index=True)
            return final_df
        
    except Exception as e:
        logging.error(f"Camelot extraction failed: {e}")
    
    return None

def process_pdf(pdf_path, output_csv="output.csv"):
    logging.info(f"Processing PDF: {pdf_path}")
    
    extracted_text = (
        extract_text_with_pdfminer(pdf_path) or 
        extract_text_with_pypdf(pdf_path) or 
        extract_text_from_images(pdf_path)
    )
    
    if extracted_text:
        with open("extracted_text.txt", "w", encoding="utf-8") as file:
            file.write(extracted_text)
        logging.info("Text extraction completed")
    else:
        logging.warning("Text extraction failed with all methods")
    
    table_df = extract_tables_with_camelot(pdf_path)
    
    if table_df is not None:
        table_df = table_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        table_df = table_df.replace(r'^\s*$', pd.NA, regex=True)
        table_df = table_df.dropna(how='all', axis=1)
        table_df = table_df.dropna(how='all', axis=0)
        
        table_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        logging.info(f"Table extracted and saved to {output_csv}")
        
        excel_output = output_csv.replace('.csv', '.xlsx')
        table_df.to_excel(excel_output, index=False)
        logging.info(f"Table also saved to Excel: {excel_output}")
    else:
        logging.warning("No tables detected or extraction failed")
    
    return output_csv

if __name__ == "__main__":

    pdf_path = r"F:\HACK-SPHERE\apreports-11.pdf"
    try:
        output_file = process_pdf(pdf_path, "extracted_data.csv")
        logging.info(f"Extraction complete. Files saved: {output_file}")
    except Exception as e:
        logging.error(f"Processing failed: {e}")
