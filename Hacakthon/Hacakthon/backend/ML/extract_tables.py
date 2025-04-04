import torch
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageDraw
import pandas as pd
import os
import re
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import logging
# Path to Tesseract OCR executable (update if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TableExtractor:
    def init(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def get_line_height_tolerance(self, boxes):
        heights = [box[3] - box[1] for box in boxes]
        return np.mean(heights) * 0.5

    def convert_pdf_to_images(self, pdf_path):
        try:
            images = convert_from_path(pdf_path)
            return images
        except Exception as e:
            logging.error(f"Error converting PDF to images: {e}")
            return []

    def extract_text_and_boxes(self, image):
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = []
        boxes = []
        confidences = []

        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip():
                words.append(ocr_data['text'][i])
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                boxes.append([x, y, x + w, y + h])
                confidences.append(ocr_data['conf'][i])

        return words, boxes, confidences

    def detect_columns(self, boxes, words):
        x_coords = [box[0] for box in boxes]
        x_coords = np.array(x_coords).reshape(-1, 1)
        
        clustering = DBSCAN(eps=20, min_samples=2).fit(x_coords)
        unique_clusters = np.unique(clustering.labels_)
        
        columns = []
        for cluster in unique_clusters:
            if cluster != -1:
                cluster_indices = np.where(clustering.labels_ == cluster)[0]
                min_x = min(x_coords[cluster_indices])
                max_x = max(x_coords[cluster_indices])
                columns.append((min_x[0], max_x[0]))
        
        return sorted(columns)

    def should_merge_words(self, box1, box2, tolerance):
        x1, y1, x1_end, y1_end = box1
        x2, y2, x2_end, y2_end = box2
        
        vertical_overlap = (min(y1_end, y2_end) - max(y1, y2)) > 0
        horizontal_distance = x2 - x1_end
        
        return vertical_overlap and horizontal_distance < tolerance

    def draw_boxes(self, image, lines, output_path):
        draw = ImageDraw.Draw(image)
        for combined_box, words_with_boxes in lines:
            for (x0, y0, x1, y1), word in words_with_boxes:
                draw.rectangle((x0, y0, x1, y1), outline='blue', width=1)
            draw.rectangle(combined_box, outline='red', width=2)
        try:
            image.save(output_path)
        except Exception as e:
            logging.error(f"Error saving image: {e}")

    def group_words_into_lines(self, boxes, words):
        tolerance = self.get_line_height_tolerance(boxes)
        lines = []
        current_line = []
        current_y = None

        sorted_items = sorted(zip(boxes, words), key=lambda x: (x[0][1], x[0][0]))
        
        for box, word in sorted_items:
            _, y, _, _ = box
            if current_y is None or abs(y - current_y) <= tolerance:
                current_line.append((box, word))
                current_y = y
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [(box, word)]
                current_y = y

        if current_line:
            lines.append(current_line)

        combined_lines = []
        for line in lines:
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            
            line.sort(key=lambda x: x[0][0])
            
            for box, _ in line:
                x0, y0, x1, y1 = box
                min_x = min(min_x, x0)
                min_y = min(min_y, y0)
                max_x = max(max_x, x1)
                max_y = max(max_y, y1)
            
            combined_box = (min_x, min_y, max_x, max_y)
            combined_lines.append((combined_box, line))

        return combined_lines

    def extract_table_structure(self, lines):
        table = []
        
        for combined_box, words_with_boxes in lines:
            row = []
            current_text = ""
            prev_x1 = None
            prev_word = None
            
            for box, word in words_with_boxes:
                x0, _, x1, _ = box
                
                # Handle hyphenated words
                if prev_word and prev_word.endswith('-'):
                    current_text = current_text[:-1] + word  # Remove hyphen and concatenate
                    prev_word = word
                    continue
                    
                if prev_x1 is not None and x0 - prev_x1 > 16:  # Gap detection
                    if current_text.strip():
                        row.append(current_text.strip())
                    current_text = word
                else:
                    if current_text:
                        current_text += " " + word
                    else:
                        current_text = word
                
                prev_x1 = x1
                prev_word = word
            
            if current_text.strip():
                row.append(current_text.strip())
            
            if row:
                table.append(row)

        return self.refine_table_structure(table)

    def refine_table_structure(self, table):
        max_cols = max(len(row) for row in table)
        refined_table = []
        
        for row in table:
            cleaned_row = []
            for cell in row:
                if '\t' in cell:
                    parts = cell.split('\t')
                    cleaned_parts = [re.sub(r'[^\w\s.,-]', '', part.strip()) for part in parts]
                    cleaned_row.extend(cleaned_parts)
                else:
                    cleaned_cell = re.sub(r'[^\w\s.,-]', '', cell.strip())
                    cleaned_row.append(cleaned_cell)
            
            while len(cleaned_row) < max_cols:
                cleaned_row.append("")
                
            refined_table.append(cleaned_row)
        
        return refined_table

    def process_pdf(self, pdf_path, output_dir="extracted_output"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        images = self.convert_pdf_to_images(pdf_path)
        all_tables = []

        for page_num, image in enumerate(images):
            words, boxes, confidences = self.extract_text_and_boxes(image)
            if not words:
                continue

            lines = self.group_words_into_lines(boxes, words)
            output_path_boxes = os.path.join(output_dir, f"page_{page_num + 1}_boxes.png")
            self.draw_boxes(image.copy(), lines, output_path_boxes)

            table = self.extract_table_structure(lines)

            if table:
                all_tables.append(table)
                try:
                    df = pd.DataFrame(table)
                    output_path_excel = os.path.join(output_dir, f"page_{page_num + 1}_table.xlsx")
                    df.to_excel(output_path_excel, index=False)
                    
                    output_path_csv = os.path.join(output_dir, f"page_{page_num + 1}_table.csv")
                    df.to_csv(output_path_csv, index=False, encoding="utf-8")
                    print(f"Table from page {page_num + 1} saved to Excel and CSV")
                except Exception as e:
                    logging.error(f"Error saving table from page {page_num + 1}: {e}")

        return all_tables
import numpy as np
from sklearn.cluster import DBSCAN
import re
import os
import pandas as pd
import logging
from PIL import ImageDraw

class EnhancedTableExtractor(TableExtractor):
    def detect_columns(self, boxes, words):
       
        x_coords = np.array([box[0] for box in boxes]).reshape(-1, 1)  
        y_coords = np.array([box[1] for box in boxes])  

        clustering = DBSCAN(eps=25, min_samples=2).fit(x_coords)
        labels = clustering.labels_

        column_clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue  
            if label not in column_clusters:
                column_clusters[label] = []
            column_clusters[label].append((x_coords[i][0], y_coords[i], words[i], boxes[i]))

        sorted_columns = sorted(column_clusters.items(), key=lambda x: min([entry[0] for entry in x[1]]))
        
        column_boundaries = []
        for cluster_id, column_words in sorted_columns:
            min_x = min([entry[0] for entry in column_words])
            max_x = max([entry[3][2] for entry in column_words])  
            column_boundaries.append((min_x, max_x))

        return column_boundaries
    
    

    def extract_table_structure(self, lines, column_boundaries):
        
        table = []
        
        for combined_box, words_with_boxes in lines:
            row = [""] * len(column_boundaries)  
            words_with_boxes.sort(key=lambda x: x[0][0])  

            for box, word in words_with_boxes:
                x0, _, x1, _ = box

                best_col_idx = -1
                min_distance = float("inf")

                for col_idx, (col_min, col_max) in enumerate(column_boundaries):
                    center_x = (x0 + x1) / 2  
                    if col_min <= center_x <= col_max:
                        best_col_idx = col_idx
                        break
                    distance = min(abs(center_x - col_min), abs(center_x - col_max))
                    if distance < min_distance:
                        min_distance = distance
                        best_col_idx = col_idx  

                if row[best_col_idx]:
                    row[best_col_idx] += " " + word
                else:
                    row[best_col_idx] = word

            row = [cell.strip() for cell in row]  
            table.append(row)

        return self.refine_table_structure(table)

    def refine_table_structure(self, table):
      
        max_cols = max(len(row) for row in table)
        refined_table = []

        for row in table:
            cleaned_row = [re.sub(r'[^\w\s.,-]', '', cell.strip()) for cell in row]
            cleaned_row.extend([""] * (max_cols - len(cleaned_row)))  
            refined_table.append(cleaned_row)

        return refined_table

    def process_pdf(self, pdf_path, output_dir="extracted_output"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        images = self.convert_pdf_to_images(pdf_path)
        all_tables = []
        final_table_data = []

        for page_num, image in enumerate(images):
            words, boxes, confidences = self.extract_text_and_boxes(image)
            if not words:
                continue

            column_boundaries = self.detect_columns(boxes, words)  
            lines = self.group_words_into_lines(boxes, words)

            table = self.extract_table_structure(lines, column_boundaries)

            if table:
                all_tables.append(table)
                final_table_data.extend(table) 

        if final_table_data:
            try:
                df = pd.DataFrame(final_table_data)
                output_path_csv = os.path.join(output_dir, "final_extracted_table.csv")
                df.to_csv(output_path_csv, index=False, encoding="utf-8")
                logging.info(f"Final extracted table saved as {output_path_csv}")
            except Exception as e:
                logging.error(f"Error saving final extracted table: {e}")

        return all_tables
