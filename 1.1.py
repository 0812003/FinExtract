import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from datasets import load_dataset
from PIL import Image, ImageDraw
import pandas as pd
import os
import re
import logging
import csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TableExtractor:
    def __init__(self, model_name="microsoft/layoutlmv3-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name).to(self.device)

    def convert_pdf_to_images(self, pdf_path):
        try:
            dataset = load_dataset("imagefolder", data_dir=pdf_path, split="train")
            images = [image["image"] for image in dataset]  # Extract PIL images
            return images
        except Exception as e:
            logging.error(f"Error converting PDF to images: {e}")
            return []

    def extract_text_and_boxes(self, image):
        encoding = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        boxes = encoding.bbox[0]  # Get bounding boxes
        words = self.processor.tokenizer.convert_ids_to_tokens(outputs.logits.argmax(dim=-1)[0])
        words = self.processor.tokenizer.decode(words)
        words = words.replace("<s>", "").replace("</s>", "").replace("</pad>", "").strip() # Clean up special tokens
        words = words.split()

        # Combine words and boxes (LayoutLM outputs are sometimes misaligned)
        extracted_data = []
        box_index = 0
        for word in words:
            if box_index < len(boxes):
                extracted_data.append({
                    "box": boxes[box_index].tolist(),  # Convert to list for easier handling
                    "word": word
                })
                box_index += 1
        
        boxes = [item["box"] for item in extracted_data]
        words = [item["word"] for item in extracted_data]

        return words, boxes, [100]*len(words) # dummy confidence values


    def draw_boxes(self, image, boxes, words, output_path):
        draw = ImageDraw.Draw(image)
        for box, word in zip(boxes, words):
            draw.rectangle(box, outline='red', width=2)
            draw.text((box[0], box[1] - 10), word, fill='red')
        image.save(output_path)

    def group_words_into_lines(self, boxes, words, tolerance=5):
        lines = []
        current_line = []
        current_y = None

        for box, word in zip(boxes, words):
            _, y, _, _ = box
            if current_y is None or abs(y - current_y) <= tolerance:
                current_line.append((box, word))
                current_y = y
            else:
                lines.append(current_line)
                current_line = [(box, word)]
                current_y = y
        lines.append(current_line)
        return lines

    def extract_table_structure(self, lines):
        table = []
        for line in lines:
            row = []
            line.sort(key=lambda item: item[0][0])  # Sort by x-coordinate of the box
            for box, word, _ in line:  # Unpack box, word, and confidence (ignore confidence)
                row.append(word)
            table.append(row)
        return table
    def create_csv_from_lines(self, lines, output_path):
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            for line in lines:
                row = []
                for box, word, _ in line: # Unpack box, word, and confidence
                    row.append(word)
                writer.writerow(row)

    def group_words_into_lines(self, boxes, words, confidences, tolerance=5):  # Added confidences here
        lines = []
        current_line = []
        current_y = None

        for i, (box, word, confidence) in enumerate(zip(boxes, words, confidences)):  # Enumerate and unpack
            _, y, _, _ = box
            if current_y is None or abs(y - current_y) <= tolerance:
                current_line.append((box, word, confidence))  # Added confidence here
                current_y = y
            else:
                lines.append(current_line)
                current_line = [(box, word, confidence)]  # Added confidence here
                current_y = y
        lines.append(current_line)
        return lines

    def create_line_boxes(self, lines):
        line_boxes = []
        for line in lines:
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')

            for item in line:  # Iterate through the items in the line
                box, _, _ = item # unpack the item to box, word, and confidence
                x1, y1, x2, y2 = box # unpack the box
                min_x = min(min_x, x1)
                min_y = min(min_y, y1)
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)

            if min_x != float('inf'):
                line_boxes.append([min_x, min_y, max_x, max_y])
            else:
                line_boxes.append(None)

        return line_boxes

    def draw_line_boxes(self, image, line_boxes, output_path):
        draw = ImageDraw.Draw(image)
        for box in line_boxes:
            if box: # check if box is not None
                draw.rectangle(box, outline='blue', width=2)  # Draw line boxes in blue
        image.save(output_path)



    def process_pdf(self, pdf_path, output_dir="extracted_output"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_tables = []
        images = self.convert_pdf_to_images(pdf_path)

        for page_num, image in enumerate(images):
            words, boxes, confidences = self.extract_text_and_boxes(image)
            if not words:
                continue

            output_path_boxes = os.path.join(output_dir, f"page_{page_num + 1}_boxes.png")
            self.draw_boxes(image.copy(), boxes, words, output_path_boxes)

            lines = self.group_words_into_lines(boxes, words, confidences)  # Pass confidences here
            line_boxes = self.create_line_boxes(lines)

            output_path_line_boxes = os.path.join(output_dir, f"page_{page_num + 1}_line_boxes.png")
            self.draw_line_boxes(image.copy(), line_boxes, output_path_line_boxes)  # Draw and save line boxes

            table = self.extract_table_structure(lines)
            if table:
                all_tables.append(table)
                try:
                    df = pd.DataFrame(table)  # Try to create DataFrame

                    # Check if DataFrame is empty or contains only NaNs
                    if df.empty or df.isnull().all().all(): 
                        print(f"Warning: Table on page {page_num + 1} is empty or contains only NaN values. Skipping.")
                        continue # Skip to the next page if the table is empty.

                    output_path_excel = os.path.join(output_dir, f"page_{page_num + 1}_table.xlsx")
                    df.to_excel(output_path_excel, index=False)
                    print(f"Table from page {page_num + 1} saved to {output_path_excel}")

                    output_path_lines_csv = os.path.join(output_dir, f"page_{page_num + 1}_lines.csv")
                    self.create_csv_from_lines(lines, output_path_lines_csv)
                    print(f"Lines from page {page_num + 1} saved to {output_path_lines_csv}")

                except ValueError:
                    print(f"Warning: Could not convert table on page {page_num + 1} to DataFrame. Likely an irregular table structure.")
                    print(table)
                except Exception as e:  # Catch other exceptions during DataFrame creation
                    print(f"An error occurred during DataFrame creation: {e}")
                    continue # Skip to the next page in case of an error

        return all_tables

def main():  
    pdf_path = r"F:\HACK-SPHERE\Data\Sample (5).pdf" 
    output_dir = "extracted_output"

    try:
        extractor = TableExtractor()
        tables = extractor.process_pdf(pdf_path, output_dir)

        for page_num, table in enumerate(tables):
            print(f"\nPage {page_num + 1} Table:")
            for row in table:
                print(row)
            print("===")

        print(f"\nExtracted tables saved to: {output_dir}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()