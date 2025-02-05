import pandas as pd
import re
from Collectpdf import documents

def prepare_training_data(documents):
    """Automatically label the first line as 'Vendor', detect horizontal lines, and label column headers.
       Allows skipping the remaining lines by pressing '5'."""
    labeled_data = []
    vendor_detected = False
    horizontal_line_detected = False

    # Regular expression for detecting horizontal lines (at least 5 consecutive dashes)
    horizontal_line_pattern = r"^\s*_{15,}\s*$"  

    for doc in documents:
        lines = doc.split("\n")

        for idx, line in enumerate(lines):
            line = line.strip()  # Remove leading/trailing spaces

            # Reset vendor detection if a document is skipped
            if vendor_detected and labeled_data and labeled_data[-1][1] == 5:
                vendor_detected = False

            # Auto-detect and label vendor (first non-empty line)
            if not vendor_detected and line:
                labeled_data.append((line, 0))  # Vendor = 0
                vendor_detected = True
                print(f"Auto-labeled as Vendor: {line}")
                continue

            # Detect horizontal line
            if re.match(horizontal_line_pattern, line):
                horizontal_line_detected = True
                labeled_data.append((line, 3))  # Label 3 = Horizontal Line
                print(f"Detected Horizontal Line: {line}")
                continue

            # After detecting horizontal line, label as headers or skip based on user input
            if horizontal_line_detected:
                while True:
                    label = input(f"Label column header '{line}' (4=Header, 2=Other, 5=Data): ")
                    if label in ["4", "2", "5"]:
                        break
                    print("Invalid input. Please enter 4, 2, or 5.")
                
                if label == "5":
                    print("Skipping and assigning label 5 to remaining lines in this document...")
                    labeled_data.extend([(l.strip(), 5) for l in lines[idx:]])  # Assign label 5 to all remaining lines
                    break  
                else:
                    labeled_data.append((line, int(label)))
            else:
                # Label for non-header lines before detecting horizontal line
                while True:
                    label = input(f"Label for '{line}' (1=Invoice, 2=Other, 5=Data): ")
                    if label in ["1", "2", "5"]:
                        break
                    print("Invalid input. Please enter 1, 2, or 5.")
                
                if label == "5":
                    print("Skipping and assigning label 5 to remaining lines in this document...")
                    labeled_data.extend([(l.strip(), 5) for l in lines[idx:]])  # Assign label 5 to all remaining lines
                    break  
                else:
                    labeled_data.append((line, int(label)))

    # Save labeled data
    df = pd.DataFrame(labeled_data, columns=["text", "label"])
    df.to_csv("invoice_training_data.csv", index=False)
    return df


df = prepare_training_data(documents)
