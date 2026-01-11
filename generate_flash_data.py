import csv
import os

INPUT_FILE = "codn_training_data.txt"
OUTPUT_FILE = "flash_clinical_data.csv"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Writing {len(lines)} lines to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text"]) # Header
        for line in lines:
            if line.strip():
                writer.writerow([line.strip()])
    
    print("Done.")

if __name__ == "__main__":
    main()
