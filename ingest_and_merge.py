import os
import csv
import glob
import pandas as pd

# CONFIG
REAL_DATA_ROOT = r"D:\PROJECT\Healthcare-Sickcare\NEST\Data for problem Statement 1\QC Anonymized Study Files"
SYNTHETIC_INPUT = "codn_training_data.txt"
OUTPUT_FILE = "flash_clinical_data.csv"

def parse_real_csv(file_path):
    """
    Intelligently parses the messy multi-header CSV to extract:
    Site ID, Missing Visits, Total Queries, SAE Counts.
    """
    narratives = []
    try:
        # Read first few lines to find headers
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [f.readline() for _ in range(5)]
        
        # Determine Separator (likely comma)
        sep = ','

        # Find Indices
        idx_site = -1
        idx_missing = -1
        idx_queries = -1
        idx_sae = -1
        
        # Helper to find index in a split line
        def find_col(row_lines, keywords):
            for i, line in enumerate(row_lines):
                parts = [p.strip().replace('"', '') for p in line.split(sep)]
                for col_i, text in enumerate(parts):
                    if any(k.lower() in text.lower() for k in keywords):
                        return col_i
            return -1

        idx_site = find_col(lines, ["site id", "site identifier"])
        idx_missing = find_col(lines, ["missing visits"])
        idx_queries = find_col(lines, ["#total queries", "total queries"])
        idx_sae = find_col(lines, ["esae dashboard review for safety", "sae count"])

        # Fallbacks (Hardcoded from visual inspection if search fails)
        if idx_site == -1: idx_site = 3
        if idx_missing == -1: idx_missing = 7
        if idx_queries == -1: idx_queries = 30 
        if idx_sae == -1: idx_sae = 14

        # Read Data
        # Skip top 4 lines of mess
        df = pd.read_csv(file_path, skiprows=4, header=None, encoding='utf-8', on_bad_lines='skip')
        
        for _, row in df.iterrows():
            try:
                site_id = str(row[idx_site]) if pd.notna(row[idx_site]) else "UNKNOWN"
                
                # Check termination row (often has "Total:" in site col)
                if "Total" in site_id: continue

                # Missing
                missing = row[idx_missing]
                missing = int(missing) if pd.notna(missing) and str(missing).isdigit() else 0
                
                # Queries
                queries = row[idx_queries]
                queries = int(queries) if pd.notna(queries) and str(queries).isdigit() else 0
                
                # SAE
                sae = row[idx_sae]
                sae = int(sae) if pd.notna(sae) and str(sae).isdigit() else 0
                
                # Logic Gate
                # Recalculate Logic to be 100% Correct
                status_token = "[CLEAN]"
                if missing > 0 or queries > 0 or sae > 0:
                     status_token = "[RISK]"
                
                # Create Narrative
                txt = (
                    f"Timepoint T(REAL). Site {site_id} context: Region RealWorld. "
                    f"Status Report: Subject REAL-{row.name} has "
                    f"{missing} missed visits and {queries} open queries. "
                    f"Safety Signal: {sae} events. "
                    f"[LOGIC_GATE]: Therefore, Status: {status_token}."
                )
                narratives.append(txt)

            except Exception as e:
                continue # Skip bad rows

    except Exception as e:
        print(f"Failed to parse {file_path}: {e}")
    
    return narratives

def main():
    all_data = []

    # 1. Ingest Synthetic Data
    if os.path.exists(SYNTHETIC_INPUT):
        print(f"Loading synthetic data from {SYNTHETIC_INPUT}...")
        with open(SYNTHETIC_INPUT, 'r', encoding='utf-8') as f:
            all_data.extend([l.strip() for l in f.readlines() if l.strip()])
    else:
        print("Warning: Synthetic data not found. Using only real data.")

    # 2. Ingest Real Data
    print(f"Scanning {REAL_DATA_ROOT} for Metrics CSVs...")
    search_pattern = os.path.join(REAL_DATA_ROOT, "**", "*Subject_Level_Metrics.csv")
    real_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(real_files)} real data files.")
    
    for f in real_files:
        print(f"Parsing {os.path.basename(f)}...")
        narratives = parse_real_csv(f)
        all_data.extend(narratives)
        print(f"  Added {len(narratives)} records.")

    # 3. Write Unified Dataset
    print(f"Writing {len(all_data)} total records to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        for line in all_data:
            writer.writerow([line])

    print("Data Ingestion Complete.")

if __name__ == "__main__":
    main()
