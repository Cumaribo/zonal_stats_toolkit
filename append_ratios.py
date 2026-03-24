import os
import pandas as pd
import glob

# Define directories
base_dir = "output_consolidated"
new_dir = "output_ratios"

# Merge keys based on file prefixes
MERGE_KEYS = {
    'biome': 'WWF_biome',
    'country': 'nev_name',
    'income_grp': 'income_grp',
    'region_wb': 'region_wb',
}

def append_ratios():
    for prefix, key in MERGE_KEYS.items():
        base_files = sorted(glob.glob(os.path.join(base_dir, f"{prefix}*.csv")), reverse=True)
        new_files = sorted(glob.glob(os.path.join(new_dir, f"{prefix}*.csv")), reverse=True)

        if not base_files:
            print(f"No base file found for {prefix} in {base_dir}")
            continue
        if not new_files:
            print(f"No new ratios file found for {prefix} in {new_dir}")
            continue

        base_file = base_files[0]
        new_file = new_files[0]

        print(f"Merging new ratios from: {os.path.basename(new_file)}")
        print(f"                   into: {os.path.basename(base_file)}")

        df_base = pd.read_csv(base_file)
        df_new = pd.read_csv(new_file)

        if key not in df_base.columns or key not in df_new.columns:
            print(f"  [ERROR] Merge key '{key}' missing! Skipping.\n")
            continue

        # Remove overlapping columns in base (e.g. if script is run twice) to prevent _x/_y suffixing
        overlap_cols = [c for c in df_base.columns if c in df_new.columns and c != key]
        if overlap_cols:
            df_base = df_base.drop(columns=overlap_cols)

        df_merged = pd.merge(df_base, df_new, on=key, how='left')
        
        # Overwrite the base file so the R script natively picks it up
        df_merged.to_csv(base_file, index=False)
        print(f"  -> Successfully updated: {base_file}\n")

if __name__ == "__main__":
    append_ratios()