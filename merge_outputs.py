#!/usr/bin/env python
import os
import pandas as pd
import glob
import re

# Dictionary mapping file prefixes to their merge keys
MERGE_KEYS = {
    'output/biome': 'WWF_biome',
    'output/continent': 'continent',
    'output/country': 'nev_name',
    'output/income_grp': 'income_grp',
    'output/region_un': 'region_un',
    'output/subregion': 'subregion'
}

def normalize_columns(df):
    """
    Normalizes column names to handle inconsistencies.
    e.g., avg_n_ret_... -> avg_n_retention_...
    """
    # Create a copy of the columns to avoid issues while iterating
    columns = df.columns.to_list()
    
    # Define patterns and their replacements
    replacements = {
        '_ret_': '_retention_',
        '_sed_': '_sediment_'
    }
    
    new_columns = {}
    for col in columns:
        original_col = col
        for pattern, replacement in replacements.items():
            col = re.sub(pattern, replacement, col)
        if original_col != col:
            new_columns[original_col] = col
            
    df.rename(columns=new_columns, inplace=True)
    return df

def merge_csvs():
    output_dir = 'output'
    # Exclude already merged files from the glob pattern
    files = glob.glob(os.path.join(output_dir, '*[0-9].csv'))
    
    if not files:
        print("No CSV files to merge were found.")
        return

    # Group files by their base name, e.g., 'output/biome'
    prefixes = set(f.split('_2026')[0] for f in files)

    for prefix in prefixes:
        prefix_files = sorted([f for f in files if f.startswith(prefix)], reverse=True)
        
        if len(prefix_files) < 2:
            print(f"Skipping {prefix}: Not enough files to merge.")
            continue

        # The newest file is the first one, the older one is the second
        new_file = prefix_files[0]
        old_file = prefix_files[1]

        print(f"Merging '{os.path.basename(old_file)}' and '{os.path.basename(new_file)}'...")

        df_new = pd.read_csv(new_file)
        df_old = pd.read_csv(old_file)

        # Normalize columns for both dataframes
        df_new = normalize_columns(df_new)
        df_old = normalize_columns(df_old)

        print("Columns for old file:", df_old.columns)
        print("Columns for new file:", df_new.columns)
        
        # Determine the merge key for the current prefix
        merge_key = MERGE_KEYS.get(prefix)
        if not merge_key or merge_key not in df_old.columns or merge_key not in df_new.columns:
            print(f"Merge key for '{prefix}' not found or not in both files. Skipping.")
            continue
            
        # Set index for update operation
        df_old.set_index(merge_key, inplace=True)
        df_new.set_index(merge_key, inplace=True)
        
        # Update old dataframe with new data. This overwrites existing
        # columns and adds new ones.
        df_old.update(df_new)
        
        # Reset index to bring the merge key back as a column
        merged_df = df_old.reset_index()

        # Define the output filename, e.g., 'output/biome_merged.csv'
        output_filename = f"{prefix}_merged.csv"
        merged_df.to_csv(output_filename, index=False)
        print(f"  -> Saved merged file to '{os.path.basename(output_filename)}'")

if __name__ == "__main__":
    merge_csvs()