import geopandas as gpd
import pandas as pd
import numpy as np
import glob
import os

OUTPUT_DIR = "output_consolidated"
VECTOR_DIR = "data/vector_basedata"

JOBS = [
    {"name": "country", "key": "nev_name", "vector": f"{VECTOR_DIR}/cartographic_ee_r264_correspondence.gpkg"},
    {"name": "region_wb", "key": "region_wb", "vector": f"{VECTOR_DIR}/cartographic_ee_r264_correspondence.gpkg"},
    {"name": "income_grp", "key": "income_grp", "vector": f"{VECTOR_DIR}/cartographic_ee_r264_correspondence.gpkg"},
    {"name": "biome", "key": "WWF_biome", "vector": f"{VECTOR_DIR}/Biome.gpkg"}
]

def get_canonical_name(col):
    """Maps complex runner.py column names to canonical service names."""
    c = col.lower()
    if 'nature_access' in c: return 'Nature_Access'
    if 'rt_ratio' in c: return 'C_Risk_Red_Ratio'
    if 'rt_' in c: return 'C_Risk'
    if 'sed_ret_ratio' in c: return 'Sed_Ret_Ratio'
    if 'n_ret_ratio' in c: return 'N_Ret_Ratio'
    if 'n_export' in c: return 'N_export'
    if 'sed_export' in c: return 'Sed_export'
    if 'pollination' in c: return 'Pollination'
    if 'n_retention' in c: return 'N_retention'
    if 'usle' in c: return 'USLE'
    # Fallback
    return col.replace('mean_', '').replace('_1992', '').replace('1992', '')

def get_latest_csv(job_name):
    """Finds the most recently generated CSV for a given job."""
    files = glob.glob(os.path.join(OUTPUT_DIR, f"{job_name}_*.csv"))
    if not files:
        files = glob.glob(os.path.join(OUTPUT_DIR, f"{job_name}.csv"))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def main():
    for job in JOBS:
        name = job["name"]
        key = job["key"]
        vector_path = job["vector"]

        csv_path = get_latest_csv(name)
        if not csv_path:
            print(f"[{name}] SKIP: No CSV found in {OUTPUT_DIR}.")
            continue

        print(f"[{name}] Processing {csv_path}...")
        df = pd.read_csv(csv_path)

        # 1. Identify 1992 columns for 'mean' stats
        mean_cols = [c for c in df.columns if c.startswith('mean_')]
        cols_1992 = [c for c in mean_cols if '1992' in c]
        
        computed_cols = []

        for c_1992 in cols_1992:
            c_2020 = c_1992.replace('1992', '2020')
            if c_2020 in df.columns:
                canonical = get_canonical_name(c_1992)
                abs_col = f"{canonical}_abs_chg"
                pct_col = f"{canonical}_pct_chg"

                old_val = df[c_1992]
                new_val = df[c_2020]

                # Absolute Change
                df[abs_col] = new_val - old_val

                # Symmetric Percentage Change
                denominator = new_val.abs() + old_val.abs()
                
                # Avoid division by zero, enforce 0-to-0 logic
                pct_chg = np.where(
                    (old_val == 0) & (new_val == 0),
                    0.0,
                    (200.0 * (new_val - old_val)) / denominator
                )
                
                df[pct_col] = pct_chg
                computed_cols.extend([c_1992, c_2020, abs_col, pct_col])
                print(f"  -> Calculated: {canonical} (abs & pct)")

        # Keep only the grouping key and the computed metrics to keep GPKG clean
        keep_cols = [key] + computed_cols
        df_clean = df[[c for c in keep_cols if c in df.columns]]

        # 2. Load Vector and Dissolve boundaries
        print(f"[{name}] Loading base vector {vector_path}...")
        gdf = gpd.read_file(vector_path)
        
        print(f"[{name}] Dissolving geometry by '{key}'...")
        # Drop rows where key is NA before dissolving
        gdf = gdf.dropna(subset=[key])
        gdf_dissolved = gdf[[key, 'geometry']].dissolve(by=key).reset_index()

        # 3. Merge data
        print(f"[{name}] Joining attributes...")
        gdf_final = gdf_dissolved.merge(df_clean, on=key, how='left')

        # 4. Save
        out_gpkg = os.path.join(OUTPUT_DIR, f"{name}_services_change.gpkg")
        print(f"[{name}] Saving to {out_gpkg}...")
        gdf_final.to_file(out_gpkg, driver="GPKG")
        print("-" * 50)

if __name__ == "__main__":
    main()