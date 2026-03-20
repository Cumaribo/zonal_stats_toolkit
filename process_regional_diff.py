import geopandas as gpd
import pandas as pd
import glob
import os

OUTPUT_DIR = "output_diff_consolidated"
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
    return col.replace('mean_', '')

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

        # Extract mean columns and rename them to our canonical absolute change suffix
        mean_cols = [c for c in df.columns if c.startswith('mean_')]
        rename_dict = {c: f"{get_canonical_name(c)}_abs_chg" for c in mean_cols}
        
        df_clean = df[[key] + mean_cols].copy()
        df_clean.rename(columns=rename_dict, inplace=True)

        # Load Vector and Dissolve boundaries
        print(f"[{name}] Loading base vector {vector_path}...")
        gdf = gpd.read_file(vector_path)
        
        print(f"[{name}] Dissolving geometry by '{key}'...")
        gdf = gdf.dropna(subset=[key])
        gdf_dissolved = gdf[[key, 'geometry']].dissolve(by=key).reset_index()

        print(f"[{name}] Joining attributes...")
        gdf_final = gdf_dissolved.merge(df_clean, on=key, how='left')

        out_gpkg = os.path.join(OUTPUT_DIR, f"{name}_services_diff.gpkg")
        print(f"[{name}] Saving to {out_gpkg}...\n" + "-"*50)
        gdf_final.to_file(out_gpkg, driver="GPKG")

if __name__ == "__main__":
    main()