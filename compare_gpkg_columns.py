import geopandas as gpd
import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime

def compare_gpkg_vs_csv(gpkg_path, csv_path, report_path="pipeline_validation_report.txt"):
    # Helper to print to both console and file
    with open(report_path, 'w') as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        log(f"--- Pipeline Comparison: Optimized (GPKG) vs Legacy (CSV) ---")
        log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Optimized Pipeline (GPKG): {gpkg_path}")
        log(f"Legacy Pipeline (CSV):     {csv_path}")
        log("-" * 40)

        # 1. Load GPKG
        if not os.path.exists(gpkg_path):
            log(f"Error: GPKG not found at {gpkg_path}")
            return
        
        log("Loading GPKG (Optimized Data)...")
        try:
            gdf = gpd.read_file(gpkg_path, ignore_geometry=True)
        except TypeError:
            gdf = gpd.read_file(gpkg_path)
            if 'geometry' in gdf.columns:
                gdf = gdf.drop(columns=['geometry'])
        
        log(f"  Rows loaded: {len(gdf)}")

        # 2. Load CSV
        if not os.path.exists(csv_path):
            log(f"Error: CSV not found at {csv_path}")
            return
        
        log("Loading CSV (Legacy Data)...")
        df_csv = pd.read_csv(csv_path)
        log(f"  Rows loaded: {len(df_csv)}")

        # 3. Align Data
        # FIX: GeoPackages often store FID in the index when read by geopandas.
        # We check if 'fid' is missing and reset the index to expose it.
        if 'fid' not in gdf.columns and not any(c in gdf.columns for c in ['FID', 'grid_fid', 'id', 'ID']):
            log("  Note: FID likely in GPKG index. Resetting index to column.")
            gdf.reset_index(inplace=True)
            if 'index' in gdf.columns:
                gdf.rename(columns={'index': 'fid'}, inplace=True)

        # Normalize column names if possible aliases exist
        for df in [gdf, df_csv]:
            if 'fid' not in df.columns:
                for alias in ['FID', 'grid_fid', 'id', 'ID']:
                    if alias in df.columns:
                        df.rename(columns={alias: 'fid'}, inplace=True)
                        break
        
        # If fid is still missing in either, fall back to row-index based alignment
        if 'fid' not in gdf.columns or 'fid' not in df_csv.columns:
            log("Warning: Explicit 'fid' column not found in one of the files.")
            if len(gdf) == len(df_csv):
                log("Generating surrogate 'fid' on the fly based on row order (Counts match).")
                gdf['fid'] = np.arange(len(gdf))
                df_csv['fid'] = np.arange(len(df_csv))
            else:
                log("Error: Cannot align data. 'fid' missing and row counts differ.")
                return

        log(f"Merging datasets on 'fid'...")
        merged = pd.merge(gdf, df_csv, on='fid', suffixes=('_gpkg', '_csv'))
        log(f"  Common aligned rows: {len(merged)}")
        log("-" * 40)

        # 4. Compare Specific Columns
        gpkg_targets = [
            ('GPKG 1992', 'nature_access_lspop2019_ESA1992_mean'),
            ('GPKG 2020', 'nature_access_lspop2019_ESA2020_mean')
        ]
        
        # Identify CSV column
        csv_col = None
        for c in df_csv.columns:
            # Look for mean or avg, case insensitive
            if ('mean' in c.lower() or 'avg' in c.lower()) and 'fid' not in c.lower():
                csv_col = c
                break
        if not csv_col:
            csv_col = [c for c in df_csv.columns if c != 'fid'][0]

        log(f"Legacy CSV Column: '{csv_col}'")

        for label, g_col in gpkg_targets:
            log(f"\nComparing Legacy CSV vs {label} ({g_col})")
            
            if g_col not in merged.columns:
                log(f"  [Skipping - Column not found in GPKG]")
                continue
                
            s_csv = merged[csv_col].astype(float)
            s_gpkg = merged[g_col].astype(float)
            
            # Compare only valid pairs
            mask = s_csv.notna() & s_gpkg.notna()
            diff = s_csv[mask] - s_gpkg[mask]
            
            if len(diff) == 0:
                log(f"  Status: No valid common data points.")
                continue
                
            mae = diff.abs().mean()
            rmse = np.sqrt((diff**2).mean())
            max_diff = diff.abs().max()
            
            # Calculate normalized error (percentage of the mean value)
            mean_val = s_gpkg[mask].mean()
            nrmse = (rmse / abs(mean_val)) if mean_val != 0 else np.inf
            
            log(f"    Mean Absolute Error (MAE): {mae:.6f}")
            log(f"    RMSE:                      {rmse:.6f}")
            log(f"    Normalized RMSE:           {nrmse:.4%}")
            log(f"    Max Difference:            {max_diff:.6f}")
            
            if nrmse < 0.0001: # 0.01% error tolerance
                log("    -> CONCLUSION: PERFECT MATCH (Identical)")
            elif nrmse < 0.01: # 1% error tolerance
                log("    -> CONCLUSION: HIGH FIDELITY (Minor numerical differences)")
            else:
                log("    -> CONCLUSION: DIFFERENT DATASET (Does not match)")

        log("\n" + "="*40)
        log(f"Report saved to: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GPKG nature access vs CSV output.")
    parser.add_argument("gpkg_path", nargs='?', default="/data/interim/10k_grid_services_base.gpkg")
    parser.add_argument("csv_path", nargs='?', default="output_access_grid/grid_10km_20260310_194649.csv")
    args = parser.parse_args()
    
    compare_gpkg_vs_csv(args.gpkg_path, args.csv_path)