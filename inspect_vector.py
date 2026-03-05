import geopandas as gpd
import sys
import os

def inspect(path):
    print(f"--- Inspecting {path} ---")
    if not os.path.exists(path):
        print("File not found!")
        return

    try:
        gdf = gpd.read_file(path)
        print(f"Rows: {len(gdf)}")
        print(f"Columns: {list(gdf.columns)}")
        if 'grid_fid' in gdf.columns:
            nulls = gdf['grid_fid'].isnull().sum()
            print(f"grid_fid nulls: {nulls}")
            print(f"grid_fid sample: {gdf['grid_fid'].head().tolist()}")
        else:
            print("grid_fid column NOT found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        print("Usage: python inspect_vector.py <path_to_gpkg>")