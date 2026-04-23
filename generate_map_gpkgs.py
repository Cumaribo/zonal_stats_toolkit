import os
import geopandas as gpd
import pandas as pd

def create_map_gpkgs():
    out_dir = "/home/jeronimo/data/global_ncp/processed/output_maps"
    os.makedirs(out_dir, exist_ok=True)
    
    groupings = {
        'country': {
            'vector': 'data/vector_basedata/cartographic_ee_r264_correspondence.gpkg',
            'field': 'nev_name'
        },
        'region_wb': {
            'vector': 'data/vector_basedata/cartographic_ee_r264_correspondence.gpkg',
            'field': 'region_wb'
        },
        'income_grp': {
            'vector': 'data/vector_basedata/cartographic_ee_r264_correspondence.gpkg',
            'field': 'income_grp'
        },
        'biome': {
            'vector': 'data/vector_basedata/Biome.gpkg',
            'field': 'WWF_biome'
        }
    }
    
    for grp, config in groupings.items():
        data_file = f"output_plots_diff/{grp}_map_data.csv"
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found. Skipping {grp}.")
            continue
        
        print(f"Processing {grp}...")
        df = pd.read_csv(data_file)
        
        # Pivot to wide format: One row per geometry, multiple columns for abs and pct
        df_wide = df.pivot(index=config['field'], columns='service', values=['mean_val', 'sym_pct_change'])
        df_wide.columns = [f"{col[1]}_{'abs' if col[0] == 'mean_val' else 'pct'}" for col in df_wide.columns]
        df_wide.reset_index(inplace=True)
        
        print(f"  Loading vector {config['vector']}...")
        gdf = gpd.read_file(config['vector'])
        
        print(f"  Dissolving by {config['field']}...")
        gdf_dissolved = gdf.dissolve(by=config['field']).reset_index()
        gdf_dissolved = gdf_dissolved[[config['field'], 'geometry']]
        
        print(f"  Merging data...")
        gdf_merged = gdf_dissolved.merge(df_wide, on=config['field'], how='left')
        
        out_file = os.path.join(out_dir, f"{grp}_change_map.gpkg")
        if os.path.exists(out_file):
            os.remove(out_file)
        print(f"  Saving to {out_file}...")
        gdf_merged.to_file(out_file, driver="GPKG")
        print(f"  Done with {grp}.\n")

if __name__ == "__main__":
    create_map_gpkgs()