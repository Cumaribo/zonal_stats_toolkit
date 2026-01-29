from pathlib import Path
import numpy as np
import rioxarray
import xarray as xr

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "data" / "2020_1992_chg"
AREA_RASTER_PATH = SCRIPT_DIR / "data" / "esa_pixel_area_ha_md5_1dd3298a7c4d25c891a11e01868b5db6.tif"
OUTPUT_DIR = SCRIPT_DIR / "data" / "ch_ha"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_rasters():
    # 1. Load the Global Area Raster (Lazily)
    print(f"Loading Area Template: {AREA_RASTER_PATH}")
    try:
        # Open with chunks to avoid memory crash on the massive global file
        area_da = rioxarray.open_rasterio(AREA_RASTER_PATH, chunks={'x': 4096, 'y': 4096})
    except Exception as e:
        print(f"Error loading area raster: {e}")
        return

    # 2. Iterate through all Service TIFs
    tif_files = list(INPUT_DIR.glob("*.tif"))
    print(f"Found {len(tif_files)} rasters to process.")

    for service_path in tif_files:
        print(f"Processing: {service_path.name}...")
        
        try:
            # Load Service Raster
            service_da = rioxarray.open_rasterio(service_path, chunks={'x': 4096, 'y': 4096})
            
            # Clean up duplicate coordinates if they exist, which can corrupt a file
            if len(np.unique(service_da.x.values)) < len(service_da.x.values):
                print("  -> Found and removing duplicate x coordinates...")
                _, index = np.unique(service_da['x'], return_index=True)
                service_da = service_da.isel(x=index)
            if len(np.unique(service_da.y.values)) < len(service_da.y.values):
                print("  -> Found and removing duplicate y coordinates...")
                _, index = np.unique(service_da['y'], return_index=True)
                service_da = service_da.isel(y=index)

            # 3. ALIGNMENT (The "Crop and Snap")
            try:
                # FAST PATH: If grids are already aligned, just select the area.
                # This is the most efficient operation.
                print("  -> Checking for perfect alignment (fast path)...")
                aligned_area = area_da.sel(x=service_da.x, y=service_da.y, method="nearest")
                # Check if the selection is valid
                if aligned_area.x.size == 0 or aligned_area.y.size == 0:
                    raise ValueError("Selection resulted in an empty raster.")
                print("  -> Success! Rasters are already aligned.")

            except (KeyError, ValueError):
                # ROBUST FALLBACK: If not aligned, perform the clip-then-reproject.
                print("  -> Not perfectly aligned. Using robust fallback method...")
                
                # To save memory, we always clip the large global raster to the target's bounds first.
                clipped_area = area_da.rio.clip_box(*service_da.rio.bounds())

                # Check if we need to do a full reprojection or just align the grid.
                if service_da.rio.crs == clipped_area.rio.crs:
                    print("  -> CRSs match. Aligning grid...")
                else:
                    print(f"  -> CRSs differ. Reprojecting from {clipped_area.rio.crs} to {service_da.rio.crs}...")

                # reproject_match handles both cases efficiently on the pre-clipped raster.
                aligned_area = clipped_area.rio.reproject_match(service_da)
            
            # 4. CALCULATION (Mass / Hectares)
            # We use .where to avoid division by zero (if area is 0, result is NaN)
            aligned_area = aligned_area.where(aligned_area > 0)
            result_da = service_da / aligned_area
            
            # Update Metadata (Important for GIS)
            result_da.name = f"{service_path.stem}_per_ha"
            result_da.attrs['units'] = 'units_per_hectare'
            
            # 5. EXPORT
            output_path = OUTPUT_DIR / f"{service_path.stem}_ha.tif"
            # If the output file already exists, try to remove it first.
            # This handles cases where a previous run (e.g., as root) created
            # a protected file that the current user cannot overwrite.
            if output_path.exists():
                print(f"  -> Attempting to remove existing output file: {output_path}")
                try:
                    output_path.unlink()
                except Exception as e:
                    print(f"  [WARNING] Could not remove existing file: {e}. Writing may fail due to permissions.")
            result_da.rio.to_raster(
                output_path,
                tiled=True,
                compress='LZW',
                windowed=True # Low memory write
            )
            print(f"  -> Saved to: {output_path}")

        except Exception as e:
            print(f"  [ERROR] Failed on {service_path.name}: {e}")

if __name__ == "__main__":
    process_rasters()