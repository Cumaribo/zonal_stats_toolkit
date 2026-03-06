from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import rioxarray
import xarray as xr

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "data" / "base_years"
AREA_RASTER_PATH = SCRIPT_DIR / "data" / "esa_pixel_area_ha_md5_1dd3298a7c4d25c891a11e01868b5db6.tif"
OUTPUT_DIR = SCRIPT_DIR / "data" / "base_years_ha"
EXCLUDE_PATTERNS = ["nature_access"]

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_single_raster(service_path):
    print(f"Processing: {service_path.name}...")
    try:
        # Load Area Template (Lazily) inside worker
        area_da = rioxarray.open_rasterio(AREA_RASTER_PATH, chunks={'x': 4096, 'y': 4096})
        
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
        print("  -> Checking alignment...")
        
        # Check CRS and Resolution for fast path
        crs_match = (area_da.rio.crs == service_da.rio.crs)
        res_match = False
        if crs_match:
            try:
                res_match = np.allclose(area_da.rio.resolution(), service_da.rio.resolution(), rtol=1e-3)
            except Exception:
                res_match = False

        if crs_match and res_match:
            print("  -> CRS and resolution match. Using fast alignment (reindex_like)...")
            # reindex_like is much faster than reproject_match as it avoids warping
            # tolerance=1e-5 handles slight floating point coordinate differences
            aligned_area = area_da.reindex_like(service_da, method="nearest", tolerance=1e-5)
        else:
            print("  -> Grid mismatch. Using robust alignment (reproject_match)...")
            try:
                clipped_area = area_da.rio.clip_box(*service_da.rio.bounds())
            except Exception:
                clipped_area = area_da
            aligned_area = clipped_area.rio.reproject_match(service_da)
        
        # 4. CALCULATION (Mass / Hectares)
        # We use .where to avoid division by zero (if area is 0, result is NaN)
        aligned_area = aligned_area.where(aligned_area > 0)
        
        # Perform calculation and inject into a copy of the original service_da
        # to preserve exact coordinates, CRS, and Transform.
        calculated = service_da / aligned_area
        result_da = service_da.copy(data=calculated.data)
        
        # Ensure CRS and Transform are explicitly set on the output
        result_da.rio.write_crs(service_da.rio.crs, inplace=True)
        result_da.rio.write_transform(service_da.rio.transform(), inplace=True)
        
        # Update Metadata (Important for GIS)
        result_da.name = f"{service_path.stem}_ha"
        result_da.attrs['units'] = 'hectares'
        
        # 5. EXPORT
        output_path = OUTPUT_DIR / service_path.name
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

def process_rasters():
    # 1. Iterate through all Service TIFs
    tif_files = list(INPUT_DIR.glob("*.tif"))
    print(f"Found {len(tif_files)} rasters to process.")

    tasks = []
    for service_path in tif_files:
        if any(pattern in service_path.name for pattern in EXCLUDE_PATTERNS):
            print(f"Skipping {service_path.name}...")
            continue
        tasks.append(service_path)

    # 2. Process in parallel
    with ProcessPoolExecutor() as executor:
        list(executor.map(process_single_raster, tasks))

if __name__ == "__main__":
    process_rasters()