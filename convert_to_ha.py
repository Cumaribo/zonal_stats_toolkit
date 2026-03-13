from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import traceback

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "data" / "2020_1992_chg"
AREA_RASTER_PATH = SCRIPT_DIR / "data" / "esa_pixel_area_ha_md5_1dd3298a7c4d25c891a11e01868b5db6.tif"
OUTPUT_DIR = SCRIPT_DIR / "data" / "2020_1992_chg_ha"
EXCLUDE_PATTERNS = ["nature_access"]  # Add any patterns to exclude specific rasters

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_single_raster(service_path):
    print(f"Processing: {service_path.name}...")
    try:
        with rasterio.open(service_path) as src_service:
            # Create a profile for the output raster
            profile = src_service.profile.copy()
            profile.update({
                'dtype': 'float32',
                'compress': 'LZW',
                'tiled': True,
                'bigtiff': 'YES'
            })
            
            # Fix for RasterBlockError: Remove source block sizes if they are not multiples of 16
            profile.pop('blockxsize', None)
            profile.pop('blockysize', None)

            # Handle Nodata value
            src_nodata = src_service.nodata
            print(f"  -> Detected NoData value: {src_nodata}")
            if src_nodata is None:
                # If input has no nodata defined, use NaN for float output
                out_nodata = np.nan
            else:
                out_nodata = src_nodata
            
            profile['nodata'] = out_nodata

            output_path = OUTPUT_DIR / service_path.name
            
            # Remove existing file to avoid permission/overwrite issues
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception as e:
                    print(f"  [WARNING] Could not remove existing file: {e}")

            with rasterio.open(output_path, 'w', **profile) as dst:
                # Set metadata
                dst.set_band_unit(1, 'hectares')
                dst.set_band_description(1, f"{service_path.stem}_ha")

                # Open Area raster with WarpedVRT to align it to the service raster on-the-fly
                with rasterio.open(AREA_RASTER_PATH) as src_area:
                    with WarpedVRT(src_area,
                                   crs=src_service.crs,
                                   transform=src_service.transform,
                                   width=src_service.width,
                                   height=src_service.height,
                                   resampling=Resampling.nearest,
                                   nodata=0) as vrt_area:
                        
                        # Process in blocks (windows) to save memory
                        for ji, window in src_service.block_windows(1):
                            service_data = src_service.read(1, window=window)
                            area_data = vrt_area.read(1, window=window)
                            
                            # Create valid masks
                            if np.isnan(out_nodata):
                                valid_service = ~np.isnan(service_data)
                            elif np.issubdtype(service_data.dtype, np.floating):
                                valid_service = ~np.isclose(service_data, out_nodata)
                            else:
                                valid_service = (service_data != out_nodata)
                            
                            # Area must be valid and > 0
                            valid_area = (area_data > 0) & (~np.isnan(area_data))
                            
                            # Combined mask where calculation is possible
                            valid_pixels = valid_service & valid_area
                            
                            # Initialize result block with nodata
                            result_block = np.full_like(service_data, out_nodata, dtype=np.float32)
                            
                            # Perform division only on valid pixels
                            safe_area = area_data.copy()
                            safe_area[~valid_area] = 1.0 # arbitrary non-zero value
                            
                            calculated_values = service_data / safe_area
                            result_block[valid_pixels] = calculated_values[valid_pixels]
                            
                            dst.write(result_block, 1, window=window)

        print(f"  -> Saved to: {output_path}")

    except Exception as e:
        print(f"  [ERROR] Failed on {service_path.name}: {e}")
        traceback.print_exc()

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
    # Limit max_workers to prevent memory exhaustion (e.g., 4 workers)
    max_workers = 1
    print(f"Starting processing with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(process_single_raster, tasks))

if __name__ == "__main__":
    process_rasters()