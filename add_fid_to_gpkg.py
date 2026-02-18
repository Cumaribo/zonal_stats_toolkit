import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_grid_fid_column(vector_path):
    path = Path(vector_path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return

    logger.info(f"Connecting to GeoPackage: {path}")
    try:
        with sqlite3.connect(path) as conn:
            cursor = conn.cursor()
            
            # 1. Find the table name for features
            cursor.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'")
            result = cursor.fetchone()
            if not result:
                logger.error("No feature table found in GPKG.")
                return
            table_name = result[0]
            logger.info(f"Found feature table: {table_name}")

            # 2. Check if column exists
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [info[1] for info in cursor.fetchall()]
            
            if "grid_fid" in columns:
                logger.info("'grid_fid' column already exists. Skipping.")
            else:
                logger.info("Adding 'grid_fid' column...")
                # Add column
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN grid_fid INTEGER")
                
                # Populate it with the internal FID (usually 'fid' or 'id' in GPKG sqlite)
                # We use the rowid which is the implicit PK in SQLite if not aliased
                logger.info("Populating 'grid_fid' from internal ID...")
                cursor.execute(f"UPDATE {table_name} SET grid_fid = rowid")
                logger.info(f"Updated {cursor.rowcount} rows.")
                
    except Exception as e:
        logger.error(f"Error updating GPKG: {e}")

if __name__ == "__main__":
    # Path to the vector file as defined in the INI
    vector_path = "data/vector_basedata/AOOGrid_10x10km_land_4326_clean.gpkg"
    add_grid_fid_column(vector_path)