
import os
import glob

files_to_delete = glob.glob("output/*_2026_01_13_14_49_09.csv")
for f in files_to_delete:
    try:
        os.remove(f)
        print(f"Deleted {f}")
    except OSError as e:
        print(f"Error deleting {f}: {e}")
