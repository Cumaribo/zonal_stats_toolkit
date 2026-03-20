import pandas as pd

def main():
    file_old = "output_access_grid/grid_10km_20260310_194649.csv"
    file_new = "output_access_grid/grid_10km_20260313_183732.csv"

    print(f"Loading Old CSV: {file_old}")
    df_old = pd.read_csv(file_old)

    print(f"Loading New CSV: {file_new}")
    df_new = pd.read_csv(file_new)

    if df_old.equals(df_new):
        print("\n🚨 BAD NEWS: The files are exactly identical. The sorting fix didn't change the output.")
    else:
        print("\n✅ GOOD NEWS: The files are different! The row order was successfully changed.")
        
        # Let's verify that ONLY the row order changed by sorting the old one
        if 'fid' in df_old.columns and 'fid' in df_new.columns:
            df_old_sorted = df_old.sort_values('fid').reset_index(drop=True)
            df_new_sorted = df_new.sort_values('fid').reset_index(drop=True)
            
            if df_old_sorted.equals(df_new_sorted):
                print("   -> Confirmed: The data values are identical, ONLY the row sorting is different.")
            else:
                print("   -> Note: The actual data values also differ between the two runs.")
                
        exact_matches = (df_old == df_new).all(axis=1).sum()
        print(f"   -> Exact row-by-row identical lines: {exact_matches} out of {len(df_old)}")

if __name__ == "__main__":
    main()