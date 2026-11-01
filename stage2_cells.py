"""Stage 2: map every GPS point to a virtual GRID_M x GRID_M wireless cell."""
import numpy as np
import pandas as pd
import config as C
from common import project_m

def main():
    df = pd.read_csv(C.PROC_DIR / "cleaned_gps.csv", parse_dates=["timestamp"])
    x, y = project_m(df.latitude.values, df.longitude.values)
    df["grid_x"] = np.floor(x / C.GRID_M).astype(np.int64)
    df["grid_y"] = np.floor(y / C.GRID_M).astype(np.int64)
    df["cell_id"] = "CELL_" + df.grid_x.astype(str) + "_" + df.grid_y.astype(str)
    cols = ["user_id", "trajectory_id", "segment_id", "timestamp", "latitude", "longitude", "cell_id", "grid_x", "grid_y"]
    df[cols].to_csv(C.PROC_DIR / "gps_with_cells.csv", index=False)
    print(f"[Stage 2] {len(df):,} points -> {df.cell_id.nunique():,} distinct virtual cells ({C.GRID_M} m grid)")

if __name__ == "__main__":
    main()
