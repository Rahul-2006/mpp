"""Stage 3: collapse repeated cells into movement sequences and compute mobility statistics."""
import numpy as np
import pandas as pd
import config as C

KEY = ["user_id", "trajectory_id", "segment_id"]

def main():
    df = pd.read_csv(C.PROC_DIR / "gps_with_cells.csv", parse_dates=["timestamp"])
    df = df.sort_values(KEY + ["timestamp"], kind="stable")
    g = df.groupby(KEY, sort=False)
    changed = df["cell_id"] != g["cell_id"].shift()
    seq = df.loc[changed, KEY + ["timestamp", "cell_id", "grid_x", "grid_y"]].copy()

    g = seq.groupby(KEY, sort=False)
    seq["seq_idx"] = g.cumcount()
    # dwell = time between entering this cell and entering the next one (NaN for last cell of a segment).
    # NOTE: for seq_idx == 0 the true entry time is unknown (recording started inside the cell) -> partial dwell.
    seq["dwell_s"] = (g["timestamp"].shift(-1) - seq["timestamp"]).dt.total_seconds()
    seq["step_cells"] = np.maximum((seq.grid_x - g["grid_x"].shift()).abs(), (seq.grid_y - g["grid_y"].shift()).abs())
    seq.rename(columns={"cell_id": "cell"}).to_csv(C.PROC_DIR / "mobility_sequences.csv", index=False)

    # ---- statistics
    st = C.OUT_DIR / "stats"; st.mkdir(exist_ok=True)
    seq["prev_cell"] = g["cell_id"].shift()
    seq["cell_id"].value_counts().head(20).rename_axis("cell").reset_index(name="visits").to_csv(st / "top_cells.csv", index=False)
    tr = seq.dropna(subset=["prev_cell"])
    tr.groupby(["prev_cell", "cell_id"]).size().sort_values(ascending=False).head(20).reset_index(name="count") \
      .rename(columns={"prev_cell": "from_cell", "cell_id": "to_cell"}).to_csv(st / "top_transitions.csv", index=False)
    full = seq[(seq.seq_idx > 0) & seq.dwell_s.notna()]
    full.groupby("cell_id")["dwell_s"].agg(["mean", "count"]).sort_values("count", ascending=False).head(20) \
        .rename_axis("cell").reset_index().to_csv(st / "avg_dwell_per_cell.csv", index=False)
    tr.groupby("user_id").size().rename("transitions").reset_index().to_csv(st / "transitions_per_user.csv", index=False)
    s = [g["cell_id"].shift(-k) for k in (1, 2, 3)]
    m = s[2].notna()
    routes = (seq.cell_id[m] + ">" + s[0][m] + ">" + s[1][m] + ">" + s[2][m]).value_counts().head(20)
    routes.rename_axis("route_4_cells").reset_index(name="count").to_csv(st / "frequent_routes.csv", index=False)

    print(f"[Stage 3] {len(df):,} GPS points -> {len(seq):,} cell entries | {tr.shape[0]:,} transitions | "
          f"mean dwell {full.dwell_s.mean():.0f}s | jumps >1 cell: {(tr.step_cells > 1).mean():.1%}")

if __name__ == "__main__":
    main()
