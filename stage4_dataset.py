"""Stage 4: sliding-window training samples (previous W cells -> next cell) with grouped train/val/test split."""
import numpy as np
import pandas as pd
import config as C

KEY = ["user_id", "trajectory_id", "segment_id"]

def build_windows(seq, W):
    parts = []
    for (u, t, s), g in seq.groupby(KEY, sort=False):
        n = len(g)
        if n < W + 1:
            continue
        idx = np.arange(W - 1, n - 1)                      # index of the *current* (last input) cell
        cells, gx, gy = g["cell"].values, g["grid_x"].values, g["grid_y"].values
        d = {"user_id": u, "trajectory_id": t, "segment_id": s}
        for j in range(1, W + 1):
            k = idx - (W - j)
            d[f"c{j}"], d[f"x{j}"], d[f"y{j}"] = cells[k], gx[k], gy[k]
        for j in range(1, W):                              # only *completed* dwells -> no leakage of the future
            d[f"dwell{j}"] = g["dwell_s"].values[idx - (W - j)]
        d["next_cell"], d["next_x"], d["next_y"] = cells[idx + 1], gx[idx + 1], gy[idx + 1]
        d["ts"] = g["timestamp"].values[idx]                # time the user ENTERED the current cell
        parts.append(pd.DataFrame(d))
    return pd.concat(parts, ignore_index=True)

def assign_split(df, by, fractions, seed):
    keys = (df.user_id + "|" + df.trajectory_id) if by == "trajectory" else df.user_id
    uniq = np.array(keys.unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr, n_va = int(round(fractions[0] * n)), int(round(fractions[1] * n))
    lab = {k: "train" for k in uniq[:n_tr]}
    lab.update({k: "val" for k in uniq[n_tr:n_tr + n_va]})
    lab.update({k: "test" for k in uniq[n_tr + n_va:]})
    return keys.map(lab)

def main():
    W = C.WINDOW
    seq = pd.read_csv(C.PROC_DIR / "mobility_sequences.csv", parse_dates=["timestamp"], dtype={"user_id": str, "trajectory_id": str})
    df = build_windows(seq, W)
    df["dx"] = df.next_x - df[f"x{W}"]
    df["dy"] = df.next_y - df[f"y{W}"]
    ts = pd.to_datetime(df["ts"])
    df["hour"], df["dow"] = ts.dt.hour, ts.dt.dayofweek
    df["split"] = assign_split(df, C.SPLIT_BY, C.SPLIT, C.SEED)
    df.to_csv(C.PROC_DIR / "training_dataset.csv", index=False)
    print(f"[Stage 4] {len(df):,} samples | split by {C.SPLIT_BY}: " + ", ".join(f"{k}={v:,}" for k, v in df.split.value_counts().items()))

if __name__ == "__main__":
    main()
