"""Stage 1: load raw GeoLife .plt files, clean them and split into continuous segments."""
import json
import numpy as np
import pandas as pd
import config as C
from common import haversine_m

KEY = ["user_id", "trajectory_id"]

def load_plt(path):
    df = pd.read_csv(path, skiprows=6, header=None,
                     names=["latitude", "longitude", "zero", "altitude", "days", "date", "time"],
                     usecols=["latitude", "longitude", "date", "time"], dtype={"date": str, "time": str})
    return df

def load_raw(raw_dir, max_users=None):
    users = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if max_users:
        users = users[:max_users]
    frames = []
    for u in users:
        for f in sorted((u / "Trajectory").glob("*.plt")):
            d = load_plt(f)
            d.insert(0, "user_id", u.name)
            d.insert(1, "trajectory_id", f.stem)
            frames.append(d)
    if not frames:
        raise FileNotFoundError(f"No .plt files found under {raw_dir}")
    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["date"] + " " + raw["time"], errors="coerce")
    return raw.drop(columns=["date", "time"])

def clean_and_segment(raw):
    rep = {"raw_rows": int(len(raw))}
    df = raw.dropna(subset=["latitude", "longitude", "timestamp"])
    ok = df.latitude.between(-90, 90) & df.longitude.between(-180, 180) & ~((df.latitude == 0) & (df.longitude == 0))
    rep["invalid_removed"] = int(len(raw) - len(df) + (~ok).sum())
    df = df[ok]

    mono = df.groupby(KEY, sort=False)["timestamp"].apply(lambda s: s.is_monotonic_increasing)
    rep["trajectories_not_chronological"] = int((~mono).sum())

    n = len(df)
    df = df.drop_duplicates(KEY + ["timestamp"])
    rep["duplicate_timestamps_removed"] = int(n - len(df))
    df = df.sort_values(KEY + ["timestamp"], kind="stable").reset_index(drop=True)

    g = df.groupby(KEY, sort=False)
    dt = g["timestamp"].diff().dt.total_seconds()
    dist = haversine_m(g["latitude"].shift(), g["longitude"].shift(), df["latitude"], df["longitude"])
    speed = dist / dt * 3.6
    gap_brk = dt > C.MAX_GAP_S
    spd_brk = (speed > C.MAX_SPEED_KMH) & ~gap_brk
    brk = dt.isna() | gap_brk | spd_brk                      # first point of each trajectory also starts a segment
    df["segment_id"] = brk.astype(int).groupby([df.user_id, df.trajectory_id]).cumsum() - 1
    df["dt_s"], df["dist_m"], df["speed_kmh"] = dt, dist, speed

    rep.update(clean_rows=int(len(df)), users=int(df.user_id.nunique()),
               trajectories=int(df.groupby(KEY).ngroups),
               segments=int(df.groupby(KEY + ["segment_id"]).ngroups),
               breaks_time_gap=int(gap_brk.sum()), breaks_speed=int(spd_brk.sum()))
    return df, rep

def main(raw_dir=None, max_users=None):
    raw_dir = raw_dir or C.RAW_DIR
    max_users = max_users or C.MAX_USERS
    raw = load_raw(raw_dir, max_users)
    df, rep = clean_and_segment(raw)
    df.to_csv(C.PROC_DIR / "cleaned_gps.csv", index=False)
    (C.OUT_DIR / "stage1_report.json").write_text(json.dumps(rep, indent=2))
    print("[Stage 1]", rep)

if __name__ == "__main__":
    main()
