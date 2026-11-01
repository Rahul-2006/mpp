"""Creates a SMALL synthetic dataset in the exact GeoLife folder/.plt format, ONLY to test the pipeline
when the real dataset is not available. Results on this data are meaningless for the project."""
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

LAT0, LON0, STEP = 39.85, 116.30, 500.0
M_LAT = 111195.0
M_LON = 111195.0 * np.cos(np.radians(39.9))

def route(a, b, h_first):
    """Manhattan path between two street-grid nodes; h_first = move along x before y."""
    (i, j), (i2, j2) = [(int(v) for v in a), (int(v) for v in b)]
    pts = [(i, j)]
    def go_x(i, j):
        while i != i2: i += int(np.sign(i2 - i)); pts.append((i, j))
        return i
    def go_y(i, j):
        while j != j2: j += int(np.sign(j2 - j)); pts.append((i, j))
        return j
    if h_first: i = go_x(i, j); j = go_y(i, j)
    else: j = go_y(i, j); i = go_x(i, j)
    return np.array(pts, float) * STEP


def trip_points(poly, speed, rng, dt=5):
    seg = np.hypot(*np.diff(poly, axis=0).T); cum = np.r_[0, np.cumsum(seg)]
    s = np.arange(0, cum[-1], speed * dt)
    x, y = np.interp(s, cum, poly[:, 0]), np.interp(s, cum, poly[:, 1])
    return x + rng.normal(0, 5, len(s)), y + rng.normal(0, 5, len(s))

def main(out_dir, n_users=8, n_trips=30, seed=7):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    for u in range(n_users):
        home = tuple(rng.integers(0, 6, 2)); work = (home[0] + rng.integers(5, 10), home[1] + rng.integers(4, 9))
        other = (home[0] + rng.integers(-4, 5), home[1] + rng.integers(4, 12))
        pref_h = rng.random() < 0.5
        d = out_dir / f"{u:03d}" / "Trajectory"; d.mkdir(parents=True, exist_ok=True)
        for t in range(n_trips):
            kind = rng.integers(0, 4)
            a, b, hr = [(home, work, 8), (work, home, 17), (home, other, 19), (other, home, 21)][kind]
            h_first = pref_h if rng.random() < 0.85 else not pref_h
            x, y = trip_points(route(a, b, h_first), rng.uniform(6, 14), rng)
            start = datetime(2009, 3, 1) + timedelta(days=int(t // 2 + 3 * u), hours=hr, minutes=int(rng.integers(0, 50)), seconds=int(rng.integers(0, 59)))
            ts = np.array([start + timedelta(seconds=5 * k) for k in range(len(x))])
            lat, lon = LAT0 + y / M_LAT, LON0 + x / M_LON
            if rng.random() < 0.15 and len(x) > 20:                     # recording gap
                k = rng.integers(10, len(x) - 10); ts[k:] += timedelta(minutes=45)
            if rng.random() < 0.10 and len(x) > 20:                     # GPS spike
                lat[rng.integers(5, len(x) - 5)] += 0.5
            lines = ["Geolife trajectory", "WGS 84", "Altitude is in Feet", "Reserved 3", "0,2,255,My Track,0,0,2,8421376", "0"]
            for k in range(len(x)):
                lines.append(f"{lat[k]:.6f},{lon[k]:.6f},0,100,0,{ts[k]:%Y-%m-%d},{ts[k]:%H:%M:%S}")
                if rng.random() < 0.02: lines.append(lines[-1])             # duplicate record
            (d / f"{start:%Y%m%d%H%M%S}.plt").write_text("\n".join(lines) + "\n")
    print("synthetic GeoLife-format data written to", out_dir)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw_synthetic/Data")
