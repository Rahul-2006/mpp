import numpy as np
import config as C

R_EARTH = 6371008.8

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (vectorised)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))

def project_m(lat, lon, ref_lat=None):
    """Local equirectangular projection to metres (fine at city scale)."""
    ref_lat = C.REF_LAT if ref_lat is None else ref_lat
    x = R_EARTH * np.radians(lon) * np.cos(np.radians(ref_lat))
    y = R_EARTH * np.radians(lat)
    return x, y

def make_cell_id(gx, gy):
    return f"CELL_{int(gx)}_{int(gy)}"

def parse_cell_id(cell):
    _, gx, gy = cell.split("_")
    return int(gx), int(gy)
