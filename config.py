"""Central configuration. Change values here, not inside the stage scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "Geolife Trajectories 1.3" / "Data"   # contains 000/, 001/, ...
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

# ---- Stage 1: cleaning / segmentation
MAX_SPEED_KMH = 300          # implied speed above this -> start a new segment
MAX_GAP_S = 30 * 60          # recording gap above this -> start a new segment
MAX_USERS = None             # e.g. 20 to run on a subset of GeoLife users

# ---- Stage 2: virtual cells
GRID_M = 200                 # virtual cell size in metres
REF_LAT = 39.9               # reference latitude for the local projection (Beijing)

# ---- Stage 4: dataset
WINDOW = 3                   # number of previous cells used as input
SPLIT = (0.70, 0.15, 0.15)   # train / val / test
SPLIT_BY = "trajectory"      # "trajectory" or "user" (user = tests generalisation to unseen people)
SEED = 42

# ---- Stage 5/6: models
TOP_K = 3
MIN_LABEL_COUNT = 3          # RF: drop training rows whose next-move label is rarer than this
MAX_TRAIN_ROWS = 300_000     # RF: subsample for speed on the full dataset
RF_GRID = [dict(n_estimators=100, max_depth=None, min_samples_leaf=1),
           dict(n_estimators=100, max_depth=None, min_samples_leaf=3),
           dict(n_estimators=100, max_depth=20, min_samples_leaf=1),
           dict(n_estimators=100, max_depth=20, min_samples_leaf=3)]
MARKOV_ORDERS = [1, 2, 3]

for _d in (PROC_DIR, OUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
