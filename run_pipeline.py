"""Runs Stages 1-6 end to end.
   python run_pipeline.py --raw "/path/to/Geolife Trajectories 1.3/Data" [--max-users 20]
   python run_pipeline.py --synthetic            (pipeline smoke test only)"""
import argparse, time
from pathlib import Path
import config as C
import make_synthetic_geolife, stage1_preprocess, stage2_cells, stage3_sequences, stage4_dataset, stage5_train, stage6_evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
ap.add_argument("--max-users", type=int, default=C.MAX_USERS)
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()

raw = a.raw
if a.synthetic:
    raw = C.ROOT / "data" / "raw_synthetic" / "Data"
    make_synthetic_geolife.main(raw)

t0 = time.time()
stage1_preprocess.main(raw, a.max_users)
stage2_cells.main()
stage3_sequences.main()
stage4_dataset.main()
stage5_train.main()
stage6_evaluate.main()
print(f"\nAll stages finished in {time.time() - t0:.0f}s. Outputs: {C.PROC_DIR}, {C.OUT_DIR}, {C.MODEL_DIR}")
