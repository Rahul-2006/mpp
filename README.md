# Mobility Pattern Prediction – Proof of Concept (Stages 1-7 groundwork)

## Run
```
pip install -r requirements.txt
# real data: extract Microsoft GeoLife so that  <path>/Data/000/Trajectory/*.plt  exists
python run_pipeline.py --raw "<path>/Geolife Trajectories 1.3/Data" --max-users 20
# smoke test without the dataset (synthetic, results are NOT meaningful):
python run_pipeline.py --synthetic
```
Tune everything in `config.py` (grid size, thresholds, window, split, RF grid). Start with `--max-users 20`, then scale up.

## Stage -> file -> output
| Stage | Script | Output |
|---|---|---|
| 1 Clean + segment GPS | stage1_preprocess.py | data/processed/cleaned_gps.csv, outputs/stage1_report.json |
| 2 Virtual 200 m cells | stage2_cells.py | data/processed/gps_with_cells.csv |
| 3 Cell sequences + stats | stage3_sequences.py | data/processed/mobility_sequences.csv, outputs/stats/*.csv |
| 4 Sliding-window dataset | stage4_dataset.py | data/processed/training_dataset.csv (has `split` column) |
| 5 Train models | stage5_train.py | models/*.joblib, outputs/rf_tuning.csv |
| 6 Evaluate + select | stage6_evaluate.py | outputs/model_comparison.csv, 3 PNG plots, best_model.json, predictions_test.csv |
| 7 Network design | (design doc, to be done with NS-3) | predictions_test.csv is the planned Python -> NS-3 interface |

## Design decisions worth knowing
- **Segmentation, not deletion**: a gap > 30 min or implied speed > 300 km/h starts a new segment; no transition is ever learned across a break. Points on both sides are kept.
- **Cell IDs** come from a local equirectangular projection floored to 200 m, so the IDs differ from the example `CELL_4557_650` in the brief (that depends on the projection/origin used). Only consistency matters.
- **RF predicts the next *move*** (dx, dy in cells), not an absolute cell ID, so it can handle cells never seen in training. Rare moves (< 3 occurrences) are dropped from training only, never from testing.
- **Markov chain** uses back-off: order n -> ... -> order 1 -> most common global move. `coverage` = share of test rows answered by a learned context (not the fallback).
- **Leakage guard**: only dwell times of *completed* cells are used as features; the current cell's dwell is unknown at prediction time. Hour/day = time of entering the current cell.
- **Split is grouped** by trajectory (or by user via `SPLIT_BY="user"`, which tests unseen people and will score much lower). Same test rows for every model.
- **"Stay in cell" scores 0 %** by construction (consecutive duplicate cells are collapsed). It is kept only as a distance-error reference. "Most frequent next cell from current cell" is exactly Markov order 1, so it isn't duplicated.
- **Dwell of the first cell in a segment is partial** (recording began mid-cell).
- Cells are virtual: GeoLife has no real base-station or handover data.
