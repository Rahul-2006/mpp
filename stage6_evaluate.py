"""Stage 6: evaluate every model on the SAME unseen test trajectories, make tables/plots, pick the best model."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from metrics import evaluate
from common import make_cell_id
from stage5_train import load_dataset

def main():
    W, K = C.WINDOW, C.TOP_K
    _, _, te = load_dataset()
    te = te.reset_index(drop=True)
    models = [joblib.load(p) for p in sorted(C.MODEL_DIR.glob("*.joblib"))]
    rows, errs, preds = [], {}, {}
    for m in models:
        p = m.predict_topk(te, K)
        r, e = evaluate(p, te, C.GRID_M, m.covered_)
        rows.append({"model": m.name, "type": "baseline" if m.name.startswith("Baseline") else "learned", **r})
        errs[m.name], preds[m.name] = e, p
    tbl = pd.DataFrame(rows).sort_values(["type", "exact_acc"], ascending=[True, False])
    tbl.to_csv(C.OUT_DIR / "model_comparison.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(f"[Stage 6] test cases: {len(te):,}\n", tbl.to_string(index=False))

    learned = tbl[tbl.type == "learned"].sort_values(["exact_acc", "median_err_m"], ascending=[False, True])
    best = learned.iloc[0]["model"]
    (C.OUT_DIR / "best_model.json").write_text(json.dumps({"best_model": best, "test_exact_acc": float(learned.iloc[0]["exact_acc"]),
                                                          f"test_top{K}_acc": float(learned.iloc[0][f"top{K}_acc"])}, indent=2))
    print(f"  Selected model for handoff preparation: {best}")

    # ---- plots
    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = tbl.set_index("model"); xs = np.arange(len(t))
    ax.bar(xs - 0.2, t.exact_acc * 100, 0.4, label="Top-1"); ax.bar(xs + 0.2, t[f"top{K}_acc"] * 100, 0.4, label=f"Top-{K}")
    ax.set_xticks(xs); ax.set_xticklabels(t.index, rotation=25, ha="right"); ax.set_ylabel("Accuracy (%)"); ax.legend()
    ax.set_title("Next-cell prediction accuracy (test set)"); fig.tight_layout(); fig.savefig(C.OUT_DIR / "accuracy_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, e in errs.items():
        s = np.sort(e); ax.plot(s, np.arange(1, len(s) + 1) / len(s), label=name)
    ax.set_xlim(0, 10 * C.GRID_M); ax.set_xlabel("Prediction error (m)"); ax.set_ylabel("Cumulative fraction"); ax.legend(fontsize=7)
    ax.set_title("Prediction error distribution (CDF)"); fig.tight_layout(); fig.savefig(C.OUT_DIR / "error_cdf.png", dpi=150); plt.close(fig)

    seg = te.groupby(["user_id", "trajectory_id", "segment_id"]).size().idxmax()
    mask = ((te.user_id == seg[0]) & (te.trajectory_id == seg[1]) & (te.segment_id == seg[2])).values
    sub, pb = te[mask], preds[best][mask][:, 0, :]
    ok = (pb == np.c_[sub.next_x, sub.next_y]).all(1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(sub[f"x{W}"], sub[f"y{W}"], "-o", c="grey", ms=3, label="actual path (cells)")
    ax.scatter(pb[ok, 0], pb[ok, 1], c="green", s=22, label="predicted next: correct")
    ax.scatter(pb[~ok, 0], pb[~ok, 1], c="red", marker="x", s=30, label="predicted next: wrong")
    ax.set_aspect("equal"); ax.legend(fontsize=7); ax.set_xlabel("grid X"); ax.set_ylabel("grid Y")
    ax.set_title(f"Actual vs predicted ({best})"); fig.tight_layout(); fig.savefig(C.OUT_DIR / "actual_vs_predicted.png", dpi=150); plt.close(fig)

    # ---- hand-off file for the later NS-3 stage
    p = preds[best]
    out = te[["user_id", "trajectory_id", "segment_id", "ts"]].copy()
    out["current_cell"] = te[f"c{W}"]
    out["predicted_cell"] = [make_cell_id(*r) for r in p[:, 0, :]]
    out[f"predicted_top{K}"] = [";".join(make_cell_id(*c) for c in r if c[0] != -10 ** 9) for r in p]
    out["actual_next_cell"] = te["next_cell"]
    out.to_csv(C.OUT_DIR / "predictions_test.csv", index=False)

if __name__ == "__main__":
    main()
