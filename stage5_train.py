"""Stage 5: train Random Forest (with small validation grid search), Markov chains (orders 1-3) and baselines."""
import joblib
import pandas as pd
import config as C
from models import RFPredictor, MarkovPredictor, StayBaseline, LastDirectionBaseline, GlobalMoveBaseline
from metrics import evaluate

def load_dataset():
    df = pd.read_csv(C.PROC_DIR / "training_dataset.csv", dtype={"user_id": str, "trajectory_id": str})
    return df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]

def main():
    W = C.WINDOW
    tr, va, te = load_dataset()
    print(f"[Stage 5] train={len(tr):,} val={len(va):,} test={len(te):,}")

    rows, best, best_acc = [], None, -1
    for p in C.RF_GRID:
        m = RFPredictor(C.MIN_LABEL_COUNT, C.MAX_TRAIN_ROWS, C.SEED, **p).fit(tr, W)
        r, _ = evaluate(m.predict_topk(va, C.TOP_K), va, C.GRID_M)
        rows.append({**p, "val_exact_acc": r["exact_acc"], f"val_top{C.TOP_K}_acc": r[f"top{C.TOP_K}_acc"]})
        if r["exact_acc"] > best_acc:
            best, best_acc = m, r["exact_acc"]
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "rf_tuning.csv", index=False)
    print(f"  RF best validation accuracy: {best_acc:.3f}")
    joblib.dump(best, C.MODEL_DIR / "random_forest.joblib")

    for n in C.MARKOV_ORDERS:
        m = MarkovPredictor(n, C.TOP_K).fit(tr, W)
        joblib.dump(m, C.MODEL_DIR / f"markov_order{n}.joblib")
        r, _ = evaluate(m.predict_topk(va, C.TOP_K), va, C.GRID_M)
        print(f"  {m.name}: validation accuracy {r['exact_acc']:.3f}")

    for b in (StayBaseline(), LastDirectionBaseline(), GlobalMoveBaseline()):
        b.fit(tr, W)
        joblib.dump(b, C.MODEL_DIR / (b.name.split(": ")[1].replace(" ", "_") + "_baseline.joblib"))

if __name__ == "__main__":
    main()
