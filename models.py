"""Prediction models. All expose  predict_topk(df, k) -> int array (n, k, 2) of absolute (grid_x, grid_y)
and set .covered_ (bool array: did the model make a *learned* prediction for that row?)."""
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from common import parse_cell_id

SENT = -10 ** 9      # padding for "no k-th candidate"

def _cur(df, W):
    return np.c_[df[f"x{W}"].values, df[f"y{W}"].values]

def _pack(cur, disp_lists, k):
    """cur (n,2); disp_lists: list of lists of (dx,dy) -> (n,k,2) absolute cells"""
    out = np.full((len(cur), k, 2), SENT, dtype=np.int64)
    for i, ds in enumerate(disp_lists):
        for j, (dx, dy) in enumerate(ds[:k]):
            out[i, j] = (cur[i, 0] + dx, cur[i, 1] + dy)
    return out

# ----------------------------------------------------------------------------- baselines
class StayBaseline:
    name = "Baseline: stay in cell"
    def fit(self, df, W): self.W = W; return self
    def predict_topk(self, df, k):
        cur = _cur(df, self.W); self.covered_ = np.ones(len(df), bool)
        return _pack(cur, [[(0, 0)]] * len(df), k)

class LastDirectionBaseline:
    name = "Baseline: repeat last move"
    def fit(self, df, W): self.W = W; return self
    def predict_topk(self, df, k):
        W = self.W; cur = _cur(df, W); self.covered_ = np.ones(len(df), bool)
        d = np.c_[df[f"x{W}"] - df[f"x{W-1}"], df[f"y{W}"] - df[f"y{W-1}"]]
        return _pack(cur, [[tuple(r)] for r in d], k)

class GlobalMoveBaseline:
    name = "Baseline: most common move"
    def fit(self, df, W):
        self.W = W
        self.moves = [m for m, _ in Counter(zip(df.dx, df.dy)).most_common(10)]
        return self
    def predict_topk(self, df, k):
        cur = _cur(df, self.W); self.covered_ = np.ones(len(df), bool)
        return _pack(cur, [self.moves] * len(df), k)

# ----------------------------------------------------------------------------- Markov chain
class MarkovPredictor:
    """Order-n Markov chain over cell IDs with back-off to lower orders, then to the globally most common move."""
    def __init__(self, order=1, top_k=3):
        self.order, self.top_k = order, top_k
        self.name = f"Markov chain (order {order})"
    def _contexts(self, df, n):
        W = self.W
        return zip(*[df[f"c{W - n + 1 + i}"].values for i in range(n)])
    def fit(self, df, W):
        self.W = W
        self.table = {}
        for n in range(1, self.order + 1):
            cnt = defaultdict(Counter)
            for ctx, nxt in zip(self._contexts(df, n), df["next_cell"].values):
                cnt[ctx][nxt] += 1
            self.table[n] = {c: [parse_cell_id(cell) for cell, _ in v.most_common(self.top_k)] for c, v in cnt.items()}
        self.fallback = [m for m, _ in Counter(zip(df.dx, df.dy)).most_common(self.top_k)]
        return self
    def predict_topk(self, df, k):
        k = min(k, self.top_k)
        cur = _cur(df, self.W)
        n_rows = len(df)
        out = np.full((n_rows, k, 2), SENT, dtype=np.int64)
        covered = np.zeros(n_rows, bool)
        ctxs = {n: list(self._contexts(df, n)) for n in range(1, self.order + 1)}
        for i in range(n_rows):
            for n in range(self.order, 0, -1):
                hit = self.table[n].get(ctxs[n][i])
                if hit:
                    for j, (gx, gy) in enumerate(hit[:k]):
                        out[i, j] = (gx, gy)
                    covered[i] = True
                    break
            else:
                for j, (dx, dy) in enumerate(self.fallback[:k]):
                    out[i, j] = (cur[i, 0] + dx, cur[i, 1] + dy)
        self.covered_ = covered
        return out

# ----------------------------------------------------------------------------- Random Forest
def feature_matrix(df, W):
    xc, yc = df[f"x{W}"], df[f"y{W}"]
    f = {"x_cur": xc, "y_cur": yc}
    for j in range(1, W):
        f[f"rel_x{j}"] = df[f"x{j}"] - xc
        f[f"rel_y{j}"] = df[f"y{j}"] - yc
        f[f"dwell{j}"] = df[f"dwell{j}"]
    f["prev_dx"], f["prev_dy"] = xc - df[f"x{W-1}"], yc - df[f"y{W-1}"]
    f["hour"], f["dow"] = df["hour"], df["dow"]
    return pd.DataFrame(f).fillna(0)

class RFPredictor:
    """Random Forest that classifies the NEXT MOVE (dx,dy in grid cells); predicted cell = current cell + move.
    Predicting a displacement (not an absolute cell ID) lets it generalise to cells never seen in training."""
    name = "Random Forest"
    def __init__(self, min_count=3, max_rows=300_000, seed=42, **rf_params):
        self.min_count, self.max_rows, self.seed, self.params = min_count, max_rows, seed, rf_params
    def fit(self, df, W):
        self.W = W
        lab = df.dx.astype(str) + "," + df.dy.astype(str)
        d = df[lab.map(lab.value_counts()) >= self.min_count]       # drop extremely rare jumps from TRAINING only
        if len(d) > self.max_rows:
            d = d.sample(self.max_rows, random_state=self.seed)
        self.rf = RandomForestClassifier(n_jobs=-1, random_state=self.seed, **self.params)
        self.rf.fit(feature_matrix(d, W), lab.loc[d.index])
        self.moves = np.array([[int(a) for a in c.split(",")] for c in self.rf.classes_])
        return self
    def predict_topk(self, df, k):
        P = self.rf.predict_proba(feature_matrix(df, self.W))
        kk = min(k, P.shape[1])
        top = np.argsort(-P, axis=1)[:, :kk]
        out = np.full((len(df), k, 2), SENT, dtype=np.int64)
        out[:, :kk, :] = _cur(df, self.W)[:, None, :] + self.moves[top]
        self.covered_ = np.ones(len(df), bool)
        return out
