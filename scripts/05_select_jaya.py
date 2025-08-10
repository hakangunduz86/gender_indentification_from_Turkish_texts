import argparse, numpy as np
from utils import load_npz
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression

def fitness(X, y, idx):
    if len(idx)==0: return -1e9
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, vl in skf.split(X, y):
        clf = LogisticRegression(max_iter=200, n_jobs=None)
        clf.fit(X[tr][:, idx], y[tr])
        p = clf.predict(X[vl][:, idx])
        scores.append(matthews_corrcoef(y[vl], p))
    return float(np.mean(scores))

def jaya_select(X, y, n_iter=60, seed=42):
    rng = np.random.RandomState(seed)
    n = X.shape[1]
    # Start from random subset
    sol = rng.choice(n, size=min(150, n), replace=False)
    best = sol.copy(); best_fit = fitness(X,y,best)
    worst = sol.copy(); worst_fit = best_fit
    for t in range(n_iter):
        cand = np.unique(rng.choice(n, size=min(150, n), replace=False))
        f = fitness(X,y,cand)
        if f > best_fit: best, best_fit = cand, f
        if f < worst_fit: worst, worst_fit = cand, f
        # Jaya update (very simplified for reference baseline)
    return np.array(sorted(best))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    X,_ = load_npz(args.emb)
    y = np.fromfile(args.labels, dtype=np.int64)
    idx = jaya_select(X,y)
    np.save(args.out, idx)
    print("Saved indices:", args.out, "count:", len(idx))

if __name__ == "__main__":
    main()
