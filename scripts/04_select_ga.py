import argparse, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from utils import load_npz
import random

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

def ga_select(X, y, n_genes, pop=30, iters=40, keep=6, mut=0.1, seed=42):
    rnd = np.random.RandomState(seed)
    pop_idx = [rnd.choice(n_genes, size=rnd.randint(20, min(200, n_genes)), replace=False) for _ in range(pop)]
    for _ in range(iters):
        scored = [(fitness(X,y,idx), idx) for idx in pop_idx]
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [idx for _, idx in scored[:keep]]
        children = elites.copy()
        # crossover
        while len(children) < pop:
            a, b = random.sample(elites, 2)
            m = np.unique(np.concatenate([np.random.choice(a, len(a)//2, replace=False),
                                          np.random.choice(b, len(b)//2, replace=False)]))
            # mutation
            if rnd.rand() < mut:
                flip = rnd.choice(n_genes, size=rnd.randint(1,5), replace=False)
                m = np.unique(np.concatenate([m, flip]))
            children.append(m)
        pop_idx = children
    best = max([(fitness(X,y,idx), idx) for idx in pop_idx], key=lambda z: z[0])[1]
    return np.array(sorted(best))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--labels", required=False, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    X, _ = load_npz(args.emb)
    if args.labels:
        import numpy as np
        y = np.fromfile(args.labels, dtype=np.int64)
    else:
        raise SystemExit("Provide --labels pointing to labels.npy")

    idx = ga_select(X, y, n_genes=X.shape[1])
    np.save(args.out, idx)
    print("Saved indices:", args.out, "count:", len(idx))

if __name__ == "__main__":
    main()
