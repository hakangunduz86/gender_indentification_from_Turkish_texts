import argparse, json, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from utils import load_npz
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--sel", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    X,_ = load_npz(args.emb)
    idx = np.load(args.sel)
    y = np.fromfile(args.labels, dtype=np.int64)
    X = X[:, idx]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs = []
    for tr, vl in skf.split(X,y):
        clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3)
        clf.fit(X[tr], y[tr])
        pv = clf.predict(X[vl])
        from sklearn.metrics import accuracy_score
        accs.append(float(accuracy_score(y[vl], pv)))

    out = {"model":"GBM", "indices": idx.tolist(), "fold_acc": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w") as f: json.dump(out, f, indent=2)
    print("Saved", args.out, "mean acc", out["mean"])

if __name__ == "__main__":
    main()
