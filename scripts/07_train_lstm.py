import argparse, json, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torch, torch.nn as nn
from utils import load_npz, set_seed
from pathlib import Path

class MLP(nn.Module):
    def __init__(self, d_in, d_h=128, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_h, d_h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_h, 2)
        )
    def forward(self, x): return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--sel", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    set_seed(42)
    X,_ = load_npz(args.emb)
    idx = np.load(args.sel)
    y = np.fromfile(args.labels, dtype=np.int64)
    X = X[:, idx]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs = []
    for tr, vl in skf.split(X,y):
        model = MLP(X.shape[1])
        if torch.cuda.is_available(): model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        Xtr = torch.tensor(X[tr], dtype=torch.float32)
        Xvl = torch.tensor(X[vl], dtype=torch.float32)
        ytr = torch.tensor(y[tr], dtype=torch.long)
        yvl = torch.tensor(y[vl], dtype=torch.long)
        if torch.cuda.is_available():
            Xtr, Xvl, ytr, yvl = Xtr.cuda(), Xvl.cuda(), ytr.cuda(), yvl.cuda()
        for _ in range(args.epochs):
            model.train(); opt.zero_grad()
            pred = model(Xtr); loss = lossf(pred, ytr); loss.backward(); opt.step()
        model.eval()
        pv = model(Xvl).argmax(1).detach().cpu().numpy()
        accs.append(float(accuracy_score(y[vl], pv)))

    out = {"model":"LSTM-MLP (tabular)", "indices": idx.tolist(), "fold_acc": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w") as f: json.dump(out, f, indent=2)
    print("Saved", args.out, "mean acc", out["mean"])

if __name__ == "__main__":
    main()
