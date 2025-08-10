import argparse, numpy as np, pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from utils import save_npz, set_seed
from pathlib import Path

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    return masked.sum(1) / torch.clamp(mask.sum(1), min=1e-9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--model_name", default="dbmdz/bert-base-turkish-cased")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(42)
    df = pd.read_csv(args.input)
    texts = df[args.text_col].astype(str).tolist()

    tok = AutoTokenizer.from_pretrained(args.model_name)
    mdl = AutoModel.from_pretrained(args.model_name)
    mdl.eval()
    if torch.cuda.is_available(): mdl.cuda()

    embs = []
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch = texts[i:i+args.batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
        if torch.cuda.is_available():
            enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        embs.append(pooled.detach().cpu().numpy())

    X = np.vstack(embs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_npz(args.out, X, ids=np.arange(X.shape[0]))

if __name__ == "__main__":
    main()
