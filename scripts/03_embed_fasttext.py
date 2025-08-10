import argparse, numpy as np, pandas as pd
from tqdm import tqdm
from gensim.models.fasttext import load_facebook_vectors
from utils import save_npz, set_seed
from pathlib import Path

def doc_embed(model, tokens):
    vecs = [model[w] for w in tokens if w in model.key_to_index]
    if not vecs: return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--ft_path", default="embeddings/fasttext/cc.tr.300.bin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(42)
    df = pd.read_csv(args.input)
    model = load_facebook_vectors(args.ft_path)

    X = np.vstack([doc_embed(model, str(t).split()) for t in tqdm(df[args.text_col].tolist())])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_npz(args.out, X, ids=np.arange(X.shape[0]))

if __name__ == "__main__":
    main()
