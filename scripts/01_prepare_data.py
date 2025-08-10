import pandas as pd, argparse, re
from pathlib import Path
from utils import ensure_dir

def basic_clean(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df[args.text_col] = df[args.text_col].astype(str).map(basic_clean)
    df = df.dropna(subset=[args.text_col, args.label_col]).reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    # also persist labels separately for downstream
    (Path(args.out).parent / "labels.npy").write_bytes(df[args.label_col].to_numpy().astype(int).tobytes())

if __name__ == "__main__":
    main()
