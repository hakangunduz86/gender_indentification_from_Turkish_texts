import argparse, os, json, numpy as np
from scipy.stats import wilcoxon

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory with *.json run files")
    ap.add_argument("--baseline", required=True, help="filename of baseline JSON inside --dir")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    files = [f for f in os.listdir(args.dir) if f.endswith(".json")]
    with open(os.path.join(args.dir, args.baseline), "r") as f:
        base = json.load(f)
    base_acc = np.array(base["fold_acc"])

    lines = ["# Wilcoxon signed-rank vs. {}".format(args.baseline), "", "| model | mean | std | p-value |", "|---|---:|---:|---:|"]
    for fn in files:
        if fn == args.baseline: continue
        with open(os.path.join(args.dir, fn),"r") as f:
            d = json.load(f)
        acc = np.array(d["fold_acc"])
        if len(acc) != len(base_acc): 
            continue
        stat = wilcoxon(base_acc, acc, alternative="two-sided")
        lines.append(f"| {fn} | {np.mean(acc):.3f} | {np.std(acc):.3f} | {stat.pvalue:.4f} |")
    with open(args.report, "w") as f:
        f.write("\n".join(lines))
    print("Report:", args.report)

if __name__ == "__main__":
    main()
