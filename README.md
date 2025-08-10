# gender_indentification_from_Turkish_texts
Implementation and resources for the manuscript “Scalable Gender Profiling from Turkish News Text Using Deep Embeddings and Meta-Heuristic Feature Selection”.


# Turkish Gender Profiling – Reproducible Pipeline

This repo provides a **clean, modular baseline** you can run end‑to‑end and extend from your Colab notebook.
It mirrors the notebook’s LSTM approach but adds:
- Reproducible scripts (CLI)
- BERTurk & FastText embeddings
- Meta‑heuristic feature selection (GA/Jaya/ARO placeholders + scikit baselines)
- LSTM & GBM training
- 10‑fold CV + Wilcoxon signed‑rank tests
- Config‑driven runs

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/01_prepare_data.py --input data/raw.csv --text_col text --label_col label --out data/clean.csv

# Embeddings
python scripts/02_embed_berturk.py --input data/clean.csv --out data/bert_emb.npz
python scripts/03_embed_fasttext.py --input data/clean.csv --out data/ft_emb.npz

# Feature selection (choose one)
python scripts/04_select_ga.py --emb data/bert_emb.npz --labels data/labels.npy --out data/bert_ga_idx.npy
python scripts/05_select_jaya.py --emb data/bert_emb.npz --labels data/labels.npy --out data/bert_jaya_idx.npy
python scripts/06_select_aro.py --emb data/bert_emb.npz --labels data/labels.npy --out data/bert_aro_idx.npy

# Train/evaluate
python scripts/07_train_lstm.py --emb data/bert_emb.npz --sel data/bert_ga_idx.npy --labels data/labels.npy --out runs/lstm_bert_ga.json
python scripts/08_train_gbm.py  --emb data/bert_emb.npz --sel data/bert_ga_idx.npy --labels data/labels.npy --out runs/gbm_bert_ga.json

# Statistics
python scripts/09_eval_stats.py --dir runs --baseline lstm_bert_ga.json --report runs/stats_report.md
```

## Data format

Provide a CSV: `data/raw.csv` with columns:
- `text` – Turkish text
- `label` – {0,1} (male/female or as appropriate)

Change column names via CLI flags.

## Notes
- BERTurk: `dbmdz/bert-base-turkish-cased`
- FastText: `cc.tr.300.bin` (put under `embeddings/fasttext/` or pass `--ft_path`)
- Meta‑heuristics include simple, well‑commented reference implementations.
- All randomness is seeded for reproducibility.
