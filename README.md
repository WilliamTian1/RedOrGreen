# Sector-Aware Earnings Analyzer (CLI)

End-to-end, production-lean command-line project combining finance-domain transformers (FinBERT), simple market features, and a supervised classifier to predict next-day stock moves from brief earnings snippets.

## Features

- Transformers: FinBERT sentiment (label, confidence, logits) with offline heuristic fallback
- Optional LoRA/PEFT fine-tune (skips gracefully without GPU)
- Ingestion: parse "TICKER: text", static ticker→sector map (CSV), minimal `yfinance` history
- Features: sector one-hot, sentiment logits + score, 5-day log return, 5-day realized vol
- Models: Logistic Regression (default) or Random Forest (fast, small)
- Tracking: MLflow experiment, params/metrics/artifacts; optional W&B mirroring
- CLI: low-latency analyzer for single snippets
- Reproducibility: seeds, config-driven, offline sample data included
- Stubs: FastAPI schemas, Kafka/Redis consumers, ONNX export placeholder

## Repository layout

```
.
├─ app/
│  ├─ cli/
│  │  └─ analyze.py              # CLI: sentiment + features → predict
│  ├─ data/
│  │  ├─ sample_earnings.jsonl   # tiny offline dataset
│  │  └─ ticker_sector_map.csv   # ticker, company, sector mapping
│  ├─ features/
│  │  └─ build_features.py       # sentiment logits + sector one-hot + price stats
│  ├─ ingest/
│  │  ├─ load_text.py            # parse "TICKER: text"; load jsonl
│  │  └─ market_data.py          # yfinance helpers (offline fallback)
│  ├─ models/
│  │  ├─ finbert_infer.py        # HF pipeline wrapper w/ offline heuristic
│  │  ├─ train_classifier.py     # sklearn model train/eval, MLflow
│  │  ├─ lora_finetune.py        # optional LoRA (GPU), graceful skip
│  │  └─ persist.py              # save/load model + preprocessors + schema
│  ├─ pipeline.py                # orchestrates end-to-end run
│  ├─ serve/
│  │  └─ export_onnx.py          # ONNX export stub
│  ├─ infra/
│  │  └─ streaming_stubs.py      # Kafka/Redis consumer stubs
│  ├─ api/
│  │  └─ schemas.py              # Pydantic models for future FastAPI
│  └─ utils/
│     ├─ config.py               # config/env + seed setup
│     ├─ logging.py              # structured logging via rich
│     └─ metrics.py              # plots (confusion, ROC) + metric helpers
├─ artifacts/                    # saved models/plots/schemas
├─ tests/
│  ├─ test_cli.py
│  ├─ test_sector_map.py
│  └─ test_features.py
├─ config.yaml                   # hyperparams, flags, paths
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## Quickstart

```bash
# 1) Create and activate a virtualenv
python -m venv venv
source venv/bin/activate      # Windows (PowerShell): .\venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train (CPU-friendly; small sample)
python -m app.pipeline --use_lora=false --limit_samples=0

# 4) Analyze a single snippet
python -m app.cli.analyze "AAPL: guidance raised for Q4"
```

Example output:

```
Sector: Technology
Sentiment: Positive (0.91)
Predicted Next-Day Move: UP (0.67)
```

## Architecture

```
[ingest] -> [FinBERT sentiment] -> [features: sector + price] -> [classifier]
                               \-> [MLflow params/metrics/artifacts]
                                         \
                                          -> [CLI inference]
```

## Configuration & environment

`config.yaml` (excerpt):

```yaml
seed: 42
use_lora: false
use_wandb: false
experiment_name: earnings-analyzer
mlflow_tracking_uri: ./mlruns
artifacts_dir: artifacts
features:
  include_price_features: true
  price_lookback_days: 5
  sector_one_hot: true
model:
  type: logistic_regression
  logistic_regression:
    C: 1.0
    penalty: l2
    solver: liblinear
    max_iter: 1000
  random_forest:
    n_estimators: 200
    max_depth: 6
```

Environment variables (override config):

- `USE_LORA=true|false`
- `USE_WANDB=true|false`
- `MLFLOW_TRACKING_URI=./mlruns` (or another local/remote URI)
- `YFINANCE_OFFLINE=true` (forces dummy price features)
- `TRANSFORMERS_OFFLINE=1` and/or `FINBERT_OFFLINE=true` (skip model downloads; use heuristic)

## Offline modes

- If network is unavailable, pipeline/CLI runs with:
  - Sample data from `app/data/sample_earnings.jsonl`
  - Heuristic sentiment (when FinBERT model is not available)
  - Dummy price features if `yfinance` fails or `YFINANCE_OFFLINE=true`
- The CLI auto-bootstraps training if artifacts are missing (runs a quick local train, then predicts).

## Modeling

- Baseline: LogisticRegression (liblinear) on standardized features
- Alternate: RandomForestClassifier (limited trees/depth for speed)
- Train/test split with fixed seed; metrics: accuracy, F1, ROC-AUC

## Experiment tracking

- MLflow experiment: `earnings-analyzer` under `./mlruns`
- Logged:
  - Params: model type, LR/RF hyperparams, use_lora, feature set
  - Metrics: `accuracy`, `f1`, `roc_auc`
  - Artifacts: `artifacts/confusion_matrix.png`, `artifacts/roc_curve.png`, optional `feature_importance.json`

Start MLflow UI locally:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then open `http://localhost:5000`.

## LoRA fine-tuning (optional)

- Requires GPU. If not available, it skips automatically.

```bash
pip install datasets
python -m app.pipeline --use_lora=true
```

This performs a tiny, demonstrative LoRA run on a small finance dataset and logs to MLflow.

## Tests

```bash
pytest -q
```

Includes:

- Sector map unit test
- Feature builder unit test
- CLI smoke test

## Artifacts

After training, `artifacts/` contains:

- `sk_model.pkl` — trained sklearn classifier
- `sector_encoder.pkl`, `scaler.pkl` — preprocessors
- `feature_schema.json`, `label_encoder.json`
- `confusion_matrix.png`, `roc_curve.png`

## Notes & ethics

This is a demo for educational purposes only. Not financial advice. Small samples and simplified features. Real deployments need robust data cleaning, more features, and thorough validation.

## Future work

- FastAPI microservice with Pydantic request/response models
- Kafka/Redis consumers for real-time feeds
- ONNX export and accelerated inference
- Richer microstructure features and cross-asset signals

## License

See `LICENSE`.
