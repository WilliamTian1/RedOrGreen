# Earnings Sentiment Analyzer

Disclaimer: Don't use this for actual trading! It's meant to demonstrate ML engineering practices, not produce alpha.

This project analyzes earnings-related text snippets and predicts whether a stock might go up or down the next day. It combines sentiment analysis from a finance-tuned language model with market data to make predictions.

## Features

- **FinBERT sentiment analysis** with keyword fallback for offline use
- **Stock sector mapping** via CSV lookup (30+ popular stocks)
- **Price features** from Yahoo Finance (5-day returns & volatility)
- **Simple classifier** (logistic regression or random forest)
- **MLflow experiment tracking** with metrics, plots, and model artifacts
- **Command-line interface** for single-snippet predictions
- **Offline mode** - works without internet using sample data
- **Optional LoRA fine-tuning** (GPU required, auto-skips if unavailable)
- **Reproducible** with fixed seeds and config-driven behavior
- **Extensible** with API stubs for FastAPI, Kafka/Redis, ONNX export

## How To

```bash
# Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install everything
pip install -r requirements.txt

# Train the model on the sample data
python -m app.pipeline --use_lora=false

# Try it out
python -m app.cli.analyze "AAPL: guidance raised for Q4"
```

You'll see something like:

```
Sector: Technology
Sentiment: Positive (0.91)
Predicted Next-Day Move: UP (0.67)
```

## Data

The system comes with 18 sample earnings snippets in `app/data/sample_earnings.jsonl`. Each one looks like:

```json
{
  "ticker": "AAPL",
  "text": "iPhone demand exceeded expectations in Q3",
  "label": "UP"
}
```

There's also a mapping file (`app/data/ticker_sector_map.csv`) that maps stock symbols to sectors:

```csv
ticker,company,sector
AAPL,Apple Inc.,Technology
MSFT,Microsoft Corp.,Technology
JPM,JPMorgan Chase & Co.,Financials
```

## Methodologies

### The Pipeline (`app/pipeline.py`)

This is the main training script. It:

- Loads the sample data
- Runs each text snippet through FinBERT to get sentiment
- Looks up each ticker's sector
- Downloads recent price data from Yahoo Finance (or skips if offline)
- Combines everything into a feature matrix
- Trains a simple classifier (logistic regression by default)
- Saves everything to the `artifacts/` folder
- Logs metrics to MLflow

### The CLI (`app/cli/analyze.py`)

This loads the trained model and applies it to new text. If you haven't trained a model yet, it automatically runs a quick training session first.

### FinBERT Sentiment (`app/models/finbert_infer.py`)

This wraps the HuggingFace FinBERT model. The interesting part is the fallback - if the model can't load (no internet, missing dependencies, etc.), it uses simple keyword matching:

- Positive words: "beat", "exceed", "strong", "raised", "record", "surprise", "profit"
- Negative words: "miss", "weak", "cut", "lowered", "loss", "delay", "probe"

### Feature Building (`app/features/build_features.py`)

For each snippet, this creates:

- 3 sentiment logits (negative, neutral, positive probabilities)
- 1 sentiment confidence score
- 1 five-day stock return
- 1 five-day volatility measure
- Several sector dummy variables (one-hot encoded)

Everything gets standardized before training.

## Configuration

The `config.yaml` file controls most behavior:

```yaml
seed: 42 # For reproducible results
use_lora: false # LoRA fine-tuning (needs GPU)
use_wandb: false # Weights & Biases logging
yfinance_offline: false # Skip price data downloads

features:
  include_price_features: true
  price_lookback_days: 5

model:
  type: logistic_regression # or random_forest
  logistic_regression:
    C: 1.0
    penalty: l2
```

You can also use environment variables:

- `YFINANCE_OFFLINE=true` - Skip price downloads
- `FINBERT_OFFLINE=true` - Use keyword fallback instead of FinBERT
- `USE_LORA=true` - Enable LoRA fine-tuning

## Offline mode

The whole system works without internet. Set `YFINANCE_OFFLINE=true` and `FINBERT_OFFLINE=true`, and it will:

- Use the bundled sample data
- Apply keyword-based sentiment analysis
- Use dummy price features (all zeros)

This is handy for demos or when you don't want to download large models.

## LoRA fine-tuning (GPU only)

By default, the system uses FinBERT as-is without any fine-tuning. However, if you have a GPU and want to experiment with adapting FinBERT to financial text, you can enable LoRA (Low-Rank Adaptation) fine-tuning:

```bash
pip install datasets
python -m app.pipeline --use_lora=true
```

**What happens when you enable LoRA:**

- Downloads the Financial PhraseBank dataset (academic financial sentiment data)
- Applies LoRA adapters to the pre-trained FinBERT model
- Runs a brief fine-tuning session (1 epoch, very small)
- Saves the adapted model to `artifacts/lora/adapter`
- Logs before/after metrics to MLflow

**Important notes:**

- Requires a CUDA-compatible GPU
- If no GPU is detected, it automatically skips and continues with base FinBERT
- This is mostly for demonstration - the training is minimal
- The base FinBERT model (`yiyanghkust/finbert-tone`) is already trained on financial text

You can also enable it via config:

```yaml
use_lora: true
```

Or environment variable:

```bash
USE_LORA=true python -m app.pipeline
```

## Experiment tracking

The system logs everything to MLflow automatically. After training, you can start the MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then open http://localhost:5000 to see your experiments, metrics, and saved artifacts.

## Testing

Run the tests with:

```bash
pytest -q
```

There are three simple tests:

- Check that ticker-to-sector mapping works
- Verify that feature building produces the right dimensions
- Make sure the CLI runs without crashing

## What gets saved

After training, the `artifacts/` folder contains:

- `sk_model.pkl` - The trained classifier
- `sector_encoder.pkl` - One-hot encoder for sectors
- `scaler.pkl` - Feature standardization
- `feature_schema.json` - List of feature names
- `confusion_matrix.png` - Classification results visualization
- `roc_curve.png` - ROC curve plot

## Extending the system

The code includes stubs for future enhancements:

- **FastAPI service**: `app/api/schemas.py` has Pydantic models ready for a REST API
- **Streaming**: `app/infra/streaming_stubs.py` shows how to wire in Kafka/Redis consumers
- **ONNX export**: `app/serve/export_onnx.py` is a placeholder for model optimization

The architecture makes it easy to swap in better sentiment models, add more features, or connect to live data feeds.

## Requirements

Everything runs on CPU and doesn't need much memory. The main dependencies are:

- `transformers` and `torch` for FinBERT
- `scikit-learn` for the classifier
- `yfinance` for price data
- `mlflow` for experiment tracking
- `pandas` and `numpy` for data handling

See `requirements.txt` for the full list.
