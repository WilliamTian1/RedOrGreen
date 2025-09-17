# Earnings Sentiment Analyzer

Disclaimer: Don't use this for actual trading! It's meant to demonstrate ML engineering practices, not produce alpha.

This project analyzes earnings-related text snippets and predicts whether a stock might go up or down the next day. It combines sentiment analysis from a finance-tuned language model with market data to make predictions.

## Features

- **FinBERT sentiment analysis** with keyword fallback for offline use
- **Stock sector mapping** via CSV lookup (30+ popular stocks)
- **Price features** from Yahoo Finance (5-day returns & volatility)
- **Simple classifier** (logistic regression or random forest)
- **MLflow experiment tracking** with metrics, plots, and model artifacts
- **Command-line interface** for single-snippet predictions with auto-bootstrap training
- **Real-time news ingestion** from Yahoo Finance, MarketWatch, SEC filings, and NewsAPI
- **Offline mode** - works without internet using sample data
- **Production-grade LoRA fine-tuning** (GPU required, auto-skips if unavailable)
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

# Quick training with limited samples (faster for testing)
python -m app.pipeline --limit_samples=10 --use_lora=false

# Try it out
python -m app.cli.analyze "AAPL: guidance raised for Q4"

# Or train with live news data (requires NEWSAPI_KEY)
export NEWSAPI_KEY=your_api_key_here
python -m app.pipeline --live_news --limit_samples=50
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

## Live News Integration

The system can fetch real-time earnings news from multiple sources instead of using static sample data:

### Supported News Sources

1. **Yahoo Finance** - Fetches recent news for specific tickers via unofficial API
2. **MarketWatch** - Scrapes earnings calendar and headlines for earnings-related content
3. **SEC Filings** - Monitors RSS feed for 8-K filings (earnings reports)
4. **NewsAPI** - Professional news aggregation (requires free API key from newsapi.org)

### Usage

```bash
# Get a free API key from newsapi.org (optional but recommended)
export NEWSAPI_KEY=your_api_key_here

# Train with live news instead of sample data
python -m app.pipeline --live_news

# Or combine with other options
python -m app.pipeline --live_news --use_lora=true --limit_samples=50
```

### How It Works

- Fetches up to 300 unique news articles across all sources
- Automatically extracts ticker symbols from headlines
- Filters for earnings-related keywords ("earnings", "revenue", "beat", "miss", etc.)
- Caches results to `app/data/live_news_cache.jsonl` for offline use
- Falls back to cached data if live fetch fails
- Deduplicates articles and handles rate limiting

### Sample Live Data

Live news creates entries like:

```json
{
  "ticker": "AAPL",
  "text": "AAPL: Apple reports record Q3 earnings - Revenue exceeded Wall Street expectations",
  "source": "Yahoo Finance",
  "timestamp": "2024-01-15T10:30:00"
}
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

This loads the trained model and applies it to new text. **Auto-bootstrap feature**: If you haven't trained a model yet, it automatically runs a quick offline training session first using the sample data. This means you can run `python -m app.cli.analyze "text"` immediately after installation without manual training - the CLI will bootstrap itself and then provide predictions.

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

## Sample Size Control

Control how many articles to train on using `--limit_samples`:

```bash
# Quick testing (5 samples, ~10 seconds)
python -m app.pipeline --limit_samples=5 --use_lora=false

# Development (10-15 samples, ~30 seconds)
python -m app.pipeline --limit_samples=10 --use_lora=false

# Production (all samples, ~60 seconds)
python -m app.pipeline --use_lora=false

# Live news with limits (20-50 samples recommended)
python -m app.pipeline --live_news --limit_samples=20
```

**Performance vs Sample Size:**

- **5 samples**: Fast testing, may overfit
- **8+ samples**: Good balance for development
- **15+ samples**: Approaching production quality
- **All samples (18)**: Best performance, longer training

Run `python demo_sample_sizes.py` to see detailed analysis of how sample size affects model performance.

## Configuration

The `config.yaml` file controls most behavior:

```yaml
seed: 42 # For reproducible results
use_lora: true # LoRA fine-tuning (needs GPU, auto-skips if unavailable)
use_wandb: false # Weights & Biases logging
yfinance_offline: false # Skip price data downloads
experiment_name: earnings-analyzer
mlflow_tracking_uri: ./mlruns
artifacts_dir: artifacts

features:
  include_price_features: true
  price_lookback_days: 5
  sector_one_hot: true

model:
  type: logistic_regression # or random_forest
  logistic_regression:
    C: 1.0
    penalty: l2
    solver: liblinear
    max_iter: 1000
  random_forest:
    n_estimators: 200
    max_depth: 6

train:
  test_size: 0.25
```

You can also use environment variables:

- `YFINANCE_OFFLINE=true` - Skip price downloads
- `FINBERT_OFFLINE=true` - Use keyword fallback instead of FinBERT
- `USE_LORA=true` - Enable LoRA fine-tuning
- `NEWSAPI_KEY=your_key` - Enable NewsAPI for live news fetching

## Offline mode

The whole system works without internet. Set `YFINANCE_OFFLINE=true` and `FINBERT_OFFLINE=true`, and it will:

- Use the bundled sample data
- Apply keyword-based sentiment analysis
- Use dummy price features (all zeros)

This is handy for demos or when you don't want to download large models.

## LoRA fine-tuning (GPU only)

The system includes production-grade LoRA (Low-Rank Adaptation) fine-tuning to adapt FinBERT specifically to your financial text patterns. This goes beyond simple demonstration with proper evaluation, hyperparameter tuning, and comprehensive logging:

```bash
pip install datasets
python -m app.pipeline --use_lora=true
```

**What happens during LoRA fine-tuning:**

- Downloads the Financial PhraseBank dataset (academic financial sentiment data)
- Applies sophisticated LoRA adapters (r=16, alpha=32) to key transformer modules
- Runs 3-epoch training with cosine annealing, gradient checkpointing, and proper evaluation
- Performs train/validation splits with comprehensive metrics tracking
- Compares baseline vs. fine-tuned performance with statistical significance
- Saves adapted model to `artifacts/lora/adapter` with full MLflow artifact logging
- Logs detailed hyperparameters, training curves, and improvement metrics

**Advanced LoRA Configuration:**

- **Target modules**: query, value, key, dense (comprehensive coverage)
- **Learning rate**: 2e-4 with cosine annealing scheduler
- **Batch size**: 16 training, 32 evaluation (optimized for GPU memory)
- **Evaluation**: Step-based with best model selection on F1 score
- **Metrics**: Accuracy, weighted F1, with improvement tracking

**Important notes:**

- Requires a CUDA-compatible GPU (automatically skips if unavailable)
- Uses proper train/validation splits for reliable evaluation
- Logs before/after performance to demonstrate actual improvement
- Production-ready with gradient checkpointing and memory optimization
- Full MLflow integration with hyperparameter and artifact tracking

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

The test suite includes three comprehensive tests:

### `test_sector_map.py`

- **Known ticker mapping**: Verifies AAPL correctly maps to "Technology" sector
- **Unknown ticker handling**: Ensures proper KeyError with descriptive message for invalid tickers

### `test_features.py`

- **Feature matrix dimensions**: Validates that feature building produces correct shape (6+ features including sentiment logits, confidence, price features, and one-hot encoded sectors)
- **Feature pipeline integrity**: Ensures the complete feature building process works end-to-end

### `test_cli.py`

- **CLI smoke test**: Runs the actual CLI command with sample input
- **Output validation**: Verifies the CLI produces expected output format with "Sector:" and "Predicted" tokens
- **Bootstrap testing**: Handles cases where training artifacts may not exist (graceful degradation)

These tests ensure core functionality works correctly and catch regressions during development.

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
