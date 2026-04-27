# Volatility Forecaster

A Python project for forecasting financial market volatility using:

- A statistical model (`GARCHModel`)
- A machine learning model (`MLVolatilityModel`)

The repository is currently in an early build stage: core modeling logic is implemented, while data loading, backtesting, and evaluation modules are scaffolded for extension.

## Project Status

Current state:

- Implemented:
  - Abstract volatility model interface (`VolatilityModel`)
  - GARCH forecasting model via `arch`
  - ML forecasting model using lagged features + `RandomForestRegressor`
- Scaffolded / placeholder:
  - `data_loader.py`
  - `backtester.py`
  - `evaluations.py`
  - `main.py` orchestration

## Repository Structure

```text
Volatility-Forcaster/
├── main.py          # Entry point (currently placeholder)
├── models.py        # Implemented volatility forecasting models
├── data_loader.py   # Data loading scaffold
├── backtester.py    # Backtesting scaffold
├── evaluations.py   # Visualization/evaluation scaffold
└── testing.ipynb    # Notebook experiments
```

## How It Works

The intended pipeline is:

1. Load price data.
2. Convert prices to returns and/or realized volatility.
3. Fit one or more forecasting models.
4. Generate out-of-sample forecasts.
5. Evaluate forecast quality and visualize results.
6. Backtest strategy logic using forecast signals.

At present, step 3 and step 4 are implemented in `models.py`.

### Implemented Models

#### `GARCHModel`

- Uses `arch.arch_model` with configurable `p`, `q`, `mean`, `vol`, and `dist`.
- Expects **returns** as input (not realized volatility).
- Rescales small returns internally to improve optimization stability.
- Predicts volatility by forecasting variance and taking square root.

#### `MLVolatilityModel`

- Supervised learning model with lag-based autoregressive features.
- Default estimator: `RandomForestRegressor`.
- Expects realized volatility (or similar volatility series) as target.
- Supports recursive multi-step forecasting.

## Installation

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

No `requirements.txt` is currently included, so install directly:

```bash
pip install numpy pandas scikit-learn arch yfinance matplotlib
```

## Running the Project

`main.py` is currently a placeholder, so there is no full pipeline command yet.

You can still run and test the implemented models directly from a Python session or script.

### Example: GARCH Volatility Forecast

```python
import yfinance as yf
from models import GARCHModel

# Download sample price data
df = yf.download("SPY", period="2y", auto_adjust=True)

# Compute daily returns
returns = df["Close"].pct_change().dropna()

# Fit + forecast
model = GARCHModel(p=1, q=1).fit(returns)
forecast = model.predict(horizon=5)

print("5-step volatility forecast:", forecast)
```

### Example: ML Volatility Forecast

```python
import yfinance as yf
from models import MLVolatilityModel

df = yf.download("SPY", period="2y", auto_adjust=True)
returns = df["Close"].pct_change().dropna()

# Example realized volatility proxy: rolling std of returns
realized_vol = returns.rolling(20).std().dropna()

model = MLVolatilityModel(lags=5).fit(realized_vol)
forecast = model.predict(horizon=5)

print("5-step volatility forecast:", forecast)
```

## Module Notes

### `models.py`

Primary implemented module with forecasting models and interfaces.

### `data_loader.py`

Contains `DataLoader` skeleton. Intended role:

- Pull market data (e.g., via `yfinance`)
- Standardize schema and preprocessing
- Return clean data for modeling/backtesting

### `backtester.py`

Stub for future simulation logic. Intended role:

- Accept forecasted volatility series/signals
- Simulate strategy rules
- Output performance metrics

### `evaluations.py`

Currently imports `matplotlib`. Intended role:

- Plot forecasts vs realized volatility
- Produce error metrics and diagnostic charts

## Development Notes

- Python 3.10+ is recommended.
- Model classes follow a common `fit` / `predict` interface, which makes adding new models straightforward.
- If you add full pipeline execution, `main.py` should orchestrate:
  - Data loading
  - Feature engineering
  - Training
  - Forecasting
  - Evaluation/backtesting

## Suggested Next Steps

1. Implement `DataLoader.__init__` and a `load()` method.
2. Add a full CLI/pipeline in `main.py`.
3. Add forecast evaluation metrics (MAE/RMSE, QLIKE, etc.).
4. Implement rolling-window backtests.

## Disclaimer

This project is for research and educational use. Forecasts are uncertain and should not be treated as financial advice.

