import pandas as pd
from data_loader import DataLoader
from models import GARCHModel, MLVolatilityModel
from evaluations import ModelEvaluator
from sklearn.ensemble import RandomForestRegressor


def main(tickers: str | list[str], start_date: str, end_date: str, train_test_split_date: str):
    """
    Main function to run the volatility forecasting pipeline.
    """
    # 1. Configuration
    TICKER = tickers  # Example: S&P 500 ETF
    START_DATE = start_date
    END_DATE = end_date
    TRAIN_TEST_SPLIT_DATE = train_test_split_date

    # 2. Data Loading and Preparation
    print("Step 1: Loading and preparing data...")
    loader = DataLoader(tickers=TICKER, start_date=START_DATE, end_date=END_DATE)
    data = loader.run()

    if data.empty:
        print("Exiting due to data loading failure.")
        return

    # For this example, we'll focus on a single ticker.
    # The DataLoader is built to handle multiple, but for modeling we simplify.
    asset_data = data[data['Ticker'] == TICKER].copy()
    print(f"Data loaded for {TICKER} from {asset_data.index.min().date()} to {asset_data.index.max().date()}.")

    # 3. Data Splitting
    train_data = asset_data[asset_data.index < TRAIN_TEST_SPLIT_DATE]
    test_data = asset_data[asset_data.index >= TRAIN_TEST_SPLIT_DATE]

    if train_data.empty or test_data.empty:
        print("Error: Not enough data for both training and testing. Adjust dates or data source.")
        return

    print(f"Training data from {train_data.index.min().date()} to {train_data.index.max().date()}")
    print(f"Testing data from {test_data.index.min().date()} to {test_data.index.max().date()}")

    # 4. Model Initialization
    print("\nStep 2: Initializing models...")
    garch_model = GARCHModel(p=1, q=1)
    ml_model = MLVolatilityModel(
        model=RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
        lags=5
    )

    # 5. Model Training
    print("\nStep 3: Training models...")
    print("Training GARCH model...")
    # GARCH is trained on log returns to model volatility clustering.
    garch_model.fit(train_data['log_return'])

    print("Training ML model (Random Forest)...")
    # The ML model is trained on historical realized volatility to predict future values.
    ml_model.fit(train_data['realized_vol'])

    # 6. Prediction
    horizon = len(test_data)
    print(f"\nStep 4: Generating forecasts for a horizon of {horizon} days...")

    garch_predictions = garch_model.predict(horizon=horizon)
    ml_predictions = ml_model.predict(horizon=horizon)

    # Create pandas Series for easier plotting and evaluation
    prediction_index = test_data.index
    predictions_dict = {
        'GARCH(1,1)': pd.Series(garch_predictions, index=prediction_index, name='GARCH'),
        'Random Forest (lags=5)': pd.Series(ml_predictions, index=prediction_index, name='ML_RF')
    }

    # 7. Evaluation
    print("\nStep 5: Evaluating models...")
    actual_vol = test_data['realized_vol']
    evaluator = ModelEvaluator(actual_vol=actual_vol, predictions=predictions_dict)

    evaluator.summary()
    evaluator.plot_forecasts()
    evaluator.plot_residuals()

    print("\n--- Volatility Forecasting Pipeline Finished ---")


if __name__ == "__main__":
    config = {
        'tickers': '^GSPC',
        'start_date': '2000-01-01',
        'end_date': '2023-01-01',
        'train_test_split_date': '2020-01-01'
    }
    main(config['tickers'], config['start_date'], config['end_date'], config['train_test_split_date'])