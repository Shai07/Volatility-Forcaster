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
    START_DATE = start_date
    END_DATE = end_date
    TRAIN_TEST_SPLIT_DATE = train_test_split_date

    # Ensure tickers is a list for the loader
    tickers_list = [tickers] if isinstance(tickers, str) else tickers

    # 2. Data Loading and Preparation for all tickers
    print("Step 1: Loading and preparing data for all tickers...")
    loader = DataLoader(tickers=tickers_list, start_date=START_DATE, end_date=END_DATE)
    all_data = loader.run()

    if all_data.empty:
        print("Exiting due to data loading failure.")
        return

    # Loop through each ticker and run the analysis
    for ticker in tickers_list:
        print(f"\n\n{'='*20} Processing Ticker: {ticker} {'='*20}")

        # Filter data for the current ticker
        asset_data = all_data[all_data['Ticker'] == ticker].copy()

        if asset_data.empty:
            print(f"No data found for {ticker} in the loaded dataset. Skipping.")
            continue

        print(f"Data available for {ticker} from {asset_data.index.min().date()} to {asset_data.index.max().date()}.")

        # 3. Data Splitting
        train_data = asset_data[asset_data.index < TRAIN_TEST_SPLIT_DATE]
        test_data = asset_data[asset_data.index >= TRAIN_TEST_SPLIT_DATE]

        if train_data.empty or test_data.empty:
            print("Error: Not enough data for both training and testing. Adjust dates or data source. Skipping ticker.")
            continue

        print(f"Training data from {train_data.index.min().date()} to {train_data.index.max().date()}")
        print(f"Testing data from {test_data.index.min().date()} to {test_data.index.max().date()}")

        # 4. Model Initialization
        print("\nStep 2: Initializing models...")
        garch_model = GARCHModel(p=1, q=1)
        # Initialize ML models using the new model_type parameter
        rf_model = MLVolatilityModel(model_type='RF', lags=5)
        gb_model = MLVolatilityModel(model_type='GB', lags=5)

        # 5. Model Training
        print("\nStep 3: Training models...")
        print("Training GARCH model...")
        # GARCH is trained on log returns to model volatility clustering.
        garch_model.fit(train_data['log_return'])

        print("Training ML model (Random Forest)...")
        # The ML models are trained on historical realized volatility to predict future values.
        rf_model.fit(train_data['realized_vol'])
        
        print("Training ML model (Gradient Boosting)...")
        gb_model.fit(train_data['realized_vol'])

        # 6. Prediction
        horizon = len(test_data)
        if horizon == 0:
            print("No test data to predict. Skipping evaluation.")
            continue
        print(f"\nStep 4: Generating forecasts for a horizon of {horizon} days...")

        garch_predictions = garch_model.predict(horizon=horizon)
        rf_predictions = rf_model.predict(horizon=horizon)
        gb_predictions = gb_model.predict(horizon=horizon)

        # Create pandas Series for easier plotting and evaluation
        prediction_index = test_data.index
        predictions_dict = {
            'GARCH(1,1)': pd.Series(garch_predictions, index=prediction_index, name='GARCH'),
            'Random Forest (lags=5)': pd.Series(rf_predictions, index=prediction_index, name='ML_RF'),
            'Gradient Boosting (lags=5)': pd.Series(gb_predictions, index=prediction_index, name='ML_GB')
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
        'tickers': ["SPY", "TSLA", "AAPL", "AMZN", "NVDA"],
        'start_date': '2010-01-01',
        'end_date': '2023-12-28',
        'train_test_split_date': '2023-12-01'
    }
    main(config['tickers'], config['start_date'], config['end_date'], config['train_test_split_date'])