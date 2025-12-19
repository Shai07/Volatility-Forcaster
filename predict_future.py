import pandas as pd
from datetime import date, timedelta
import numpy as np
from data_loader import DataLoader
from models import GARCHModel, MLVolatilityModel
import matplotlib.pyplot as plt



def predict_future_volatility(tickers: str | list[str], history_start_date: str, forecast_horizon: int):
    """
    Trains volatility models on historical data and predicts future volatility.

    :param tickers: A single ticker string or a list of ticker strings.
    :param history_start_date: The start date for fetching historical data (e.g., '2010-01-01').
    :param forecast_horizon: The number of future days to forecast.
    """
    # Use today as the end date for historical data
    history_end_date = date.today().strftime('%Y-%m-%d')
    
    # Ensure tickers is a list
    tickers_list = [tickers] if isinstance(tickers, str) else tickers

    print("Step 1: Loading and preparing data...")
    loader = DataLoader(tickers=tickers_list, start_date=history_start_date, end_date=history_end_date)
    all_data = loader.run()

    if all_data.empty:
        print("Exiting due to data loading failure.")
        return

    # Loop through each ticker for prediction
    for ticker in tickers_list:
        print(f"\n\n{'='*20} Predicting for Ticker: {ticker} {'='*20}")

        asset_data = all_data[all_data['Ticker'] == ticker].copy()

        if asset_data.empty:
            print(f"No data found for {ticker}. Skipping.")
            continue

        print(f"Training models on data from {asset_data.index.min().date()} to {asset_data.index.max().date()}.")

        # Initialize models
        garch_model = GARCHModel()
        rf_model = MLVolatilityModel(model_type='RF', lags=5)
        gb_model = MLVolatilityModel(model_type='GB', lags=5)

        # Train models on the full available history
        print("Training GARCH model...")
        garch_model.fit(asset_data['log_return'])

        print("Training Random Forest model...")
        rf_model.fit(asset_data['realized_vol'])
        
        print("Training Gradient Boosting model...")
        gb_model.fit(asset_data['realized_vol'])

        # Predict future volatility
        print(f"\nGenerating forecasts for the next {forecast_horizon} day(s)...")
        garch_predictions = garch_model.predict(horizon=forecast_horizon)
        rf_predictions = rf_model.predict(horizon=forecast_horizon)
        gb_predictions = gb_model.predict(horizon=forecast_horizon)

        # Convert daily variance predictions to annualized volatility percentage.
        # Volatility (%) = sqrt(variance) * sqrt(trading_days) * 100
        # We use np.maximum(0, pred) to handle potential negative predictions from ML models.
        trading_days = 252
        garch_vol_pct = np.sqrt(np.maximum(0, garch_predictions)) * np.sqrt(trading_days) * 100
        rf_vol_pct = np.sqrt(np.maximum(0, rf_predictions)) * np.sqrt(trading_days) * 100
        gb_vol_pct = np.sqrt(np.maximum(0, gb_predictions)) * np.sqrt(trading_days) * 100

        # Display the predictions
        future_dates = pd.bdate_range(start=date.today() + timedelta(days=1), periods=forecast_horizon)
        
        predictions_df = pd.DataFrame({
            'GARCH (%)': garch_vol_pct,
            'Random Forest (%)': rf_vol_pct,
            'Gradient Boosting (%)': gb_vol_pct
        }, index=future_dates)
        
        predictions_df.index.name = 'Forecast Date'

        print(f"\n--- Predicted Annualized Volatility (%) for {ticker} ---")
        print(predictions_df.to_string(float_format="%.2f"))
        print("-----------------------------------------------------\n")
        
        
        # Display predicions
        plt.figure(figsize=(12, 6))
        plt.plot(predictions_df.index, predictions_df['GARCH (%)'], label='GARCH', color='blue')
        plt.plot(predictions_df.index, predictions_df['Random Forest (%)'], label='Random Forest', color='green')
        plt.plot(predictions_df.index, predictions_df['Gradient Boosting (%)'], label='Gradient Boosting', color='red') 
        plt.title(f'Predicted Annualized Volatility (%) for {ticker}')
        plt.xlabel('Forecast Date')
        plt.ylabel('Volatility (%)')
        plt.legend()
        plt.grid(True)
        plt.show()



if __name__ == "__main__":
    predict_future_volatility(
        tickers=["SPY", "TSLA"],
        history_start_date="2010-01-01",
        forecast_horizon=10
    )