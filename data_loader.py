import pandas as pd
import numpy as np
import yfinance as yf


class DataLoader:
    def __init__(self, tickers: str | list[str], start_date: str, end_date: str):
        if isinstance(tickers, str):
            self.tickers = [tickers]
        else:
            self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
    
    def load_data(self) -> pd.DataFrame:
        """Downloads historical market data using yfinance."""
        # auto_adjust=True provides adjusted prices and handles splits/dividends.
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=True)
        # For a single ticker, yfinance might not return a MultiIndex.
        # We ensure columns are a MultiIndex for consistent processing.
        if len(self.tickers) == 1 and not data.empty and not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_product([data.columns, self.tickers])
        return data
    
    @staticmethod
    def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """Cleans the raw price data."""
        # We expect multi-index columns: ('Open', 'TICKER'), ('Close', 'TICKER'), etc.
        # Let's stack the tickers to handle them individually.
        df = data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker'])
        
        # Forward-fill missing values, a common approach for price data
        df = df.groupby('Ticker').ffill()
        
        # Drop any remaining NaNs, likely at the start of the series
        df = df.dropna()
        df = df.drop_duplicates()
        
        return df.reset_index().set_index('Date')

    @staticmethod
    def _prepare_data_for_vol_forecast(data: pd.DataFrame) -> pd.DataFrame:
        """Engineers features required for volatility forecasting."""
        # Calculate log returns
        data['log_return'] = data.groupby('Ticker')['Close'].transform(lambda x: np.log(x / x.shift(1)))

        data['realized_vol'] = 0.5 * np.log(data['High'] / data['Low'])**2 - \
                               (2 * np.log(2) - 1) * np.log(data['Close'] / data['Open'])**2
        
        # The target is to forecast next day's volatility
        data['target_vol'] = data.groupby('Ticker')['realized_vol'].shift(-1)

        # Clean up NaNs created by shifts and calculations
        data = data.dropna()
        return data
    
    def run(self) -> pd.DataFrame:
        """Executes the full data loading, cleaning, and preparation pipeline."""
        price_data = self.load_data()
        if price_data.empty:
            print(f"No data loaded for tickers {self.tickers}. Please check ticker symbols and date range.")
            return pd.DataFrame()
        cleaned_data = self._clean_data(price_data)
        prepared_data = self._prepare_data_for_vol_forecast(cleaned_data)
        return prepared_data
    