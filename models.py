import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from arch import arch_model



class VolatilityModel(ABC):
    """
    Abstract base class for volatility forecasting models.
    """
    @abstractmethod
    def fit(self, data: pd.Series | pd.DataFrame):
        """
        Train the model on historical data.
        """
        pass

    @abstractmethod
    def predict(self, horizon: int):
        """
        Forecast volatility for a given horizon.
        """
        pass


class GARCHModel(VolatilityModel):
    """
    GARCH(p, q) model using the 'arch' library.
    """
    def __init__(self, p=1, q=1, mean='Constant', vol='GARCH', dist='Normal'):
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self.model_res = None
        self.scale = 1.0

    def fit(self, returns: pd.Series):
        """
        Fits the GARCH model.
        Note: GARCH models expect returns (e.g. pct_change), not realized volatility.
        """
        
        # Rescaling returns often helps GARCH convergence
        # If mean abs return is small (<1), scale up to ~100
        if returns.abs().mean() < 1:
            self.scale = 100.0
        else:
            self.scale = 1.0

        scaled_returns = returns * self.scale

        model = arch_model(scaled_returns, p=self.p, q=self.q,
                           mean=self.mean, vol=self.vol, dist=self.dist)
        
        self.model_res = model.fit(disp='off')
        return self

    def predict(self, horizon: int = 1):
        if self.model_res is None:
            raise ValueError("Model must be fitted before predicting.")

        forecasts = self.model_res.forecast(horizon=horizon)
        # Extract variance forecast (last step)
        var_forecast = forecasts.variance.iloc[-1].values
        # The model was fit on scaled returns, so the variance forecast is for scaled returns.
        # To get the variance forecast for the original returns, we must divide by scale**2.
        var_forecast_rescaled = var_forecast / (self.scale ** 2)
        return var_forecast_rescaled


class MLVolatilityModel(VolatilityModel):
    """
    Machine Learning based volatility forecasting.
    Uses lagged values of the input (volatility) to predict future values.
    """
    def __init__(self, model: BaseEstimator = None, lags: int = 5):
        self.lags = lags
        self.model = model if model is not None else RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.last_window = None

    def fit(self, vol_data: pd.Series | pd.DataFrame):
        """
        Fits the ML model.
        Note: Expects realized volatility (or squared returns) as input.
        """
        if isinstance(vol_data, pd.DataFrame):
            # Try to find a 'volatility' or 'vol' column (case-insensitive)
            target_col = next((col for col in vol_data.columns if 'vol' in str(col).lower()), None)
            if target_col:
                vol_data = vol_data[target_col]
            elif vol_data.shape[1] == 1:
                vol_data = vol_data.iloc[:, 0]
            else:
                raise ValueError("Input is a DataFrame. Please ensure it has a 'volatility' column or is a single column.")

        # Create lagged features
        df = pd.DataFrame(vol_data.copy())
        df.columns = ['y']
        for i in range(1, self.lags + 1):
            df[f'lag_{i}'] = df['y'].shift(i)
        df.dropna(inplace=True)

        X = df.drop(columns=['y']).values
        y = df['y'].values

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        # Save last window for prediction
        self.last_window = vol_data.iloc[-self.lags:].values
        return self

    def predict(self, horizon: int = 1):
        if self.last_window is None:
            raise ValueError("Model must be fitted before predicting.")

        predictions = []
        # Prepare initial input: reverse order so index 0 is lag_1 (most recent)
        current_input = self.last_window[::-1].reshape(1, -1)

        for _ in range(horizon):
            input_scaled = self.scaler.transform(current_input)
            pred = self.model.predict(input_scaled)[0]
            predictions.append(pred)

            # Update input for next step (recursive forecasting)
            # Shift right and place new prediction at lag_1
            current_input = np.roll(current_input, 1)
            current_input[0, 0] = pred

        return np.array(predictions)
