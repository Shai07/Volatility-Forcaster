# This will include graphs and charts to display data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluator:
    def __init__(self, actual_vol: pd.Series, predictions: dict):
        """
        Initialize the evaluator.
        
        :param actual_vol: pd.Series containing the realized volatility (ground truth).
        :param predictions: Dictionary where keys are model names and values are predictions 
                            (pd.Series or np.array).
        """
        self.actual = actual_vol
        self.predictions = predictions
        self.results_df = None

    def calculate_metrics(self):
        """
        Calculates MSE, RMSE, MAE, and R2 for each model.
        """
        metrics_list = []

        for name, preds in self.predictions.items():
            # Align data lengths
            # Assuming predictions correspond to the end of the actual data if lengths differ
            min_len = min(len(self.actual), len(preds))
            
            y_true = self.actual.iloc[-min_len:].values
            
            if isinstance(preds, pd.Series):
                y_pred = preds.iloc[-min_len:].values
            else:
                y_pred = preds[-min_len:]

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            metrics_list.append({
                'Model': name,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })

        self.results_df = pd.DataFrame(metrics_list).set_index('Model')
        return self.results_df

    def plot_forecasts(self):
        """
        Plots the actual volatility against the model forecasts.
        """
        plt.figure(figsize=(12, 6))
        
        # Plot actual
        plt.plot(self.actual.index, self.actual, label='Actual Volatility', color='black', linewidth=2, alpha=0.7)

        # Plot predictions
        for name, preds in self.predictions.items():
            min_len = min(len(self.actual), len(preds))
            # Align index
            plot_index = self.actual.index[-min_len:]
            
            if isinstance(preds, pd.Series):
                plot_data = preds.iloc[-min_len:]
            else:
                plot_data = preds[-min_len:]
                
            plt.plot(plot_index, plot_data, label=f'{name} Forecast', linestyle='--')

        plt.title('Volatility Forecast Comparison')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_residuals(self):
        """
        Plots the distribution of residuals for each model.
        """
        plt.figure(figsize=(12, 6))
        
        for name, preds in self.predictions.items():
            min_len = min(len(self.actual), len(preds))
            y_true = self.actual.iloc[-min_len:].values
            if isinstance(preds, pd.Series):
                y_pred = preds.iloc[-min_len:].values
            else:
                y_pred = preds[-min_len:]
                
            residuals = y_true - y_pred
            plt.hist(residuals, bins=30, alpha=0.5, label=f'{name} Residuals')

        plt.title('Residuals Distribution')
        plt.xlabel('Residual (Actual - Forecast)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def summary(self):
        if self.results_df is None:
            self.calculate_metrics()
        
        print("\n--- Model Evaluation Summary ---")
        print(self.results_df)
        print("--------------------------------\n")
