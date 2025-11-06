"""
Backtest Results Container

Stores and serializes backtest results.
"""

import json
import pickle
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass
class BacktestResults:
    """
    Container for backtest results.
    """
    # Summary Metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_period: float = 0.0
    expectancy: float = 0.0
    
    # Time Series Data
    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    trades: List[Dict] = field(default_factory=list)
    positions: List[Dict] = field(default_factory=list)
    
    # Period Breakdown
    period_results: List[Dict] = field(default_factory=list)
    num_periods: int = 0
    
    # Model Performance
    prediction_accuracy: Optional[float] = None
    prediction_rmse: Optional[float] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    backtest_start_date: str = ""
    backtest_end_date: str = ""
    symbol: str = ""
    model_type: str = ""
    config_snapshot: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        result_dict = asdict(self)
        
        # Convert pandas Series to lists
        if isinstance(result_dict.get('equity_curve'), pd.Series):
            equity_data = {
                'values': self.equity_curve.values.tolist(),
                'index': [str(idx) for idx in self.equity_curve.index]
            }
            result_dict['equity_curve'] = equity_data
        
        if isinstance(result_dict.get('daily_returns'), pd.Series):
            returns_data = {
                'values': self.daily_returns.values.tolist(),
                'index': [str(idx) for idx in self.daily_returns.index]
            }
            result_dict['daily_returns'] = returns_data
        
        # Convert numpy types to native Python types
        for key, value in result_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                result_dict[key] = float(value)
            elif isinstance(value, np.ndarray):
                result_dict[key] = value.tolist()
        
        return result_dict
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert summary metrics to DataFrame.
        
        Returns:
            DataFrame with one row per metric
        """
        metrics = {
            'Total Return (%)': self.total_return,
            'Annualized Return (%)': self.annualized_return,
            'Sharpe Ratio': self.sharpe_ratio,
            'Sortino Ratio': self.sortino_ratio,
            'Calmar Ratio': self.calmar_ratio,
            'Max Drawdown (%)': self.max_drawdown,
            'Max Drawdown Duration (days)': self.max_drawdown_duration,
            'Win Rate (%)': self.win_rate,
            'Profit Factor': self.profit_factor,
            'Total Trades': self.total_trades,
            'Winning Trades': self.winning_trades,
            'Losing Trades': self.losing_trades,
            'Avg Win': self.avg_win,
            'Avg Loss': self.avg_loss,
            'Largest Win': self.largest_win,
            'Largest Loss': self.largest_loss,
            'Avg Holding Period (days)': self.avg_holding_period,
            'Expectancy': self.expectancy
        }
        
        df = pd.DataFrame([metrics]).T
        df.columns = ['Value']
        
        return df
    
    def save_to_file(self, path: str, format: str = 'json'):
        """
        Save results to file.
        
        Args:
            path: File path
            format: 'json', 'csv', or 'pickle'
        """
        filepath = Path(path)
        
        if format == 'json' or filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        
        elif format == 'csv' or filepath.suffix == '.csv':
            # Save summary metrics and trades
            summary_df = self.to_dataframe()
            summary_df.to_csv(filepath)
            
            # Save trades to separate file if available
            if self.trades:
                trades_file = filepath.parent / f"{filepath.stem}_trades.csv"
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv(trades_file, index=False)
        
        elif format == 'pickle' or filepath.suffix == '.pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load_from_file(cls, path: str) -> 'BacktestResults':
        """
        Load results from file.
        
        Args:
            path: File path
            
        Returns:
            BacktestResults instance
        """
        filepath = Path(path)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct pandas Series
            if 'equity_curve' in data and isinstance(data['equity_curve'], dict):
                equity_data = data['equity_curve']
                data['equity_curve'] = pd.Series(
                    equity_data['values'],
                    index=pd.to_datetime(equity_data['index'])
                )
            
            if 'daily_returns' in data and isinstance(data['daily_returns'], dict):
                returns_data = data['daily_returns']
                data['daily_returns'] = pd.Series(
                    returns_data['values'],
                    index=pd.to_datetime(returns_data['index'])
                )
            
            return cls(**data)
        
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot equity curve."""
        try:
            import matplotlib.pyplot as plt
            
            if len(self.equity_curve) == 0:
                return
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.equity_curve.index, self.equity_curve.values)
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value')
            ax.set_title('Equity Curve')
            ax.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
        
        except ImportError:
            pass
    
    def plot_drawdown(self, save_path: Optional[str] = None):
        """Plot drawdown chart."""
        try:
            import matplotlib.pyplot as plt
            
            if len(self.equity_curve) == 0:
                return
            
            # Calculate drawdown
            running_max = self.equity_curve.expanding().max()
            drawdown = (self.equity_curve - running_max) / running_max * 100
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            ax.plot(drawdown.index, drawdown.values, color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.set_title('Drawdown Chart')
            ax.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
        
        except ImportError:
            pass
    
    def plot_monthly_returns(self, save_path: Optional[str] = None):
        """Plot monthly returns heatmap."""
        try:
            import matplotlib.pyplot as plt
            
            if len(self.daily_returns) == 0:
                return
            
            # Create monthly returns
            monthly_returns = self.daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            
            # Create pivot table (years x months)
            monthly_returns.index = pd.to_datetime(monthly_returns.index)
            monthly_returns_df = pd.DataFrame({
                'year': monthly_returns.index.year,
                'month': monthly_returns.index.month,
                'return': monthly_returns.values
            })
            
            pivot = monthly_returns_df.pivot(index='year', columns='month', values='return')
            
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(range(12))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
            ax.set_title('Monthly Returns Heatmap (%)')
            plt.colorbar(im, ax=ax)
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
        
        except ImportError:
            pass
    
    def plot_rolling_sharpe(self, window: int = 60, save_path: Optional[str] = None):
        """Plot rolling Sharpe ratio."""
        try:
            import matplotlib.pyplot as plt
            
            if len(self.daily_returns) < window:
                return
            
            # Calculate rolling Sharpe
            rolling_sharpe = self.daily_returns.rolling(window=window).apply(
                lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0
            )
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(rolling_sharpe.index, rolling_sharpe.values)
            ax.axhline(y=1.0, color='r', linestyle='--', label='Sharpe = 1.0')
            ax.set_xlabel('Date')
            ax.set_ylabel('Rolling Sharpe Ratio')
            ax.set_title(f'Rolling Sharpe Ratio (window={window} days)')
            ax.legend()
            ax.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
        
        except ImportError:
            pass
    
    def generate_report(self) -> str:
        """
        Generate HTML report.
        
        Returns:
            HTML string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Results - {self.symbol}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Backtest Results</h1>
            <h2>Summary</h2>
            <p><strong>Symbol:</strong> {self.symbol}</p>
            <p><strong>Period:</strong> {self.backtest_start_date} to {self.backtest_end_date}</p>
            <p><strong>Model:</strong> {self.model_type}</p>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Return</td><td>{self.total_return:.2f}%</td></tr>
                <tr><td>Annualized Return</td><td>{self.annualized_return:.2f}%</td></tr>
                <tr><td>Sharpe Ratio</td><td>{self.sharpe_ratio:.2f}</td></tr>
                <tr><td>Sortino Ratio</td><td>{self.sortino_ratio:.2f}</td></tr>
                <tr><td>Max Drawdown</td><td>{self.max_drawdown:.2f}%</td></tr>
                <tr><td>Win Rate</td><td>{self.win_rate:.2f}%</td></tr>
                <tr><td>Profit Factor</td><td>{self.profit_factor:.2f}</td></tr>
            </table>
            
            <h2>Trade Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{self.total_trades}</td></tr>
                <tr><td>Winning Trades</td><td>{self.winning_trades}</td></tr>
                <tr><td>Losing Trades</td><td>{self.losing_trades}</td></tr>
                <tr><td>Average Win</td><td>${self.avg_win:.2f}</td></tr>
                <tr><td>Average Loss</td><td>${self.avg_loss:.2f}</td></tr>
            </table>
        </body>
        </html>
        """
        
        return html
    
    def print_summary(self):
        """Print formatted summary to console."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        print(f"Symbol: {self.symbol}")
        print(f"Period: {self.backtest_start_date} to {self.backtest_end_date}")
        print(f"Model: {self.model_type}")
        print("\nPerformance Metrics:")
        print(f"  Total Return:         {self.total_return:.2f}%")
        print(f"  Annualized Return:    {self.annualized_return:.2f}%")
        print(f"  Sharpe Ratio:         {self.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:        {self.sortino_ratio:.2f}")
        print(f"  Max Drawdown:         {self.max_drawdown:.2f}%")
        print(f"  Win Rate:             {self.win_rate:.2f}%")
        print(f"  Profit Factor:        {self.profit_factor:.2f}")
        print("\nTrade Statistics:")
        print(f"  Total Trades:         {self.total_trades}")
        print(f"  Winning Trades:       {self.winning_trades}")
        print(f"  Losing Trades:        {self.losing_trades}")
        print(f"  Avg Win:              ${self.avg_win:.2f}")
        print(f"  Avg Loss:             ${self.avg_loss:.2f}")
        print("="*60 + "\n")



