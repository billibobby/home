"""
Backtest Visualization Module

Generates comprehensive visualizations for backtest results.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from pathlib import Path
import base64
import io

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from trading_bot.backtesting.results import BacktestResults


class BacktestVisualizer:
    """
    Generates visualizations for backtest results.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("matplotlib not available, visualizations disabled")
        
        self.logger.info("BacktestVisualizer initialized")
    
    def plot_equity_curve(self, results: BacktestResults,
                         save_path: Optional[str] = None) -> Optional[object]:
        """
        Plot equity curve with drawdown shading.
        
        Args:
            results: BacktestResults object
            save_path: Optional path to save figure
            
        Returns:
            matplotlib figure object or None
        """
        if not MATPLOTLIB_AVAILABLE or len(results.equity_curve) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot equity curve
        ax.plot(results.equity_curve.index, results.equity_curve.values, linewidth=2, label='Equity')
        
        # Add initial capital line
        initial_capital = results.equity_curve.iloc[0]
        ax.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Shade drawdown periods
        running_max = results.equity_curve.expanding().max()
        drawdown = (results.equity_curve - running_max) / running_max * 100
        ax.fill_between(results.equity_curve.index, results.equity_curve.values,
                       running_max.values, where=(results.equity_curve < running_max),
                       alpha=0.3, color='red', label='Drawdown')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.set_title('Equity Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            plt.close()
            return None
        
        return fig
    
    def plot_drawdown(self, results: BacktestResults,
                     save_path: Optional[str] = None) -> Optional[object]:
        """
        Plot drawdown chart (underwater plot).
        
        Args:
            results: BacktestResults object
            save_path: Optional path to save figure
            
        Returns:
            matplotlib figure object or None
        """
        if not MATPLOTLIB_AVAILABLE or len(results.equity_curve) == 0:
            return None
        
        # Calculate drawdown
        running_max = results.equity_curve.expanding().max()
        drawdown = (results.equity_curve - running_max) / running_max * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Fill area below zero
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=2)
        
        # Annotate max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax.annotate(f'Max DD: {max_dd_value:.2f}%', xy=(max_dd_idx, max_dd_value),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Chart')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            plt.close()
            return None
        
        return fig
    
    def plot_monthly_returns(self, results: BacktestResults,
                            save_path: Optional[str] = None) -> Optional[object]:
        """
        Plot monthly returns heatmap.
        
        Args:
            results: BacktestResults object
            save_path: Optional path to save figure
            
        Returns:
            matplotlib figure object or None
        """
        if not MATPLOTLIB_AVAILABLE or len(results.daily_returns) == 0:
            return None
        
        # Create monthly returns
        monthly_returns = results.daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        # Create pivot table
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        pivot = monthly_returns_df.pivot(index='year', columns='month', values='return')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot heatmap
        if SEABORN_AVAILABLE:
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                       cbar_kws={'label': 'Return (%)'}, ax=ax)
        else:
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(range(12))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            plt.colorbar(im, ax=ax, label='Return (%)')
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        ax.set_title('Monthly Returns Heatmap (%)')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            plt.close()
            return None
        
        return fig
    
    def plot_rolling_sharpe(self, results: BacktestResults, window: int = 60,
                           save_path: Optional[str] = None) -> Optional[object]:
        """
        Plot rolling Sharpe ratio.
        
        Args:
            results: BacktestResults object
            window: Rolling window in days
            save_path: Optional path to save figure
            
        Returns:
            matplotlib figure object or None
        """
        if not MATPLOTLIB_AVAILABLE or len(results.daily_returns) < window:
            return None
        
        # Calculate rolling Sharpe
        rolling_sharpe = results.daily_returns.rolling(window=window).apply(
            lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        ax.axhline(y=1.0, color='r', linestyle='--', label='Sharpe = 1.0')
        ax.axhline(y=results.sharpe_ratio, color='g', linestyle='--',
                  label=f'Overall Sharpe = {results.sharpe_ratio:.2f}')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Rolling Sharpe Ratio')
        ax.set_title(f'Rolling Sharpe Ratio (window={window} days)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            plt.close()
            return None
        
        return fig
    
    def plot_returns_distribution(self, results: BacktestResults,
                                  save_path: Optional[str] = None) -> Optional[object]:
        """
        Plot histogram of daily returns.
        
        Args:
            results: BacktestResults object
            save_path: Optional path to save figure
            
        Returns:
            matplotlib figure object or None
        """
        if not MATPLOTLIB_AVAILABLE or len(results.daily_returns) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        ax.hist(results.daily_returns.values, bins=50, alpha=0.7, edgecolor='black')
        
        # Overlay normal distribution
        mean = results.daily_returns.mean()
        std = results.daily_returns.std()
        x = np.linspace(results.daily_returns.min(), results.daily_returns.max(), 100)
        normal = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        normal = normal * len(results.daily_returns) * (x[1] - x[0])
        ax.plot(x, normal, 'r-', linewidth=2, label='Normal Distribution')
        
        # Add statistics
        skew = results.daily_returns.skew()
        kurt = results.daily_returns.kurtosis()
        ax.axvline(mean, color='g', linestyle='--', label=f'Mean: {mean:.4f}')
        
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Returns Distribution (Skew: {skew:.2f}, Kurtosis: {kurt:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            plt.close()
            return None
        
        return fig
    
    def plot_win_loss_distribution(self, results: BacktestResults,
                                   save_path: Optional[str] = None) -> Optional[object]:
        """
        Plot histogram comparing winning vs losing trades.
        
        Args:
            results: BacktestResults object
            save_path: Optional path to save figure
            
        Returns:
            matplotlib figure object or None
        """
        if not MATPLOTLIB_AVAILABLE or len(results.trades) == 0:
            return None
        
        # Extract PnL values
        pnls = [t.get('pnl', 0) for t in results.trades if t.get('pnl') is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        if len(wins) == 0 and len(losses) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        if wins:
            ax.hist(wins, bins=30, alpha=0.7, color='green', label='Wins', edgecolor='black')
            ax.axvline(results.avg_win, color='green', linestyle='--', linewidth=2,
                      label=f'Avg Win: ${results.avg_win:.2f}')
        
        if losses:
            ax.hist(losses, bins=30, alpha=0.7, color='red', label='Losses', edgecolor='black')
            ax.axvline(results.avg_loss, color='red', linestyle='--', linewidth=2,
                      label=f'Avg Loss: ${results.avg_loss:.2f}')
        
        ax.set_xlabel('PnL ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Win/Loss Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            plt.close()
            return None
        
        return fig
    
    def plot_trade_duration_distribution(self, results: BacktestResults,
                                        save_path: Optional[str] = None) -> Optional[object]:
        """
        Plot histogram of trade holding periods.
        
        Args:
            results: BacktestResults object
            save_path: Optional path to save figure
            
        Returns:
            matplotlib figure object or None
        """
        if not MATPLOTLIB_AVAILABLE or len(results.trades) == 0:
            return None
        
        # Calculate holding periods
        holding_periods = []
        for trade in results.trades:
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            if entry_time and exit_time:
                try:
                    entry_dt = pd.to_datetime(entry_time)
                    exit_dt = pd.to_datetime(exit_time)
                    holding_periods.append((exit_dt - entry_dt).days)
                except:
                    pass
        
        if len(holding_periods) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(holding_periods, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(results.avg_holding_period, color='r', linestyle='--', linewidth=2,
                  label=f'Avg: {results.avg_holding_period:.1f} days')
        
        ax.set_xlabel('Holding Period (days)')
        ax.set_ylabel('Frequency')
        ax.set_title('Trade Duration Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            plt.close()
            return None
        
        return fig
    
    def plot_cumulative_pnl(self, results: BacktestResults,
                           save_path: Optional[str] = None) -> Optional[object]:
        """
        Plot cumulative PnL over time.
        
        Args:
            results: BacktestResults object
            save_path: Optional path to save figure
            
        Returns:
            matplotlib figure object or None
        """
        if not MATPLOTLIB_AVAILABLE or len(results.trades) == 0:
            return None
        
        # Extract PnL values
        pnls = [t.get('pnl', 0) for t in results.trades if t.get('pnl') is not None]
        cumulative_pnl = np.cumsum(pnls)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative PnL ($)')
        ax.set_title('Cumulative PnL')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            plt.close()
            return None
        
        return fig
    
    def plot_feature_importance(self, results: BacktestResults, top_n: int = 20,
                               save_path: Optional[str] = None) -> Optional[object]:
        """
        Plot feature importance.
        
        Args:
            results: BacktestResults object
            top_n: Number of top features to show
            save_path: Optional path to save figure
            
        Returns:
            matplotlib figure object or None
        """
        if not MATPLOTLIB_AVAILABLE or not results.feature_importance:
            return None
        
        # Sort features by importance
        sorted_features = sorted(results.feature_importance.items(),
                               key=lambda x: x[1], reverse=True)[:top_n]
        
        if len(sorted_features) == 0:
            return None
        
        features, importances = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(features)), importances)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            plt.close()
            return None
        
        return fig
    
    def plot_all(self, results: BacktestResults, output_dir: str) -> List[str]:
        """
        Generate all plots and save to directory.
        
        Args:
            results: BacktestResults object
            output_dir: Output directory path
            
        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        # Generate all plots
        plots = [
            ('equity_curve', self.plot_equity_curve),
            ('drawdown', self.plot_drawdown),
            ('monthly_returns', self.plot_monthly_returns),
            ('rolling_sharpe', self.plot_rolling_sharpe),
            ('returns_distribution', self.plot_returns_distribution),
            ('win_loss_distribution', self.plot_win_loss_distribution),
            ('trade_duration', self.plot_trade_duration_distribution),
            ('cumulative_pnl', self.plot_cumulative_pnl),
        ]
        
        if results.feature_importance:
            plots.append(('feature_importance', self.plot_feature_importance))
        
        for plot_name, plot_func in plots:
            try:
                save_path = output_path / f"{plot_name}.png"
                plot_func(results, save_path=str(save_path))
                saved_files.append(str(save_path))
            except Exception as e:
                self.logger.error(f"Error generating {plot_name}: {str(e)}")
        
        return saved_files
    
    def generate_html_report(self, results: BacktestResults, output_path: str) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            results: BacktestResults object
            output_path: Path to save HTML file
            
        Returns:
            Path to saved HTML file
        """
        # Generate plots and convert to base64
        plot_images = {}
        
        plots_to_generate = [
            ('equity_curve', self.plot_equity_curve),
            ('drawdown', self.plot_drawdown),
            ('monthly_returns', self.plot_monthly_returns),
            ('rolling_sharpe', self.plot_rolling_sharpe),
        ]
        
        for plot_name, plot_func in plots_to_generate:
            try:
                fig = plot_func(results)
                if fig:
                    img_str = self._fig_to_base64(fig)
                    plot_images[plot_name] = img_str
                    plt.close(fig)
            except Exception as e:
                self.logger.error(f"Error generating {plot_name} for report: {str(e)}")
        
        # Generate HTML
        html = results.generate_report()
        
        # Embed plot images
        for plot_name, img_str in plot_images.items():
            html = html.replace(f'</body>',
                              f'<h2>{plot_name.replace("_", " ").title()}</h2>'
                              f'<img src="data:image/png;base64,{img_str}" />'
                              f'</body>')
        
        # Save HTML
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
    
    def _save_figure(self, fig, path: str):
        """Save figure to file."""
        fig.savefig(path, dpi=150, bbox_inches='tight')
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        buf.close()
        return img_str



