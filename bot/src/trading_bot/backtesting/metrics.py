"""
Performance Metrics Module

Calculates comprehensive performance metrics for backtesting results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import stats


class PerformanceMetrics:
    """
    Calculates performance metrics for trading strategies.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the performance metrics calculator.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Risk-free rate (default 2% annual)
        self.risk_free_rate = config.get('backtesting.risk_free_rate', 0.02)
        
        self.logger.info("PerformanceMetrics initialized")
    
    def calculate_all_metrics(self, trades: List[Dict], equity_curve: pd.Series,
                             daily_returns: pd.Series) -> Dict:
        """
        Calculate all performance metrics.
        
        Args:
            trades: List of trade dictionaries
            equity_curve: Equity curve over time
            daily_returns: Daily returns series
            
        Returns:
            Dictionary with all performance metrics
        """
        if len(trades) == 0:
            self.logger.warning("No trades provided, returning zero metrics")
            return self._get_zero_metrics()
        
        # Calculate returns
        returns_dict = self.calculate_returns(trades, equity_curve)
        
        # Calculate risk-adjusted metrics
        sharpe = self.calculate_sharpe_ratio(daily_returns, self.risk_free_rate)
        sortino = self.calculate_sortino_ratio(daily_returns, self.risk_free_rate)
        
        # Calculate drawdown
        drawdown_dict = self.calculate_max_drawdown(equity_curve)
        
        # Calculate Calmar ratio
        calmar = self.calculate_calmar_ratio(
            returns_dict['annualized_return'], 
            drawdown_dict['max_drawdown_pct']
        )
        
        # Calculate trade statistics
        trade_stats = self.calculate_trade_statistics(trades)
        
        # Calculate consecutive stats
        consecutive_stats = self.calculate_consecutive_stats(trades)
        
        # Calculate recovery factor
        net_profit = returns_dict['total_return_pct'] * equity_curve.iloc[0] / 100 if len(equity_curve) > 0 else 0
        recovery_factor = self.calculate_recovery_factor(net_profit, drawdown_dict['max_drawdown_pct'])
        
        # Rolling Sharpe
        rolling_sharpe = self.calculate_rolling_sharpe(daily_returns, window=60)
        
        # Statistical tests
        statistical_tests = self.run_statistical_tests(daily_returns)
        
        # Combine all metrics
        metrics = {
            **returns_dict,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            **drawdown_dict,
            **trade_stats,
            **consecutive_stats,
            'recovery_factor': recovery_factor,
            'rolling_sharpe': rolling_sharpe,
            'statistical_tests': statistical_tests
        }
        
        return metrics
    
    def calculate_returns(self, trades: List[Dict], equity_curve: pd.Series) -> Dict:
        """
        Calculate return metrics.
        
        Args:
            trades: List of trades
            equity_curve: Equity curve
            
        Returns:
            Dictionary with return metrics
        """
        if len(equity_curve) == 0:
            return {
                'total_return_pct': 0.0,
                'annualized_return_pct': 0.0,
                'cumulative_return': 1.0
            }
        
        initial_equity = equity_curve.iloc[0]
        final_equity = equity_curve.iloc[-1]
        
        total_return_pct = ((final_equity - initial_equity) / initial_equity) * 100
        
        # Annualized return
        num_days = len(equity_curve)
        annualized_return_pct = self._annualize_return(total_return_pct, num_days)
        
        cumulative_return = final_equity / initial_equity
        
        return {
            'total_return_pct': total_return_pct,
            'annualized_return_pct': annualized_return_pct,
            'cumulative_return': cumulative_return
        }
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            returns: Daily returns series
            risk_free_rate: Risk-free rate (default: from config)
            
        Returns:
            Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if len(returns) == 0:
            return 0.0
        
        # Daily risk-free rate
        daily_rf = risk_free_rate / 252
        
        # Mean excess return
        mean_excess = returns.mean() - daily_rf
        
        # Standard deviation
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe
        sharpe = (mean_excess / std_return) * np.sqrt(252)
        
        return sharpe
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted return).
        
        Args:
            returns: Daily returns series
            risk_free_rate: Risk-free rate
            
        Returns:
            Sortino ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if len(returns) == 0:
            return 0.0
        
        # Daily risk-free rate
        daily_rf = risk_free_rate / 252
        
        # Mean excess return
        mean_excess = returns.mean() - daily_rf
        
        # Downside deviation
        downside_std = self._calculate_downside_deviation(returns)
        
        if downside_std == 0:
            return 0.0
        
        # Annualized Sortino
        sortino = (mean_excess / downside_std) * np.sqrt(252)
        
        return sortino
    
    def calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).
        
        Args:
            total_return: Annualized return percentage
            max_drawdown: Maximum drawdown percentage
            
        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return float('inf') if total_return > 0 else 0.0
        
        calmar = abs(total_return / max_drawdown)
        
        return calmar
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Dict:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Equity curve over time
            
        Returns:
            Dictionary with drawdown metrics
        """
        if len(equity_curve) == 0:
            return {
                'max_drawdown_pct': 0.0,
                'max_drawdown_duration_days': 0,
                'drawdown_start_date': None,
                'drawdown_end_date': None
            }
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max * 100
        
        # Maximum drawdown
        max_drawdown_pct = drawdown.min()
        
        # Find drawdown period
        max_dd_idx = drawdown.idxmin()
        
        # Find start of drawdown (last time equity was at peak)
        if max_dd_idx in equity_curve.index:
            peak_idx = equity_curve.loc[:max_dd_idx].idxmax()
            drawdown_start_date = str(peak_idx) if peak_idx is not None else None
            drawdown_end_date = str(max_dd_idx)
            
            # Calculate duration
            if peak_idx is not None and max_dd_idx in equity_curve.index:
                duration = (pd.to_datetime(max_dd_idx) - pd.to_datetime(peak_idx)).days
            else:
                duration = 0
        else:
            drawdown_start_date = None
            drawdown_end_date = None
            duration = 0
        
        return {
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_duration_days': duration,
            'drawdown_start_date': drawdown_start_date,
            'drawdown_end_date': drawdown_end_date
        }
    
    def calculate_trade_statistics(self, trades: List[Dict]) -> Dict:
        """
        Calculate trade statistics.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with trade statistics
        """
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate_pct': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'avg_holding_period_days': 0.0
            }
        
        # Extract PnL values
        pnls = [trade.get('pnl', 0) for trade in trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        # Basic counts
        total_trades = len(trades)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        win_rate_pct = (winning_count / total_trades * 100) if total_trades > 0 else 0.0
        
        # Averages
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        # Extremes
        largest_win = max(winning_trades) if winning_trades else 0.0
        largest_loss = min(losing_trades) if losing_trades else 0.0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
        
        # Expectancy
        expectancy = (win_rate_pct / 100 * avg_win) - ((100 - win_rate_pct) / 100 * abs(avg_loss))
        
        # Average holding period
        holding_periods = []
        for trade in trades:
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            if entry_time and exit_time:
                try:
                    entry_dt = pd.to_datetime(entry_time)
                    exit_dt = pd.to_datetime(exit_time)
                    holding_periods.append((exit_dt - entry_dt).days)
                except:
                    pass
        
        avg_holding_period_days = np.mean(holding_periods) if holding_periods else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'win_rate_pct': win_rate_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_holding_period_days': avg_holding_period_days
        }
    
    def calculate_consecutive_stats(self, trades: List[Dict]) -> Dict:
        """
        Calculate consecutive win/loss statistics.
        
        Args:
            trades: List of trades
            
        Returns:
            Dictionary with consecutive stats
        """
        if len(trades) == 0:
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'current_streak': 0
            }
        
        # Extract PnL
        pnls = [trade.get('pnl', 0) for trade in trades]
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                # Break streak on break-even
                current_wins = 0
                current_losses = 0
        
        # Current streak
        if pnls:
            last_pnl = pnls[-1]
            if last_pnl > 0:
                current_streak = current_wins
            elif last_pnl < 0:
                current_streak = -current_losses
            else:
                current_streak = 0
        else:
            current_streak = 0
        
        return {
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'current_streak': current_streak
        }
    
    def calculate_recovery_factor(self, net_profit: float, max_drawdown: float) -> float:
        """
        Calculate recovery factor.
        
        Args:
            net_profit: Net profit
            max_drawdown: Maximum drawdown percentage
            
        Returns:
            Recovery factor
        """
        if max_drawdown == 0:
            return float('inf') if net_profit > 0 else 0.0
        
        recovery_factor = net_profit / abs(max_drawdown)
        
        return recovery_factor
    
    def calculate_rolling_sharpe(self, returns: pd.Series, window: int = 60) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            returns: Daily returns series
            window: Rolling window in days
            
        Returns:
            Rolling Sharpe ratio series
        """
        if len(returns) < window:
            return pd.Series(dtype=float)
        
        rolling_sharpe = returns.rolling(window=window).apply(
            lambda x: self.calculate_sharpe_ratio(x, self.risk_free_rate)
        )
        
        return rolling_sharpe
    
    def run_statistical_tests(self, returns: pd.Series) -> Dict:
        """
        Run statistical tests on returns.
        
        Args:
            returns: Daily returns series
            
        Returns:
            Dictionary with test results
        """
        if len(returns) == 0:
            return {
                't_test': {'statistic': 0.0, 'pvalue': 1.0},
                'jarque_bera': {'statistic': 0.0, 'pvalue': 1.0},
                'ljung_box': {'statistic': 0.0, 'pvalue': 1.0}
            }
        
        # t-test: mean return > 0
        t_stat, t_pvalue = stats.ttest_1samp(returns, 0)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # Ljung-Box test for autocorrelation
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(returns, lags=10, return_df=False)
            # acorr_ljungbox returns (statistic, pvalue) tuple
            lb_stat = lb_result[0] if isinstance(lb_result, tuple) else lb_result.statistic
            lb_pvalue = lb_result[1] if isinstance(lb_result, tuple) else lb_result.pvalue
            # Handle array return (take first lag if multiple)
            if isinstance(lb_stat, (list, np.ndarray)):
                lb_stat = lb_stat[0] if len(lb_stat) > 0 else 0.0
            if isinstance(lb_pvalue, (list, np.ndarray)):
                lb_pvalue = lb_pvalue[0] if len(lb_pvalue) > 0 else 1.0
        except ImportError:
            self.logger.warning("statsmodels not available, using default Ljung-Box values")
            lb_stat, lb_pvalue = 0.0, 1.0
        except Exception as e:
            self.logger.warning(f"Ljung-Box test failed: {str(e)}, using default values")
            lb_stat, lb_pvalue = 0.0, 1.0
        
        return {
            't_test': {
                'statistic': float(t_stat),
                'pvalue': float(t_pvalue),
                'null_hypothesis': 'mean_return = 0'
            },
            'jarque_bera': {
                'statistic': float(jb_stat),
                'pvalue': float(jb_pvalue),
                'null_hypothesis': 'returns are normally distributed'
            },
            'ljung_box': {
                'statistic': float(lb_stat),
                'pvalue': float(lb_pvalue),
                'null_hypothesis': 'no autocorrelation'
            }
        }
    
    def _annualize_return(self, total_return: float, num_days: int) -> float:
        """Annualize return."""
        if num_days == 0:
            return 0.0
        
        annualized = total_return * (252 / num_days)
        
        return annualized
    
    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside standard deviation."""
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = downside_returns.std()
        
        return downside_std
    
    def _get_zero_metrics(self) -> Dict:
        """Return zero metrics for empty trades."""
        return {
            'total_return_pct': 0.0,
            'annualized_return_pct': 0.0,
            'cumulative_return': 1.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'max_drawdown_duration_days': 0,
            'drawdown_start_date': None,
            'drawdown_end_date': None,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_pct': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'avg_holding_period_days': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_streak': 0,
            'recovery_factor': 0.0
        }

