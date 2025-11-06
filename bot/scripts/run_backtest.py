#!/usr/bin/env python3
"""
Command-line interface for running walk-forward backtests.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.config_loader import Config
from trading_bot.logger import setup_logger
from trading_bot.data import StockDataFetcher, FeatureEngineer
from trading_bot.trading import SignalGenerator
from trading_bot.models import XGBoostTrainer
from trading_bot.backtesting import WalkForwardBacktest
from trading_bot.backtesting.visualizations import BacktestVisualizer
from tqdm import tqdm
import pandas as pd


class ProgressCallback:
    """Progress callback for tqdm."""
    
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc='Backtesting', unit='window')
    
    def update(self, current, total, message):
        self.pbar.update(1)
        self.pbar.set_description(message)
    
    def close(self):
        self.pbar.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run walk-forward backtest')
    parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--model', default='xgboost', help='Model type')
    parser.add_argument('--output-dir', default='backtest_results/', help='Output directory')
    parser.add_argument('--train-period', type=int, default=252, help='Training period (days)')
    parser.add_argument('--test-period', type=int, default=21, help='Test period (days)')
    parser.add_argument('--step-size', type=int, default=21, help='Step size (days)')
    parser.add_argument('--generate-report', action='store_true', help='Generate HTML report')
    parser.add_argument('--save-plots', action='store_true', help='Save all plots')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Setup logging
    logger = setup_logger('backtest_cli', config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting backtest for {args.symbol} from {args.start_date} to {args.end_date}")
    
    try:
        # Initialize components
        fetcher = StockDataFetcher(config, logger)
        feature_engineer = FeatureEngineer(config, logger)
        signal_generator = SignalGenerator(config, logger)
        
        # Fetch data
        logger.info(f"Fetching historical data for {args.symbol}...")
        data = fetcher.fetch_historical_data(args.symbol, args.start_date, args.end_date)
        
        logger.info(f"Fetched {len(data)} rows of data")
        logger.info(f"Date range: {data.iloc[0]['Date']} to {data.iloc[-1]['Date']}")
        
        # Create backtest engine
        backtest_engine = WalkForwardBacktest(
            data=data,
            config=config,
            logger=logger,
            feature_engineer=feature_engineer,
            signal_generator=signal_generator,
            model_class=XGBoostTrainer
        )
        
        # Set symbol on backtest engine
        backtest_engine.symbol = args.symbol
        
        # Override window parameters
        backtest_engine.train_period_days = args.train_period
        backtest_engine.test_period_days = args.test_period
        backtest_engine.step_size_days = args.step_size
        
        # Create progress callback
        # Estimate total windows (rough estimate)
        total_days = (pd.to_datetime(args.end_date) - pd.to_datetime(args.start_date)).days
        estimated_windows = max(1, (total_days - args.train_period - args.test_period) // args.step_size)
        progress_callback = ProgressCallback(estimated_windows)
        
        def progress_wrapper(current, total, message):
            progress_callback.update(current, total, message)
        
        # Run backtest
        logger.info("Starting walk-forward backtest...")
        results = backtest_engine.run(
            initial_capital=10000.0,
            progress_callback=progress_wrapper
        )
        
        progress_callback.close()
        
        # Print summary
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        print(f"Symbol: {results.symbol or args.symbol}")
        print(f"Period: {results.backtest_start_date} to {results.backtest_end_date}")
        print(f"Model: {results.model_type}")
        print("\nPerformance Metrics:")
        print(f"  Total Return:         {results.total_return:.2f}%")
        print(f"  Annualized Return:    {results.annualized_return:.2f}%")
        print(f"  Sharpe Ratio:         {results.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:        {results.sortino_ratio:.2f}")
        print(f"  Max Drawdown:         {results.max_drawdown:.2f}%")
        print(f"  Win Rate:             {results.win_rate:.2f}%")
        print(f"  Profit Factor:        {results.profit_factor:.2f}")
        print("\nTrade Statistics:")
        print(f"  Total Trades:         {results.total_trades}")
        print(f"  Winning Trades:       {results.winning_trades}")
        print(f"  Losing Trades:        {results.losing_trades}")
        print(f"  Avg Win:              ${results.avg_win:.2f}")
        print(f"  Avg Loss:             ${results.avg_loss:.2f}")
        print("="*60 + "\n")
        
        # Save results
        results_file = output_dir / f"{args.symbol}_{args.start_date}_{args.end_date}_results.json"
        results.save_to_file(str(results_file), format='json')
        logger.info(f"Results saved to {results_file}")
        
        # Generate report if requested
        if args.generate_report:
            visualizer = BacktestVisualizer(config, logger)
            report_file = output_dir / f"{args.symbol}_{args.start_date}_{args.end_date}_report.html"
            visualizer.generate_html_report(results, str(report_file))
            logger.info(f"HTML report saved to {report_file}")
            print(f"Report:  {report_file}")
        
        # Save plots if requested
        if args.save_plots:
            plots_dir = output_dir / "plots"
            visualizer = BacktestVisualizer(config, logger)
            saved_files = visualizer.plot_all(results, str(plots_dir))
            logger.info(f"Plots saved to {plots_dir}")
            print(f"Plots:   {plots_dir}/")
        
        print("\nFiles Saved:")
        print(f"  Results: {results_file}")
        if args.generate_report:
            print(f"  Report:  {report_file}")
        if args.save_plots:
            print(f"  Plots:   {plots_dir}/")
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

