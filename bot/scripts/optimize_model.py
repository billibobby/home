#!/usr/bin/env python3
"""
CLI Script for Hyperparameter Optimization

Standalone script for running hyperparameter optimization from command line.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.config_loader import Config
from trading_bot.logger import setup_logger
from trading_bot.data.stock_fetcher import StockDataFetcher
from trading_bot.data.feature_engineer import FeatureEngineer
from trading_bot.models.optimizer import XGBoostOptimizer
from trading_bot.models.optimization_monitor import OptimizationMonitor
from trading_bot.models.param_analyzer import ParameterAnalyzer


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Optimize XGBoost hyperparameters for trading bot',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Stock symbol (e.g., AAPL)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date for data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date for data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of optimization trials (default: 100)'
    )
    
    parser.add_argument(
        '--objective',
        type=str,
        default='sharpe',
        choices=['sharpe', 'sortino', 'returns', 'calmar'],
        help='Optimization objective (default: sharpe)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='best_params.json',
        help='Output file for best parameters (default: best_params.json)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint file (optional)'
    )
    
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Enable distributed optimization'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Load configuration
    config = Config()
    config.load()
    
    # Setup logger
    logger = setup_logger('optimize_model', config)
    logger.info("Starting hyperparameter optimization")
    
    # Parse dates
    try:
        start_date = pd.to_datetime(args.start_date)
        end_date = pd.to_datetime(args.end_date)
    except Exception as e:
        logger.error(f"Invalid date format: {str(e)}")
        sys.exit(1)
    
    # Fetch historical data
    logger.info(f"Fetching data for {args.symbol} from {start_date} to {end_date}")
    fetcher = StockDataFetcher(config, logger)
    
    try:
        data = fetcher.fetch_historical_data(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        
        if data is None or len(data) == 0:
            logger.error("No data fetched")
            sys.exit(1)
        
        logger.info(f"Fetched {len(data)} data points")
    except Exception as e:
        logger.error(f"Failed to fetch data: {str(e)}")
        sys.exit(1)
    
    # Create features
    logger.info("Creating features")
    feature_engineer = FeatureEngineer(config, logger)
    
    try:
        features_df = feature_engineer.create_features(data)
        
        # Prepare training data
        from trading_bot.data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor(config, logger)
        
        # Split into train/validation
        from sklearn.model_selection import train_test_split
        
        X, y = preprocessor.prepare_training_data(features_df, target_column='Close')
        
        # Handle NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        sys.exit(1)
    
    # Check for distributed optimization
    storage = None
    study_name = None
    if args.distributed:
        dist_config = config.get('models.optimization.distributed', {})
        storage = dist_config.get('storage', 'sqlite:///optuna.db')
        study_name = dist_config.get('study_name', f'xgboost_optimization_{args.objective}')
        logger.info(f"Distributed optimization enabled: storage={storage}, study_name={study_name}")
        logger.info("Multiple workers can run this command simultaneously")
    
    # Create optimizer
    optimizer = XGBoostOptimizer(
        X_train, y_train, X_val, y_val,
        config, logger, objective=args.objective,
        storage=storage, study_name=study_name
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        try:
            optimizer.load_study(args.resume)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {str(e)}, starting fresh")
    
    # Create monitor
    monitor = OptimizationMonitor(config, logger)
    monitor.start_time = datetime.now()
    
    # Progress callback
    def progress_callback(study, trial):
        monitor._on_trial_complete(study, trial)
        if hasattr(progress_callback, 'pbar'):
            progress_callback.pbar.update(1)
            progress_callback.pbar.set_description(
                f"Best: {study.best_value:.4f}" if study.best_value is not None else "Best: N/A"
            )
    
    # Create progress bar
    progress_callback.pbar = tqdm(total=args.n_trials, desc="Optimizing")
    
    # Run optimization
    logger.info(f"Starting optimization with {args.n_trials} trials")
    try:
        best_params = optimizer.optimize(
            n_trials=args.n_trials,
            timeout=config.get('models.optimization.timeout_seconds', None),
            callbacks=[progress_callback]
        )
        
        progress_callback.pbar.close()
        
        logger.info("Optimization completed")
        logger.info(f"Best parameters: {best_params}")
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        progress_callback.pbar.close()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        progress_callback.pbar.close()
        sys.exit(1)
    
    # Save results
    logger.info(f"Saving results to {args.output}")
    
    # Save best parameters
    with open(args.output, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Save optimization history
    history_df = optimizer.get_optimization_history()
    history_file = args.output.replace('.json', '_history.csv')
    history_df.to_csv(history_file, index=False)
    logger.info(f"Optimization history saved to {history_file}")
    
    # Generate and save plots
    try:
        plot_dir = Path(config.get('models.optimization.monitoring.plot_dir', 'optimization_plots'))
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization history plot
        fig = optimizer.plot_optimization_history()
        plot_file = plot_dir / 'optimization_history.html'
        fig.write_html(str(plot_file))
        logger.info(f"Optimization history plot saved to {plot_file}")
        
        # Parameter importance plot
        fig = optimizer.plot_param_importances()
        plot_file = plot_dir / 'param_importance.html'
        fig.write_html(str(plot_file))
        logger.info(f"Parameter importance plot saved to {plot_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save plots: {str(e)}")
    
    # Analyze parameters
    try:
        analyzer = ParameterAnalyzer(optimizer.study, config, logger)
        insights = analyzer.generate_insights()
        
        insights_file = args.output.replace('.json', '_insights.txt')
        with open(insights_file, 'w') as f:
            f.write(insights)
        logger.info(f"Insights saved to {insights_file}")
        print("\n" + insights)
    except Exception as e:
        logger.warning(f"Failed to generate insights: {str(e)}")
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Objective: {args.objective}")
    print(f"Trials: {args.n_trials}")
    print(f"Best Value: {optimizer.best_value:.4f}" if optimizer.best_value else "N/A")
    print("\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print("="*60)
    
    logger.info("Optimization script completed successfully")


if __name__ == '__main__':
    main()

