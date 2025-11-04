"""
Trading Strategy Module

Implements trading strategies based on XGBoost predictions.
"""

import pandas as pd
from typing import Dict, List, Optional

from trading_bot.data import StockDataFetcher, FeatureEngineer
from trading_bot.models import XGBoostPredictor
from trading_bot.trading.signal_generator import SignalGenerator, SignalType


class XGBoostStrategy:
    """
    Trading strategy based on XGBoost model predictions.
    
    Integrates data fetching, feature engineering, prediction, and signal generation.
    """
    
    def __init__(self, config, logger, predictor: XGBoostPredictor, 
                 signal_generator: SignalGenerator, paper_trading_engine=None):
        """
        Initialize the XGBoost trading strategy.
        
        Args:
            config: Configuration object
            logger: Logger instance
            predictor: XGBoost predictor instance
            signal_generator: Signal generator instance
            paper_trading_engine: Optional PaperTradingEngine instance for paper trading
        """
        self.config = config
        self.logger = logger
        self.predictor = predictor
        self.signal_generator = signal_generator
        self.paper_trading_engine = paper_trading_engine
        
        # Initialize dependencies
        self.data_fetcher = StockDataFetcher(config, logger)
        self.feature_engineer = FeatureEngineer(config, logger)
        
        # Load trading parameters
        self.position_size_pct = config.get('trading.position_size_percentage', 10)
        self.risk_pct = config.get('trading.risk_percentage', 2)
        self.max_positions = config.get('trading.max_positions', 5)
        self.stop_loss_pct = config.get('trading.stop_loss_percentage', 2)
        self.take_profit_pct = config.get('trading.take_profit_percentage', 5)
        
        # State tracking
        self.active_positions = {}
        self.signal_history = []
        
        # Log paper trading status
        if paper_trading_engine and config.get('paper_trading.enabled', False):
            self.logger.info("XGBoost strategy initialized with paper trading enabled")
        else:
            self.logger.info("XGBoost strategy initialized")
    
    def analyze(self, symbol: str, account_balance: float = None) -> Optional[Dict]:
        """
        Analyze a symbol and generate trading decision.
        
        Args:
            symbol: Stock ticker symbol
            account_balance: Current account balance (optional)
            
        Returns:
            Dictionary with trading decision, or None if no action
        """
        try:
            self.logger.info(f"Analyzing {symbol}")
            
            # Step 1: Fetch latest data
            data = self._fetch_latest_data(symbol)
            
            if data is None or len(data) == 0:
                self.logger.warning(f"No data available for {symbol}")
                return None
            
            # Step 2: Prepare features
            features = self._prepare_features(data)
            
            if features is None:
                return None
            
            # Step 3: Get prediction
            current_price = data['Close'].iloc[-1]
            prediction_result = self._get_prediction(features, current_price)
            
            if prediction_result is None:
                return None
            
            # Step 4: Generate signal
            signal = self._generate_signal(
                prediction_result['prediction'],
                prediction_result['confidence'],
                current_price,
                symbol
            )
            
            # Store signal in history
            self.signal_history.append(signal)
            
            # Step 5: Check if we should execute
            if not self.should_execute_trade(signal):
                self.logger.info(f"Signal not actionable: {signal['type']}")
                return None
            
            # Step 6: Calculate position size
            if account_balance:
                position_size = self._calculate_position_size(signal, account_balance)
            else:
                position_size = None
            
            # Create trading decision
            decision = {
                'symbol': symbol,
                'signal': signal,
                'position_size': position_size,
                'current_price': current_price,
                'stop_loss': current_price * (1 - self.stop_loss_pct / 100),
                'take_profit': current_price * (1 + self.take_profit_pct / 100),
                'timestamp': signal['timestamp']
            }
            
            self.logger.info(
                f"Trading decision: {signal['type']} {symbol} "
                f"at ${current_price:.2f}"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {symbol}: {str(e)}")
            return None
    
    def _fetch_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get current market data.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with market data
        """
        try:
            # Get lookback period from config
            lookback_days = self.config.get('models.xgboost.lookback_days', 60)
            
            # Fetch data (add buffer for feature engineering)
            period = f"{lookback_days + 30}d"
            data = self.data_fetcher.fetch_latest_data(symbol, period=period)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
            return None
    
    def _prepare_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Apply feature engineering.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            DataFrame with features
        """
        try:
            features = self.feature_engineer.create_features(data)
            
            # Get the most recent row for prediction
            # (Remove any rows with NaN after feature engineering)
            features = features.dropna()
            
            if len(features) == 0:
                self.logger.warning("No valid features after preprocessing")
                return None
            
            return features.tail(1)  # Return only the latest row
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {str(e)}")
            return None
    
    def _get_prediction(self, features: pd.DataFrame, 
                       current_price: float) -> Optional[Dict]:
        """
        Get model prediction.
        
        Args:
            features: Feature DataFrame
            current_price: Current market price
            
        Returns:
            Dictionary with prediction and confidence
        """
        try:
            # Make prediction
            prediction = self.predictor.predict(features)[0]
            
            # Get confidence
            confidence = self.predictor.get_confidence(prediction, features)
            
            self.logger.debug(
                f"Prediction: {prediction:.2f}, Confidence: {confidence:.2f}"
            )
            
            return {
                'prediction': prediction,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return None
    
    def _generate_signal(self, prediction, confidence: float, 
                        current_price: float, symbol: str) -> Dict:
        """
        Convert prediction to trading signal.
        
        Args:
            prediction: Model prediction
            confidence: Prediction confidence
            current_price: Current price
            symbol: Trading symbol
            
        Returns:
            Signal dictionary
        """
        signal = self.signal_generator.generate_signal(
            prediction, confidence, current_price, symbol
        )
        
        return signal
    
    def _calculate_position_size(self, signal: Dict, 
                                 account_balance: float) -> float:
        """
        Determine trade size based on risk parameters.
        
        Args:
            signal: Trading signal
            account_balance: Available balance
            
        Returns:
            Position size in USD
        """
        # Base position size on percentage of account
        base_size = account_balance * (self.position_size_pct / 100)
        
        # Adjust by signal strength
        strength = signal.get('strength', 0.5)
        adjusted_size = base_size * strength
        
        # Apply risk limits
        max_risk = account_balance * (self.risk_pct / 100)
        risk_adjusted_size = min(adjusted_size, max_risk / (self.stop_loss_pct / 100))
        
        self.logger.debug(
            f"Position size: ${risk_adjusted_size:.2f} "
            f"(strength: {strength:.2f}, risk: ${max_risk:.2f})"
        )
        
        return risk_adjusted_size
    
    def should_execute_trade(self, signal: Dict) -> bool:
        """
        Final decision gate for trade execution.
        
        Args:
            signal: Trading signal
            
        Returns:
            True if should execute, False otherwise
        """
        # Check if signal is actionable
        if not self.signal_generator.should_execute_signal(signal):
            return False
        
        # Check max positions limit
        if len(self.active_positions) >= self.max_positions:
            self.logger.info(
                f"Max positions reached ({self.max_positions}), skipping trade"
            )
            return False
        
        # Check if we already have a position in this symbol
        symbol = signal.get('symbol')
        if symbol and symbol in self.active_positions:
            self.logger.info(f"Already have position in {symbol}, skipping")
            return False
        
        return True
    
    def analyze_portfolio(self, symbols: List[str], 
                         account_balance: float = None) -> List[Dict]:
        """
        Analyze multiple symbols and rank opportunities.
        
        Args:
            symbols: List of stock symbols
            account_balance: Available balance
            
        Returns:
            List of trading decisions, sorted by signal strength
        """
        decisions = []
        
        for symbol in symbols:
            decision = self.analyze(symbol, account_balance)
            
            if decision:
                decisions.append(decision)
        
        # Sort by signal strength (descending)
        decisions.sort(
            key=lambda x: x['signal'].get('strength', 0),
            reverse=True
        )
        
        # Limit to max_positions
        decisions = decisions[:self.max_positions]
        
        self.logger.info(
            f"Portfolio analysis complete: {len(decisions)} opportunities found"
        )
        
        return decisions
    
    def update_position(self, symbol: str, entry_price: float, 
                       size: float) -> None:
        """
        Track active position.
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            size: Position size
        """
        self.active_positions[symbol] = {
            'entry_price': entry_price,
            'size': size,
            'stop_loss': entry_price * (1 - self.stop_loss_pct / 100),
            'take_profit': entry_price * (1 + self.take_profit_pct / 100)
        }
        
        # Sync with paper trading engine if available
        if self.paper_trading_engine:
            # Position is already tracked in paper trading engine after execution
            # This is just for consistency
            pass
        
        self.logger.info(f"Position updated: {symbol} at ${entry_price:.2f}")
    
    def close_position(self, symbol: str, current_price: float = None) -> None:
        """
        Remove position from tracking.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price (required for paper trading)
        """
        if symbol in self.active_positions:
            # If paper trading engine exists, close position there first
            if self.paper_trading_engine:
                if current_price is None:
                    self.logger.warning(f"Current price required to close position in paper trading, but not provided for {symbol}")
                else:
                    self.paper_trading_engine.close_position(symbol, current_price, reason='manual')
            
            del self.active_positions[symbol]
            self.logger.info(f"Position closed: {symbol}")
    
    def get_active_positions(self) -> Dict:
        """
        Get current active positions.
        
        Returns:
            Dictionary of active positions
        """
        return self.active_positions.copy()
    
    def get_signal_history(self, limit: int = None) -> List[Dict]:
        """
        Get historical signals.
        
        Args:
            limit: Maximum number of signals to return
            
        Returns:
            List of signal dictionaries
        """
        if limit:
            return self.signal_history[-limit:]
        return self.signal_history.copy()
    
    def execute_decision(self, decision: Dict) -> Dict:
        """
        Execute a trading decision using paper trading engine.
        
        Args:
            decision: Trading decision dictionary from analyze() method
            
        Returns:
            Execution result dictionary
        """
        if self.paper_trading_engine is None:
            self.logger.warning("Paper trading engine not available - cannot execute decision")
            return {
                'success': False,
                'error': 'Paper trading engine not available'
            }
        
        try:
            signal = decision.get('signal')
            symbol = decision.get('symbol')
            current_price = decision.get('current_price')
            
            if not signal or not symbol or current_price is None:
                return {
                    'success': False,
                    'error': 'Invalid decision: missing required fields'
                }
            
            # Determine order side
            signal_type = signal.get('type')
            if signal_type in ['BUY', 'STRONG_BUY']:
                side = 'buy'
            elif signal_type in ['SELL', 'STRONG_SELL']:
                side = 'sell'
            else:
                return {
                    'success': False,
                    'error': f'Cannot execute HOLD signal'
                }
            
            # Calculate quantity based on order side
            if side == 'buy':
                # For BUY, calculate quantity from position_size
                position_size = decision.get('position_size')
                if position_size is None:
                    # Use current balance if available
                    account_summary = self.paper_trading_engine.get_account_summary()
                    account_balance = account_summary.get('cash_balance', 0)
                    position_size = account_balance * (self.position_size_pct / 100)
                
                quantity = position_size / current_price if current_price > 0 else 0
                
                result = self.paper_trading_engine.execute_buy_order(
                    symbol, quantity, current_price, order_type='market', signal=signal
                )
            else:  # sell
                # For SELL, look up actual holdings
                position = self.paper_trading_engine.get_position(symbol)
                if position is None:
                    return {
                        'success': False,
                        'error': f'No position found for {symbol} to sell'
                    }
                
                quantity = position['quantity']
                
                result = self.paper_trading_engine.execute_sell_order(
                    symbol, quantity, current_price, order_type='market', signal=signal
                )
            
            # Update in-memory position tracking if successful
            if result.get('success'):
                if side == 'buy':
                    self.update_position(
                        symbol,
                        result['execution_price'],
                        position_size
                    )
                else:  # sell
                    if result.get('position_closed', False):
                        self.close_position(symbol, current_price)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Failed to execute decision: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def sync_positions_with_engine(self):
        """
        Synchronize in-memory positions with paper trading engine state.
        
        Useful for initialization and recovery scenarios.
        """
        if self.paper_trading_engine is None:
            return
        
        try:
            engine_positions = self.paper_trading_engine.get_all_positions()
            
            # Update active_positions to match engine
            self.active_positions = {}
            for symbol, pos in engine_positions.items():
                self.active_positions[symbol] = {
                    'entry_price': pos['entry_price'],
                    'size': pos['quantity'] * pos.get('current_price', pos['entry_price']),
                    'stop_loss': pos.get('stop_loss'),
                    'take_profit': pos.get('take_profit')
                }
            
            self.logger.info(f"Synced {len(self.active_positions)} positions with paper trading engine")
        
        except Exception as e:
            self.logger.error(f"Failed to sync positions: {str(e)}")

