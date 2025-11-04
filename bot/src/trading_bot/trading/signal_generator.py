"""
Signal Generator Module

Converts model predictions into trading signals.
"""

from enum import Enum
from typing import Dict, Optional
from datetime import datetime


class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SignalGenerator:
    """
    Converts ML model predictions into actionable trading signals.
    
    Applies thresholds and confidence filters to generate signals.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the signal generator.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Load thresholds from config
        self.buy_threshold = config.get('models.prediction.buy_threshold', 0.6)
        self.sell_threshold = config.get('models.prediction.sell_threshold', 0.6)
        self.min_confidence = config.get('models.prediction.min_confidence', 0.5)
        
        # Target type
        self.target_type = config.get('models.xgboost.target_type', 'regression')
        
        self.logger.info(
            f"Signal generator initialized (buy: {self.buy_threshold}, "
            f"sell: {self.sell_threshold}, min_conf: {self.min_confidence})"
        )
    
    def generate_signal(self, prediction, confidence: float, 
                       current_price: float, symbol: str = None) -> Dict:
        """
        Convert prediction to trading signal.
        
        Args:
            prediction: Model prediction (price or class)
            confidence: Prediction confidence (0-1)
            current_price: Current market price
            symbol: Trading symbol (optional)
            
        Returns:
            Dictionary with signal details
        """
        # Choose method based on target type
        if self.target_type == 'regression':
            signal_type = self._regression_to_signal(prediction, current_price, confidence)
        else:
            signal_type = self._classification_to_signal(prediction, confidence)
        
        # Calculate signal strength
        strength = self.get_signal_strength(signal_type, confidence)
        
        # Create signal object
        signal = {
            'type': signal_type.value,
            'confidence': confidence,
            'strength': strength,
            'prediction': prediction,
            'current_price': current_price,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'reasoning': self._get_reasoning(signal_type, prediction, current_price, confidence)
        }
        
        self.logger.info(
            f"Generated signal: {signal_type.value} for {symbol or 'unknown'} "
            f"(confidence: {confidence:.2f}, strength: {strength:.2f})"
        )
        
        return signal
    
    def _regression_to_signal(self, predicted_price: float, 
                              current_price: float, 
                              confidence: float) -> SignalType:
        """
        Convert regression prediction to signal.
        
        Args:
            predicted_price: Predicted future price
            current_price: Current market price
            confidence: Prediction confidence
            
        Returns:
            SignalType enum
        """
        # Calculate expected return
        expected_return = (predicted_price - current_price) / current_price
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            return SignalType.HOLD
        
        # Generate signal based on expected return
        if expected_return > self.buy_threshold:
            # Strong buy if very high expected return
            if expected_return > self.buy_threshold * 2 and confidence > 0.7:
                return SignalType.STRONG_BUY
            return SignalType.BUY
        
        elif expected_return < -self.sell_threshold:
            # Strong sell if very high expected loss
            if expected_return < -self.sell_threshold * 2 and confidence > 0.7:
                return SignalType.STRONG_SELL
            return SignalType.SELL
        
        else:
            return SignalType.HOLD
    
    def _classification_to_signal(self, predicted_class: int, 
                                  probability: float) -> SignalType:
        """
        Convert classification prediction to signal.
        
        Args:
            predicted_class: Predicted class (0=down, 1=up)
            probability: Class probability
            
        Returns:
            SignalType enum
        """
        # Check confidence threshold
        if probability < self.min_confidence:
            return SignalType.HOLD
        
        # Generate signal based on predicted class
        if predicted_class == 1:  # Up
            if probability > self.buy_threshold:
                # Strong buy if very high probability
                if probability > 0.8:
                    return SignalType.STRONG_BUY
                return SignalType.BUY
        
        elif predicted_class == 0:  # Down
            if probability > self.sell_threshold:
                # Strong sell if very high probability
                if probability > 0.8:
                    return SignalType.STRONG_SELL
                return SignalType.SELL
        
        return SignalType.HOLD
    
    def validate_signal(self, signal: Dict) -> bool:
        """
        Ensure signal is valid.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['type', 'confidence', 'timestamp']
        
        for field in required_fields:
            if field not in signal:
                self.logger.warning(f"Invalid signal: missing field '{field}'")
                return False
        
        # Validate signal type
        try:
            SignalType(signal['type'])
        except ValueError:
            self.logger.warning(f"Invalid signal type: {signal['type']}")
            return False
        
        # Validate confidence
        if not 0 <= signal['confidence'] <= 1:
            self.logger.warning(f"Invalid confidence: {signal['confidence']}")
            return False
        
        return True
    
    def get_signal_strength(self, signal_type: SignalType, 
                           confidence: float) -> float:
        """
        Calculate signal strength (0-1).
        
        Args:
            signal_type: Type of signal
            confidence: Prediction confidence
            
        Returns:
            Signal strength value
        """
        # Base strength on signal type
        if signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            base_strength = 1.0
        elif signal_type in [SignalType.BUY, SignalType.SELL]:
            base_strength = 0.7
        else:
            base_strength = 0.0
        
        # Adjust by confidence
        strength = base_strength * confidence
        
        return min(strength, 1.0)
    
    def _get_reasoning(self, signal_type: SignalType, prediction, 
                      current_price: float, confidence: float) -> str:
        """
        Generate human-readable reasoning for signal.
        
        Args:
            signal_type: Generated signal type
            prediction: Model prediction
            current_price: Current price
            confidence: Prediction confidence
            
        Returns:
            Reasoning string
        """
        if self.target_type == 'regression':
            expected_return = (prediction - current_price) / current_price * 100
            return (
                f"{signal_type.value}: Predicted price ${prediction:.2f} "
                f"vs current ${current_price:.2f} "
                f"(expected return: {expected_return:+.2f}%, "
                f"confidence: {confidence:.2f})"
            )
        else:
            direction = "up" if prediction == 1 else "down"
            return (
                f"{signal_type.value}: Predicted {direction} movement "
                f"with {confidence:.2f} confidence"
            )
    
    def should_execute_signal(self, signal: Dict) -> bool:
        """
        Determine if signal should be executed.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            True if should execute, False otherwise
        """
        # Don't execute HOLD signals
        if signal['type'] == SignalType.HOLD.value:
            return False
        
        # Check minimum confidence
        if signal['confidence'] < self.min_confidence:
            self.logger.debug(f"Signal confidence too low: {signal['confidence']}")
            return False
        
        # Validate signal
        if not self.validate_signal(signal):
            return False
        
        return True

