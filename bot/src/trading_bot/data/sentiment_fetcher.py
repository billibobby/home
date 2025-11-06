"""
Sentiment Data Fetcher Module

Fetches news sentiment data from Finnhub and Alpha Vantage APIs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import json
import time
import requests

from trading_bot.utils.helpers import retry_on_failure, ensure_dir
from trading_bot.utils.exceptions import DataError, APIError
from trading_bot.utils.secrets_store import get_api_key
from trading_bot.utils.paths import get_writable_app_dir


class SentimentDataFetcher:
    """
    Fetches news sentiment data from Finnhub and Alpha Vantage APIs.
    
    Handles fallback logic, caching, and rate limiting.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the sentiment data fetcher.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Get API keys
        self.finnhub_api_key = get_api_key('finnhub', 'api_key')
        self.alpha_vantage_api_key = get_api_key('alpha_vantage', 'api_key')
        
        # Configuration
        self.sentiment_enabled = config.get('data.sentiment.enabled', False)
        self.primary_source = config.get('data.sentiment.primary_source', 'finnhub')
        self.fallback_enabled = config.get('data.sentiment.fallback_enabled', True)
        self.cache_ttl_hours = config.get('data.sentiment.cache_ttl_hours', 1)
        self.lookback_days = config.get('data.sentiment.lookback_days', 7)
        
        # Setup cache directory
        sentiment_cache_path = config.get('data.sentiment.sentiment_cache_path', 'data/sentiment/')
        if Path(sentiment_cache_path).is_absolute():
            self.cache_dir = str(Path(sentiment_cache_path))
        else:
            subdir = Path(sentiment_cache_path).parts[-1] if Path(sentiment_cache_path).parts else 'sentiment'
            self.cache_dir = get_writable_app_dir(subdir)
        
        ensure_dir(self.cache_dir)
        self.logger.info(f"Sentiment data cache enabled at: {self.cache_dir}")
        
        # Initialize Finnhub client if API key available
        self.finnhub_client = None
        if self.finnhub_api_key:
            try:
                import finnhub
                self.finnhub_client = finnhub.Client(api_key=self.finnhub_api_key)
                self.logger.debug("Finnhub client initialized")
            except ImportError:
                self.logger.warning("finnhub-python not installed. Finnhub sentiment will be unavailable.")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Finnhub client: {str(e)}")
    
    @retry_on_failure(max_attempts=3, delay=2.0, backoff=2.0)
    def _fetch_finnhub_sentiment(self, symbol: str, days: int = None) -> Dict:
        """
        Fetch sentiment data from Finnhub API.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of sentiment history (default: from config)
            
        Returns:
            Dictionary with sentiment data
            
        Raises:
            APIError: If API call fails
        """
        if not self.finnhub_client:
            raise APIError(
                "Finnhub client not initialized. Check API key.",
                exchange='finnhub'
            )
        
        if days is None:
            days = self.lookback_days
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch news sentiment
            sentiment_data = self.finnhub_client.news_sentiment(symbol)
            
            # Fetch company news for additional context
            news_data = self.finnhub_client.company_news(
                symbol,
                _from=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d')
            )
            
            # Parse sentiment data
            if not sentiment_data or 'sentiment' not in sentiment_data:
                return None
            
            sentiment = sentiment_data.get('sentiment', {})
            buzz = sentiment_data.get('buzz', {})
            
            # Read bullishPercent and bearishPercent directly from sentiment_data
            bullish_percent = sentiment.get('bullishPercent', 0)
            bearish_percent = sentiment.get('bearishPercent', 0)
            
            # Calculate sentiment score (-1 to 1)
            sentiment_score = (bullish_percent - bearish_percent) / 100
            
            # Extract additional metrics
            articles_count = len(news_data) if news_data else 0
            buzz_score = buzz.get('articlesInLastWeek', 0) if buzz else 0
            
            result = {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'confidence': abs(sentiment_score),  # Use absolute value as confidence
                'source': 'finnhub',
                'timestamp': datetime.now(),
                'articles_count': articles_count,
                'bullish_percent': bullish_percent,
                'bearish_percent': bearish_percent,
                'buzz_score': buzz_score
            }
            
            self.logger.debug(f"Finnhub sentiment for {symbol}: {sentiment_score:.3f}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            if 'rate limit' in error_msg.lower() or '429' in error_msg:
                self.logger.warning(f"Finnhub rate limit exceeded for {symbol}")
                raise APIError(
                    f"Finnhub rate limit exceeded: {error_msg}",
                    exchange='finnhub',
                    status_code=429
                )
            raise APIError(
                f"Finnhub API error: {error_msg}",
                exchange='finnhub',
                details={'error': error_msg}
            )
    
    @retry_on_failure(max_attempts=3, delay=2.0, backoff=2.0)
    def _fetch_alpha_vantage_sentiment(self, symbol: str, days: int = None) -> Dict:
        """
        Fetch sentiment data from Alpha Vantage API.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of sentiment history (default: from config)
            
        Returns:
            Dictionary with sentiment data
            
        Raises:
            APIError: If API call fails
        """
        if not self.alpha_vantage_api_key:
            raise APIError(
                "Alpha Vantage API key not configured",
                exchange='alpha_vantage'
            )
        
        if days is None:
            days = self.lookback_days
        
        try:
            # Alpha Vantage NEWS_SENTIMENT endpoint
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.alpha_vantage_api_key,
                'limit': 1000  # Maximum limit
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise APIError(
                    f"Alpha Vantage API error: {data['Error Message']}",
                    exchange='alpha_vantage',
                    details=data
                )
            
            if 'Note' in data:
                # Rate limit message
                self.logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                raise APIError(
                    f"Alpha Vantage rate limit: {data['Note']}",
                    exchange='alpha_vantage',
                    status_code=429
                )
            
            if 'feed' not in data or not data['feed']:
                return None
            
            # Calculate weighted average sentiment
            feed = data['feed']
            sentiment_scores = []
            relevance_scores = []
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for item in feed:
                # Filter by date
                try:
                    item_date = datetime.fromisoformat(item.get('time_published', '').replace('Z', '+00:00'))
                    if item_date < cutoff_date:
                        continue
                except Exception:
                    pass
                
                # Extract ticker sentiment
                ticker_sentiment = item.get('ticker_sentiment', [])
                for ticker_data in ticker_sentiment:
                    if ticker_data.get('ticker') == symbol:
                        relevance = float(ticker_data.get('relevance_score', '0'))
                        sentiment_score = float(ticker_data.get('ticker_sentiment_score', '0'))
                        
                        sentiment_scores.append(sentiment_score)
                        relevance_scores.append(relevance)
                        break
            
            # Calculate weighted average
            if sentiment_scores and relevance_scores:
                total_relevance = sum(relevance_scores)
                if total_relevance > 0:
                    weighted_sentiment = sum(
                        score * weight for score, weight in zip(sentiment_scores, relevance_scores)
                    ) / total_relevance
                else:
                    weighted_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            else:
                weighted_sentiment = 0.0
            
            # Normalize to -1 to 1 range (Alpha Vantage uses -1 to 1 already)
            sentiment_score = max(-1.0, min(1.0, weighted_sentiment))
            articles_count = len([item for item in feed if item.get('time_published')])
            
            result = {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'confidence': abs(sentiment_score),
                'source': 'alpha_vantage',
                'timestamp': datetime.now(),
                'articles_count': articles_count
            }
            
            self.logger.debug(f"Alpha Vantage sentiment for {symbol}: {sentiment_score:.3f}")
            return result
            
        except requests.exceptions.RequestException as e:
            raise APIError(
                f"Alpha Vantage request failed: {str(e)}",
                exchange='alpha_vantage',
                details={'error': str(e)}
            )
        except Exception as e:
            raise APIError(
                f"Alpha Vantage API error: {str(e)}",
                exchange='alpha_vantage',
                details={'error': str(e)}
            )
    
    def fetch_sentiment(self, symbol: str, days: int = None) -> Optional[Dict]:
        """
        Fetch sentiment data with fallback logic.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of sentiment history (default: from config)
            
        Returns:
            Dictionary with sentiment data or None if both sources fail
            
        Raises:
            DataError: If both sources fail and fallback is disabled
        """
        if days is None:
            days = self.lookback_days
        
        # Check cache first
        cached_sentiment = self._load_cached_sentiment(symbol)
        if cached_sentiment is not None:
            self.logger.debug(f"Using cached sentiment for {symbol}")
            return cached_sentiment
        
        # Try primary source
        primary_failed = False
        if self.primary_source == 'finnhub':
            try:
                sentiment = self._fetch_finnhub_sentiment(symbol, days)
                if sentiment:
                    self._cache_sentiment(symbol, sentiment)
                    self.logger.info(f"Fetched sentiment for {symbol} from Finnhub")
                    return sentiment
            except Exception as e:
                primary_failed = True
                self.logger.warning(f"Primary source (Finnhub) failed for {symbol}: {str(e)}")
        elif self.primary_source == 'alpha_vantage':
            try:
                sentiment = self._fetch_alpha_vantage_sentiment(symbol, days)
                if sentiment:
                    self._cache_sentiment(symbol, sentiment)
                    self.logger.info(f"Fetched sentiment for {symbol} from Alpha Vantage")
                    return sentiment
            except Exception as e:
                primary_failed = True
                self.logger.warning(f"Primary source (Alpha Vantage) failed for {symbol}: {str(e)}")
        
        # Try fallback if enabled and primary failed
        if self.fallback_enabled and primary_failed:
            fallback_source = 'alpha_vantage' if self.primary_source == 'finnhub' else 'finnhub'
            try:
                if fallback_source == 'finnhub':
                    sentiment = self._fetch_finnhub_sentiment(symbol, days)
                else:
                    sentiment = self._fetch_alpha_vantage_sentiment(symbol, days)
                
                if sentiment:
                    self._cache_sentiment(symbol, sentiment)
                    self.logger.info(f"Fetched sentiment for {symbol} from {fallback_source} (fallback)")
                    return sentiment
            except Exception as e:
                self.logger.warning(f"Fallback source ({fallback_source}) also failed for {symbol}: {str(e)}")
        
        # Both sources failed
        self.logger.warning(f"Failed to fetch sentiment for {symbol} from both sources")
        if not self.fallback_enabled:
            raise DataError(
                f"Sentiment fetch failed for {symbol} and fallback is disabled",
                symbol=symbol,
                details={'primary_source': self.primary_source}
            )
        
        return None
    
    def fetch_multi_symbol_sentiment(self, symbols: List[str], days: int = None) -> Dict[str, Dict]:
        """
        Fetch sentiment for multiple symbols with rate limit handling.
        
        Args:
            symbols: List of stock ticker symbols
            days: Number of days of sentiment history
            
        Returns:
            Dictionary mapping symbols to sentiment data
        """
        if days is None:
            days = self.lookback_days
        
        results = {}
        
        for i, symbol in enumerate(symbols):
            try:
                sentiment = self.fetch_sentiment(symbol, days)
                if sentiment:
                    results[symbol] = sentiment
                
                # Add delay between requests to respect rate limits
                if i < len(symbols) - 1:
                    time.sleep(0.5)  # 500ms delay between requests
                    
            except Exception as e:
                self.logger.warning(f"Failed to fetch sentiment for {symbol}: {str(e)}")
                continue
        
        self.logger.info(f"Fetched sentiment for {len(results)}/{len(symbols)} symbols")
        return results
    
    def _cache_sentiment(self, symbol: str, sentiment_data: Dict) -> None:
        """
        Cache sentiment data.
        
        Args:
            symbol: Stock ticker symbol
            sentiment_data: Sentiment data dictionary
        """
        try:
            cache_file = Path(self.cache_dir) / f"{symbol}_sentiment.json"
            
            # Convert datetime to string for JSON serialization
            cache_data = sentiment_data.copy()
            if isinstance(cache_data.get('timestamp'), datetime):
                cache_data['timestamp'] = cache_data['timestamp'].isoformat()
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.debug(f"Cached sentiment for {symbol}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache sentiment for {symbol}: {str(e)}")
    
    def _load_cached_sentiment(self, symbol: str) -> Optional[Dict]:
        """
        Load cached sentiment data if available and fresh.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Cached sentiment data or None if not available/stale
        """
        try:
            cache_file = Path(self.cache_dir) / f"{symbol}_sentiment.json"
            
            if not cache_file.exists():
                return None
            
            # Check file age
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.total_seconds() > (self.cache_ttl_hours * 3600):
                self.logger.debug(f"Cached sentiment for {symbol} is stale ({file_age.total_seconds()/3600:.1f} hours old)")
                return None
            
            # Load cached data
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Convert timestamp string back to datetime
            if 'timestamp' in cache_data and isinstance(cache_data['timestamp'], str):
                cache_data['timestamp'] = datetime.fromisoformat(cache_data['timestamp'])
            
            self.logger.debug(f"Loaded cached sentiment for {symbol}")
            return cache_data
            
        except Exception as e:
            self.logger.warning(f"Failed to load cached sentiment for {symbol}: {str(e)}")
            return None

