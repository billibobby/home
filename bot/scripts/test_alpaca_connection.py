"""
Alpaca API Connection Test Script

Standalone script to validate Alpaca API credentials and test basic connectivity.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.config_loader import Config
from trading_bot.logger import setup_logger
from trading_bot.utils.secrets_store import get_api_key
from trading_bot.exchanges.alpaca_client import AlpacaClient


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_check(name, status, details=""):
    """Print check result."""
    symbol = "✓" if status else "✗"
    status_text = "PASS" if status else "FAIL"
    print(f"{symbol} {name}: {status_text}")
    if details:
        print(f"    {details}")


def test_credentials():
    """Test if API credentials exist."""
    print_header("Testing Credentials")
    
    api_key = get_api_key('alpaca', 'api_key')
    api_secret = get_api_key('alpaca', 'api_secret')
    
    key_found = api_key is not None
    secret_found = api_secret is not None
    
    print_check("API Key found", key_found)
    print_check("API Secret found", secret_found)
    
    return key_found and secret_found


def test_connection(client):
    """Test API connection."""
    print_header("Testing Connection")
    
    try:
        account = client.get_account()
        
        if 'error' in account:
            print_check("Connection", False, account.get('message', 'Unknown error'))
            return False
        
        print_check("Connection", True)
        print(f"    Cash: ${account.get('cash', 0):,.2f}")
        print(f"    Equity: ${account.get('equity', 0):,.2f}")
        print(f"    Buying Power: ${account.get('buying_power', 0):,.2f}")
        
        return True
    except Exception as e:
        print_check("Connection", False, str(e))
        return False


def test_market_status(client):
    """Test market status check."""
    print_header("Testing Market Status")
    
    try:
        is_open = client.is_market_open()
        hours = client.get_market_hours()
        
        print_check("Market Status", True)
        status = "Open" if is_open else "Closed"
        print(f"    Market: {status}")
        
        if hours.get('next_open'):
            print(f"    Next Open: {hours['next_open']}")
        if hours.get('next_close'):
            print(f"    Next Close: {hours['next_close']}")
        
        return True
    except Exception as e:
        print_check("Market Status", False, str(e))
        return False


def test_positions(client):
    """Test position retrieval."""
    print_header("Testing Positions")
    
    try:
        positions = client.get_all_positions()
        
        print_check("Positions", True)
        print(f"    Number of positions: {len(positions)}")
        
        if positions:
            for pos in positions[:5]:  # Show first 5
                print(f"    - {pos['symbol']}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f}")
        
        return True
    except Exception as e:
        print_check("Positions", False, str(e))
        return False


def test_market_data(client):
    """Test market data retrieval."""
    print_header("Testing Market Data")
    
    try:
        # Test with a common symbol
        test_symbol = 'AAPL'
        bars = client.get_bars(test_symbol, '1Day', limit=5)
        
        if bars.empty:
            print_check("Market Data", False, f"No data returned for {test_symbol}")
            return False
        
        print_check("Market Data", True)
        print(f"    Retrieved {len(bars)} bars for {test_symbol}")
        if not bars.empty:
            latest = bars.iloc[-1]
            print(f"    Latest Close: ${latest['close']:.2f}")
            print(f"    Volume: {latest['volume']:,}")
        
        return True
    except Exception as e:
        print_check("Market Data", False, str(e))
        return False


def main():
    """Main test function."""
    print_header("Alpaca API Connection Test")
    
    # Initialize
    config = Config()
    logger = setup_logger('alpaca_test', log_level='INFO')
    
    # Test credentials
    if not test_credentials():
        print("\n❌ Credentials not found. Please set up API keys.")
        print("\nOptions:")
        print("  1. Use keyring:")
        print("     python -c \"from trading_bot.utils.secrets_store import store_api_key; store_api_key('alpaca', 'api_key', 'YOUR_KEY')\"")
        print("  2. Set environment variables: ALPACA_API_KEY and ALPACA_API_SECRET")
        return False
    
    # Get credentials
    api_key = get_api_key('alpaca', 'api_key')
    api_secret = get_api_key('alpaca', 'api_secret')
    
    # Create client
    print_header("Creating Alpaca Client")
    try:
        client = AlpacaClient(api_key, api_secret, paper_mode=True, logger=logger)
        print_check("Client initialization", True, "Mode: Paper Trading")
    except Exception as e:
        print_check("Client initialization", False, str(e))
        return False
    
    # Run tests
    results = {
        "Connection": test_connection(client),
        "Market Status": test_market_status(client),
        "Positions": test_positions(client),
        "Market Data": test_market_data(client),
    }
    
    # Summary
    print_header("Test Summary")
    passed = sum(results.values())
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    if passed == total:
        print("\n🎉 All tests passed! Alpaca connection is working.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)




