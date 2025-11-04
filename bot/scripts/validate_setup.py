"""
Setup Validation Script

Validates that the trading bot setup is correct:
- Checks required directories exist
- Validates configuration files
- Checks environment variables
- Tests logging system
- Verifies dependencies
"""

import os
import sys
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import Config
from src.logger import setup_logger
from src.utils.helpers import validate_api_keys


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_check(name, status, details=""):
    """Print validation check result."""
    symbol = "✓" if status else "✗"
    status_text = "PASS" if status else "FAIL"
    print(f"{symbol} {name}: {status_text}")
    if details:
        print(f"    {details}")


def validate_directories():
    """Validate that required directories exist."""
    print_header("Validating Directory Structure")
    
    required_dirs = [
        ("config/", "Configuration files"),
        ("data/", "Data storage"),
        ("models/", "ML models"),
        ("logs/", "Application logs"),
        ("src/", "Source code"),
        ("src/data/", "Data modules"),
        ("src/models/", "Model modules"),
        ("src/trading/", "Trading modules"),
        ("src/utils/", "Utility modules"),
        ("tests/", "Test suite"),
        ("scripts/", "Utility scripts"),
    ]
    
    all_passed = True
    
    for directory, description in required_dirs:
        exists = Path(directory).exists()
        print_check(f"{directory:<20} ({description})", exists)
        if not exists:
            all_passed = False
    
    return all_passed


def validate_config_files():
    """Validate configuration files exist and are valid."""
    print_header("Validating Configuration Files")
    
    all_passed = True
    
    # Check config.yaml
    config_file = Path("config/config.yaml")
    config_exists = config_file.exists()
    print_check("config/config.yaml exists", config_exists)
    
    if config_exists:
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Check for key sections
            required_sections = ['logging', 'trading', 'models', 'data', 'api']
            for section in required_sections:
                has_section = section in config_data
                print_check(f"  Section '{section}'", has_section)
                if not has_section:
                    all_passed = False
        
        except yaml.YAMLError as e:
            print_check("config.yaml YAML syntax", False, f"Parse error: {e}")
            all_passed = False
    else:
        all_passed = False
    
    # Check .env file
    env_file = Path(".env")
    env_exists = env_file.exists()
    print_check(".env file exists", env_exists, "Required for API keys")
    
    if not env_exists:
        all_passed = False
        print("    Hint: Copy .env.example to .env and fill in your API keys")
    
    return all_passed


def validate_environment_variables():
    """Validate environment variables are configured."""
    print_header("Validating Environment Variables")
    
    # Check if .env file is loaded
    from dotenv import load_dotenv
    load_dotenv()
    
    # Required variables
    recommended_vars = [
        ("LOG_LEVEL", "Logging level configuration"),
        ("ENVIRONMENT", "Runtime environment"),
    ]
    
    # API keys (at least one exchange should be configured)
    api_key_groups = {
        'Binance': ['BINANCE_API_KEY', 'BINANCE_API_SECRET'],
        'Coinbase': ['COINBASE_API_KEY', 'COINBASE_API_SECRET'],
        'Alpaca': ['ALPACA_API_KEY', 'ALPACA_API_SECRET'],
    }
    
    all_passed = True
    
    # Check recommended variables
    for var, description in recommended_vars:
        value = os.getenv(var)
        is_set = value is not None and value != ""
        print_check(f"{var:<25} ({description})", is_set)
    
    # Check API keys
    print("\nAPI Key Configuration:")
    api_status = validate_api_keys()
    
    any_configured = False
    for exchange, is_configured in api_status.items():
        print_check(f"  {exchange.capitalize():<20}", is_configured)
        if is_configured:
            any_configured = True
    
    if not any_configured:
        print("\n    Warning: No exchange API keys are configured.")
        print("    The bot will have limited functionality without API access.")
        print("    Add your API keys to the .env file.")
    
    return all_passed


def validate_dependencies():
    """Validate that required dependencies are installed."""
    print_header("Validating Dependencies")
    
    required_packages = [
        'dotenv',
        'yaml',
        'pandas',
        'numpy',
        'requests',
        'colorlog',
    ]
    
    all_passed = True
    
    for package in required_packages:
        try:
            __import__(package)
            print_check(f"Package: {package}", True)
        except ImportError:
            print_check(f"Package: {package}", False, "Not installed")
            all_passed = False
    
    if not all_passed:
        print("\n    Run: pip install -r requirements.txt")
    
    return all_passed


def test_configuration_loader():
    """Test that configuration loader works."""
    print_header("Testing Configuration Loader")
    
    try:
        config = Config()
        print_check("Config initialization", True)
        
        # Test getting values
        log_level = config.get('logging.level')
        print_check("Config value retrieval", log_level is not None, f"Log level: {log_level}")
        
        return True
    
    except Exception as e:
        print_check("Config initialization", False, str(e))
        return False


def test_logging_system():
    """Test that logging system works."""
    print_header("Testing Logging System")
    
    try:
        # Setup logger
        logger = setup_logger('validation_test', log_level='INFO')
        print_check("Logger initialization", True)
        
        # Test log message
        logger.info("Validation test message")
        print_check("Log message creation", True)
        
        # Check log file exists
        log_files = list(Path('logs').glob('*.log'))
        log_file_exists = len(log_files) > 0
        print_check("Log file created", log_file_exists)
        
        return True
    
    except Exception as e:
        print_check("Logging system", False, str(e))
        return False


def display_summary(results):
    """Display validation summary."""
    print_header("Validation Summary")
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"Total Checks: {total}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print()
    
    if failed == 0:
        print("🎉 All validation checks passed!")
        print("Your trading bot setup is complete and ready to use.")
        print()
        print("Run the bot with: python main.py")
    else:
        print("⚠️  Some validation checks failed.")
        print("Please review the issues above and fix them before running the bot.")
        return False
    
    return True


def main():
    """Main validation function."""
    print_header("AI Trading Bot - Setup Validation")
    
    # Run all validation checks
    results = {
        "Directory Structure": validate_directories(),
        "Configuration Files": validate_config_files(),
        "Environment Variables": validate_environment_variables(),
        "Dependencies": validate_dependencies(),
        "Configuration Loader": test_configuration_loader(),
        "Logging System": test_logging_system(),
    }
    
    # Display summary
    success = display_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

