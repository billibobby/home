from setuptools import setup, find_packages

# Core runtime requirements (excluding dev/test dependencies)
core_requirements = [
    'python-dotenv==1.0.0',
    'PyYAML==6.0.1',
    'colorlog==6.8.0',
    'pandas>=2.2.0',
    'numpy>=1.26.0',
    'requests==2.31.0',
    'python-dateutil==2.8.2',
    'keyring>=24.0.0',  # For secure credential storage
]

# GUI-specific dependencies
gui_requirements = [
    'PySide6>=6.5.0',
    'pyqtgraph>=0.13.0',  # For future charting/visualization
]

# Development and testing dependencies
dev_requirements = [
    'pytest==7.4.3',
    'pytest-mock==3.12.0',
    'pytest-cov>=4.1.0',
    'black>=23.0.0',
    'flake8>=6.0.0',
    'mypy>=1.5.0',
]

# Packaging dependencies (for building executables)
build_requirements = [
    'pyinstaller>=6.0.0',
]

setup(
    name='ai-trading-bot',
    version='0.1.0',
    description='AI-powered trading bot for cryptocurrency and stock markets',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/ai-trading-bot',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.9',
    install_requires=core_requirements,
    extras_require={
        'gui': gui_requirements,
        'dev': dev_requirements,
        'build': build_requirements,
        'all': gui_requirements + dev_requirements + build_requirements,
    },
    entry_points={
        'console_scripts': [
            'trading-bot=trading_bot.main:main',
            'trading-bot-gui=trading_bot.gui_main:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Office/Business :: Financial :: Investment',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='trading bot cryptocurrency stocks ai machine-learning',
    package_data={
        'trading_bot': ['resources/config/*.yaml'],
    },
    include_package_data=True,
)

