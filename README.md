# Sector-Specific Portfolio Optimization for Indian Stock Markets

This project implements a portfolio optimization tool specifically designed for the Indian stock market, incorporating sector-specific constraints. It uses modern portfolio theory with the Ledoit-Wolf shrinkage estimator for robust covariance estimation and sector-based constraints for practical portfolio allocation.

## Features

- Sector-based portfolio optimization
- Real-time data fetching using yfinance
- Ledoit-Wolf shrinkage for robust covariance estimation
- Customizable sector constraints
- Interactive visualizations of portfolio composition
- Risk-adjusted return optimization (Sharpe Ratio)

## Installation

1. Clone the repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the demo script to see the optimizer in action with Indian stocks:
```bash
python demo_indian_portfolio.py
```

The demo includes:
- 16 major Indian stocks across 4 sectors
- Sector-specific allocation constraints
- Portfolio optimization for maximum Sharpe ratio
- Visualization of results

## Project Structure

- `portfolio_optimizer.py`: Main optimization engine
- `demo_indian_portfolio.py`: Demo script with Indian stocks
- `requirements.txt`: Required Python packages

## Methodology

The optimizer uses:
- Modern Portfolio Theory (MPT) framework
- Sharpe Ratio optimization
- Sector-based constraints
- Ledoit-Wolf shrinkage for covariance estimation

## Dependencies

- numpy: Numerical computations
- pandas: Data manipulation
- yfinance: Stock data fetching
- scipy: Optimization algorithms
- matplotlib: Visualization
- seaborn: Enhanced visualizations
- scikit-learn: Covariance estimation

## Note

This tool is for educational and research purposes only. Always conduct thorough research and consult financial advisors before making investment decisions.