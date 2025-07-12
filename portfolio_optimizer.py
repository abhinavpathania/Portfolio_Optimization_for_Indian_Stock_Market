import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf

class SectorPortfolioOptimizer:
    def __init__(self, stocks_data, sectors):
        """
        Initialize the portfolio optimizer with stock data and sector information
        
        Parameters:
        stocks_data (dict): Dictionary with stock symbols as keys and their sector as values
        sectors (list): List of unique sectors
        """
        self.stocks = list(stocks_data.keys())
        self.sectors = sectors
        self.stock_sectors = stocks_data
        self.returns = None
        self.cov_matrix = None
        self.sector_constraints = None
        
    def fetch_data(self, start_date, end_date):
        """
        Fetch historical data for the stocks using yfinance
        """
        data = pd.DataFrame()
        for stock in self.stocks:
            try:
                ticker = yf.Ticker(stock)
                hist = ticker.history(start=start_date, end=end_date)['Close']
                data[stock] = hist
            except Exception as e:
                print(f"Error fetching data for {stock}: {e}")
        
        # Calculate daily returns
        self.returns = data.pct_change().dropna()
        
        # Calculate covariance matrix using Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        self.cov_matrix = pd.DataFrame(
            lw.fit(self.returns).covariance_,
            index=self.returns.columns,
            columns=self.returns.columns
        )
        
    def set_sector_constraints(self, min_weights, max_weights):
        """
        Set minimum and maximum weights for each sector
        
        Parameters:
        min_weights (dict): Minimum weights for each sector
        max_weights (dict): Maximum weights for each sector
        """
        self.sector_constraints = {
            'min': min_weights,
            'max': max_weights
        }
    
    def _get_portfolio_stats(self, weights):
        """
        Calculate portfolio statistics (returns and volatility)
        """
        portfolio_return = np.sum(self.returns.mean() * weights) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return portfolio_return, portfolio_vol
    
    def _objective_function(self, weights):
        """
        Objective function to minimize (negative Sharpe Ratio)
        """
        portfolio_return, portfolio_vol = self._get_portfolio_stats(weights)
        sharpe_ratio = portfolio_return / portfolio_vol
        return -sharpe_ratio
    
    def _get_sector_weights(self, weights):
        """
        Calculate total weights for each sector
        """
        sector_weights = {sector: 0 for sector in self.sectors}
        for stock, weight in zip(self.stocks, weights):
            sector = self.stock_sectors[stock]
            sector_weights[sector] += weight
        return sector_weights
    
    def optimize_portfolio(self):
        """
        Optimize the portfolio using sector constraints
        """
        n_assets = len(self.stocks)
        
        # Initial weights
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Add sector constraints
        if self.sector_constraints:
            for sector in self.sectors:
                sector_stocks_idx = [i for i, stock in enumerate(self.stocks)
                                   if self.stock_sectors[stock] == sector]
                
                # Minimum sector weight constraint
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=sector_stocks_idx:
                        np.sum(x[idx]) - self.sector_constraints['min'][sector]
                })
                
                # Maximum sector weight constraint
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=sector_stocks_idx:
                        self.sector_constraints['max'][sector] - np.sum(x[idx])
                })
        
        # Individual stock constraints (non-negative weights)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Optimize
        result = minimize(
            self._objective_function,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_vol = self._get_portfolio_stats(optimal_weights)
            sharpe_ratio = portfolio_return / portfolio_vol
            
            return {
                'weights': dict(zip(self.stocks, optimal_weights)),
                'sector_weights': self._get_sector_weights(optimal_weights),
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio
            }
        else:
            raise Exception("Optimization failed to converge")
    
    def plot_portfolio_composition(self, optimal_weights):
        """
        Create visualizations for portfolio composition
        """
        # Set style
        plt.style.use('seaborn-v0_8')  # Use a valid style name
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Stock Weights
        stock_weights = pd.Series(optimal_weights['weights'])
        stock_weights.sort_values(ascending=True).plot(kind='barh', ax=ax1)
        ax1.set_title('Individual Stock Weights')
        ax1.set_xlabel('Weight')
        
        # Plot 2: Sector Weights
        sector_weights = pd.Series(optimal_weights['sector_weights'])
        sector_weights.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
        ax2.set_title('Sector Allocation')
        
        plt.tight_layout()
        plt.show()
        
        # Print portfolio statistics
        print("\nPortfolio Statistics:")
        print(f"Expected Annual Return: {optimal_weights['portfolio_return']*100:.2f}%")
        print(f"Annual Volatility: {optimal_weights['portfolio_volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {optimal_weights['sharpe_ratio']:.2f}")