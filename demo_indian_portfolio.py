from portfolio_optimizer import SectorPortfolioOptimizer
from datetime import datetime, timedelta

# Define Indian stocks and their sectors
indian_stocks = {
    # Technology
    'TCS.NS': 'Technology',
    'INFY.NS': 'Technology',
    'WIPRO.NS': 'Technology',
    'HCLTECH.NS': 'Technology',
    
    # Banking & Financial Services
    'HDFCBANK.NS': 'Banking',
    'ICICIBANK.NS': 'Banking',
    'SBIN.NS': 'Banking',
    'AXISBANK.NS': 'Banking',
    
    # Energy
    'RELIANCE.NS': 'Energy',
    'ONGC.NS': 'Energy',
    'POWERGRID.NS': 'Energy',
    'NTPC.NS': 'Energy',
    
    # Consumer Goods
    'HINDUNILVR.NS': 'Consumer',
    'ITC.NS': 'Consumer',
    'NESTLEIND.NS': 'Consumer',
    'BRITANNIA.NS': 'Consumer'
}

# Get unique sectors
sectors = list(set(indian_stocks.values()))

# Set sector constraints
sector_min_weights = {
    'Technology': 0.1,
    'Banking': 0.15,
    'Energy': 0.1,
    'Consumer': 0.1
}

sector_max_weights = {
    'Technology': 0.4,
    'Banking': 0.4,
    'Energy': 0.35,
    'Consumer': 0.35
}

def main():
    # Initialize optimizer
    optimizer = SectorPortfolioOptimizer(indian_stocks, sectors)
    
    # Set date range for analysis (last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    try:
        # Fetch data
        print("Fetching historical data...")
        optimizer.fetch_data(start_date, end_date)
        
        # Set sector constraints
        print("Setting sector constraints...")
        optimizer.set_sector_constraints(sector_min_weights, sector_max_weights)
        
        # Optimize portfolio
        print("\nOptimizing portfolio...")
        optimal_portfolio = optimizer.optimize_portfolio()
        
        # Export data for Tableau/Power BI
        print("\nExporting data for visualization tools...")
        export_path = optimizer.export_data_for_visualization(optimal_portfolio)
        print(f"Data exported to: {export_path}")
        
        # Plot results
        print("\nGenerating portfolio visualization...")
        optimizer.plot_portfolio_composition(optimal_portfolio)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()