import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

def get_stock_data():
    """Read stock data from local CSV files"""
    # Define file mappings
    file_mappings = {
        'AAPL': 'aapl data 2015-2025 - Sheet1.csv',
        'AMZN': 'AMZN data 2015-2025 - Sheet1.csv',
        'GOOG': 'GOOG data 2015-2025 - Sheet1.csv',
        'MSFT': 'msft data 2015-2025 - Sheet1.csv'
    }
    
    # Read and process each file
    data = {}
    for ticker, filename in file_mappings.items():
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        data[ticker] = df['Close']
    
    return pd.DataFrame(data)

def calculate_zscore(series, window=20):
    """Calculate rolling z-score of a series"""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / std

def find_correlated_pairs(data, min_correlation=0.7, lookback=60):
    """Find highly correlated pairs of stocks"""
    returns = data.pct_change()
    correlations = returns.rolling(window=lookback).corr()
    
    # Get all possible pairs
    pairs = []
    for i, j in combinations(data.columns, 2):
        # Calculate average correlation over the period
        try:
            avg_corr = correlations.loc[pd.IndexSlice[:, i], j].mean()
            if avg_corr > min_correlation:
                pairs.append((i, j, avg_corr))
        except:
            continue
    
    return sorted(pairs, key=lambda x: x[2], reverse=True)

def calculate_pair_spread(pair, data, window=20):
    """Calculate the spread between a pair of stocks"""
    stock1, stock2 = pair
    # Calculate the ratio of prices
    ratio = data[stock1] / data[stock2]
    # Calculate z-score of the ratio
    zscore = calculate_zscore(ratio, window)
    return zscore

def backtest_pairs_trading(data, window=20, entry_threshold=2.0, exit_threshold=0.0, min_correlation=0.7):
    """Backtest pairs trading strategy with mean reversion"""
    # Find correlated pairs
    print("Finding correlated pairs...")
    pairs = find_correlated_pairs(data, min_correlation)
    print(f"Found {len(pairs)} correlated pairs")
    
    # Calculate returns for each stock
    returns = data.pct_change()
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=data.index)
    results['Portfolio_Value'] = 1.0  # Start with $1
    
    # Initialize positions DataFrame
    positions = pd.DataFrame(0, index=data.index, columns=data.columns)
    
    # Calculate z-scores for individual stocks
    z_scores = pd.DataFrame(index=data.index)
    for column in data.columns:
        z_scores[column] = calculate_zscore(data[column], window)
    
    # Calculate pair spreads
    pair_spreads = {}
    for pair in pairs:
        stock1, stock2, corr = pair
        pair_spreads[pair] = calculate_pair_spread((stock1, stock2), data, window)
    
    for i in range(window, len(data)):
        # Find stocks with extreme z-scores for individual mean reversion
        current_z_scores = z_scores.iloc[i]
        
        # Long positions for stocks with low z-scores
        long_stocks = current_z_scores[current_z_scores < -entry_threshold].index
        # Short positions for stocks with high z-scores
        short_stocks = current_z_scores[current_z_scores > entry_threshold].index
        
        # Exit positions when z-score crosses zero
        exit_long = current_z_scores[(current_z_scores > exit_threshold) & (positions.iloc[i-1] > 0)].index
        exit_short = current_z_scores[(current_z_scores < -exit_threshold) & (positions.iloc[i-1] < 0)].index
        
        # Update individual stock positions
        positions.loc[data.index[i], long_stocks] = 1
        positions.loc[data.index[i], short_stocks] = -1
        positions.loc[data.index[i], exit_long] = 0
        positions.loc[data.index[i], exit_short] = 0
        
        # Check pair spreads for pairs trading opportunities
        for pair, spread in pair_spreads.items():
            stock1, stock2, corr = pair
            current_spread = spread.iloc[i]
            
            # If spread is too wide, short the expensive stock and long the cheap one
            if current_spread > entry_threshold:
                positions.loc[data.index[i], stock1] = -1  # Short the expensive stock
                positions.loc[data.index[i], stock2] = 1   # Long the cheap stock
            elif current_spread < -entry_threshold:
                positions.loc[data.index[i], stock1] = 1   # Long the cheap stock
                positions.loc[data.index[i], stock2] = -1  # Short the expensive stock
            elif abs(current_spread) < exit_threshold:
                # Exit pair positions if spread has normalized
                positions.loc[data.index[i], stock1] = 0
                positions.loc[data.index[i], stock2] = 0
        
        # Calculate daily returns
        daily_returns = (positions.iloc[i-1] * returns.iloc[i]).sum()
        results.loc[data.index[i], 'Portfolio_Value'] = results.loc[data.index[i-1], 'Portfolio_Value'] * (1 + daily_returns)
    
    # Calculate strategy metrics
    results['Returns'] = results['Portfolio_Value'].pct_change()
    results['Cumulative_Returns'] = results['Portfolio_Value']
    
    return results, positions, z_scores, pair_spreads

def plot_results(data, results, positions, z_scores, pair_spreads):
    """Plot strategy results"""
    # Plot portfolio value
    plt.figure(figsize=(15, 15))
    
    # Plot portfolio value
    plt.subplot(3, 1, 1)
    plt.plot(results.index, results['Portfolio_Value'], label='Strategy')
    plt.title('Portfolio Value')
    plt.legend()
    
    # Plot z-scores
    plt.subplot(3, 1, 2)
    for column in z_scores.columns:
        plt.plot(z_scores.index, z_scores[column], label=column, alpha=0.7)
    plt.axhline(y=2, color='r', linestyle='--', label='Entry Threshold')
    plt.axhline(y=-2, color='r', linestyle='--')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Individual Stock Z-Scores')
    plt.legend()
    
    # Plot pair spreads
    plt.subplot(3, 1, 3)
    for pair, spread in pair_spreads.items():
        stock1, stock2, corr = pair
        plt.plot(spread.index, spread, label=f'{stock1}/{stock2} (corr: {corr:.2f})', alpha=0.7)
    plt.axhline(y=2, color='r', linestyle='--', label='Entry Threshold')
    plt.axhline(y=-2, color='r', linestyle='--')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Pair Spread Z-Scores')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pairs_trading_results.png')
    plt.close()

def main():
    # Get data from local CSV files
    print("Reading stock data from CSV files...")
    data = get_stock_data()
    
    # Run backtest
    print("Running backtest...")
    results, positions, z_scores, pair_spreads = backtest_pairs_trading(data)
    
    # Calculate performance metrics
    total_return = results['Portfolio_Value'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(results)) - 1
    sharpe_ratio = np.sqrt(252) * results['Returns'].mean() / results['Returns'].std()
    
    # Calculate additional metrics
    max_drawdown = (results['Portfolio_Value'] / results['Portfolio_Value'].cummax() - 1).min()
    win_rate = (results['Returns'] > 0).mean()
    
    print(f"\nBacktest Results:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    
    # Plot results
    plot_results(data, results, positions, z_scores, pair_spreads)

if __name__ == "__main__":
    main()
