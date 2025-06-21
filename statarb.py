import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression

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

def test_cointegration(series1, series2, significance_level=0.05):
    """
    Test for cointegration between two series using Engle-Granger test
    Returns: (is_cointegrated, p_value, cointegration_vector)
    """
    # Remove any NaN values
    clean_data = pd.DataFrame({'series1': series1, 'series2': series2}).dropna()
    
    if len(clean_data) < 50:  # Need sufficient data for reliable test
        return False, 1.0, None
    
    # Engle-Granger cointegration test
    score, pvalue, _ = coint(clean_data['series1'], clean_data['series2'])
    
    # Calculate cointegration vector (beta)
    if pvalue < significance_level:
        # Estimate cointegration relationship
        model = LinearRegression()
        model.fit(clean_data['series2'].values.reshape(-1, 1), 
                 clean_data['series1'].values.reshape(-1, 1))
        beta = model.coef_[0][0]
        return True, pvalue, beta
    
    return False, pvalue, None

def calculate_zscore(series, window=20):
    """Calculate rolling z-score of a series"""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / std

def find_cointegrated_pairs(data, min_correlation=0.7, lookback=60, cointegration_window=252):
    """Find highly correlated and cointegrated pairs of stocks"""
    returns = data.pct_change()
    correlations = returns.rolling(window=lookback).corr()
    
    # Get all possible pairs
    pairs = []
    for i, j in combinations(data.columns, 2):
        # Calculate average correlation over the period
        try:
            avg_corr = correlations.loc[pd.IndexSlice[:, i], j].mean()
            if avg_corr > min_correlation:
                # Test for cointegration
                is_cointegrated, p_value, beta = test_cointegration(data[i], data[j])
                if is_cointegrated:
                    pairs.append((i, j, avg_corr, p_value, beta))
                else:
                    # Still include highly correlated pairs but mark as non-cointegrated
                    pairs.append((i, j, avg_corr, p_value, None))
        except:
            continue
    
    return sorted(pairs, key=lambda x: x[2], reverse=True)

def calculate_cointegration_spread(pair, data, window=252, min_window=60):
    """
    Calculate cointegration-based spread with rolling window
    """
    stock1, stock2, corr, p_value, beta = pair
    
    spread = pd.Series(index=data.index, dtype=float)
    betas = pd.Series(index=data.index, dtype=float)
    cointegration_status = pd.Series(index=data.index, dtype=bool)
    
    for i in range(min_window, len(data)):
        # Use rolling window for cointegration testing
        start_idx = max(0, i - window)
        window_data = data.iloc[start_idx:i+1]
        
        # Test for cointegration in the window
        is_cointegrated, _, rolling_beta = test_cointegration(
            window_data[stock1], window_data[stock2]
        )
        
        cointegration_status.iloc[i] = is_cointegrated
        
        if is_cointegrated and rolling_beta is not None:
            betas.iloc[i] = rolling_beta
            # Calculate spread: stock1 - beta * stock2
            spread.iloc[i] = data[stock1].iloc[i] - rolling_beta * data[stock2].iloc[i]
        else:
            # If not cointegrated, use simple price ratio as fallback
            betas.iloc[i] = data[stock1].iloc[i] / data[stock2].iloc[i]
            spread.iloc[i] = np.log(data[stock1].iloc[i] / data[stock2].iloc[i])
    
    return spread, betas, cointegration_status

def calculate_pair_spread(pair, data, window=20):
    """Calculate the spread between a pair of stocks"""
    stock1, stock2 = pair[0], pair[1]
    # Calculate the ratio of prices
    ratio = data[stock1] / data[stock2]
    # Calculate z-score of the ratio
    zscore = calculate_zscore(ratio, window)
    return zscore

def backtest_cointegration_statarb(data, window=20, entry_threshold=2.0, exit_threshold=0.0, 
                                  min_correlation=0.7, cointegration_window=252):
    """Backtest statistical arbitrage strategy with cointegration"""
    # Find correlated and cointegrated pairs
    print("Finding correlated and cointegrated pairs...")
    pairs = find_cointegrated_pairs(data, min_correlation, window, cointegration_window)
    
    cointegrated_pairs = [p for p in pairs if p[4] is not None]  # Has beta
    non_cointegrated_pairs = [p for p in pairs if p[4] is None]  # No beta
    
    print(f"Found {len(pairs)} correlated pairs")
    print(f"  - Cointegrated pairs: {len(cointegrated_pairs)}")
    print(f"  - Non-cointegrated pairs: {len(non_cointegrated_pairs)}")
    
    # Calculate returns for each stock
    returns = data.pct_change()
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=data.index)
    results['Portfolio_Value'] = 1.0  # Start with $1
    
    # Initialize positions DataFrame
    positions = pd.DataFrame(0.0, index=data.index, columns=data.columns)
    
    # Calculate z-scores for individual stocks
    z_scores = pd.DataFrame(index=data.index)
    for column in data.columns:
        z_scores[column] = calculate_zscore(data[column], window)
    
    # Calculate cointegration spreads for cointegrated pairs
    cointegration_spreads = {}
    cointegration_status = {}
    for pair in cointegrated_pairs:
        stock1, stock2 = pair[0], pair[1]
        spread, betas, status = calculate_cointegration_spread(pair, data, cointegration_window)
        cointegration_spreads[(stock1, stock2)] = {
            'spread': spread,
            'betas': betas,
            'status': status,
            'pair_info': pair
        }
        cointegration_status[(stock1, stock2)] = status
    
    # Calculate simple spreads for non-cointegrated pairs
    simple_spreads = {}
    for pair in non_cointegrated_pairs:
        stock1, stock2 = pair[0], pair[1]
        simple_spreads[(stock1, stock2)] = calculate_pair_spread(pair, data, window)
    
    # Track trades
    trades = []
    current_positions = {}
    
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
        
        # Check cointegration spreads for pairs trading opportunities
        for pair_key, spread_data in cointegration_spreads.items():
            stock1, stock2 = pair_key
            spread = spread_data['spread']
            betas = spread_data['betas']
            status = spread_data['status']
            pair_info = spread_data['pair_info']
            
            # Only trade when cointegrated
            if i < len(status) and status.iloc[i]:
                current_spread = spread.iloc[i]
                current_beta = betas.iloc[i]
                
                # Calculate z-score of the spread
                spread_zscore = calculate_zscore(spread.iloc[:i+1], window=60)
                current_spread_z = spread_zscore.iloc[-1] if len(spread_zscore) > 0 else 0
                
                # Entry signals based on spread z-score
                if current_spread_z > entry_threshold:
                    positions.loc[data.index[i], stock1] = -1  # Short the expensive stock
                    positions.loc[data.index[i], stock2] = current_beta  # Long the cheap stock
                    trades.append({
                        'date': data.index[i],
                        'type': f'short_{stock1}_long_{stock2}',
                        'pair': f'{stock1}/{stock2}',
                        'zscore': current_spread_z,
                        'method': 'cointegration'
                    })
                elif current_spread_z < -entry_threshold:
                    positions.loc[data.index[i], stock1] = 1  # Long the cheap stock
                    positions.loc[data.index[i], stock2] = -current_beta  # Short the expensive stock
                    trades.append({
                        'date': data.index[i],
                        'type': f'long_{stock1}_short_{stock2}',
                        'pair': f'{stock1}/{stock2}',
                        'zscore': current_spread_z,
                        'method': 'cointegration'
                    })
                elif abs(current_spread_z) < exit_threshold:
                    # Exit pair positions if spread has normalized
                    positions.loc[data.index[i], stock1] = 0
                    positions.loc[data.index[i], stock2] = 0
        
        # Check simple spreads for non-cointegrated pairs (less aggressive)
        for pair_key, spread in simple_spreads.items():
            stock1, stock2 = pair_key
            current_spread = spread.iloc[i]
            
            # Use higher thresholds for non-cointegrated pairs
            non_coint_threshold = entry_threshold * 1.5
            
            if current_spread > non_coint_threshold:
                positions.loc[data.index[i], stock1] = -1
                positions.loc[data.index[i], stock2] = 1
                trades.append({
                    'date': data.index[i],
                    'type': f'short_{stock1}_long_{stock2}',
                    'pair': f'{stock1}/{stock2}',
                    'zscore': current_spread,
                    'method': 'correlation'
                })
            elif current_spread < -non_coint_threshold:
                positions.loc[data.index[i], stock1] = 1
                positions.loc[data.index[i], stock2] = -1
                trades.append({
                    'date': data.index[i],
                    'type': f'long_{stock1}_short_{stock2}',
                    'pair': f'{stock1}/{stock2}',
                    'zscore': current_spread,
                    'method': 'correlation'
                })
        
        # Calculate daily returns
        daily_returns = (positions.iloc[i-1] * returns.iloc[i]).sum()
        results.loc[data.index[i], 'Portfolio_Value'] = results.loc[data.index[i-1], 'Portfolio_Value'] * (1 + daily_returns)
    
    # Calculate strategy metrics
    results['Returns'] = results['Portfolio_Value'].pct_change()
    results['Cumulative_Returns'] = results['Portfolio_Value']
    
    return results, positions, z_scores, cointegration_spreads, simple_spreads, trades

def plot_enhanced_results(data, results, positions, z_scores, cointegration_spreads, simple_spreads):
    """Plot enhanced strategy results with cointegration analysis"""
    plt.figure(figsize=(15, 20))
    
    # Plot portfolio value
    plt.subplot(5, 1, 1)
    plt.plot(results.index, results['Portfolio_Value'], label='Strategy', linewidth=2)
    plt.title('Portfolio Value - Cointegration-Based Statistical Arbitrage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot stock prices
    plt.subplot(5, 1, 2)
    for column in data.columns:
        plt.plot(data.index, data[column], label=column, alpha=0.7, linewidth=1.5)
    plt.title('Stock Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot individual z-scores
    plt.subplot(5, 1, 3)
    for column in z_scores.columns:
        plt.plot(z_scores.index, z_scores[column], label=column, alpha=0.7)
    plt.axhline(y=2, color='r', linestyle='--', label='Entry Threshold')
    plt.axhline(y=-2, color='r', linestyle='--')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Individual Stock Z-Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot cointegration spreads
    plt.subplot(5, 1, 4)
    for pair_key, spread_data in cointegration_spreads.items():
        stock1, stock2 = pair_key
        spread = spread_data['spread']
        pair_info = spread_data['pair_info']
        corr = pair_info[2]
        p_value = pair_info[3]
        plt.plot(spread.index, spread, 
                label=f'{stock1}/{stock2} (corr: {corr:.2f}, p: {p_value:.3f})', 
                alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Cointegration Spreads')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot simple spreads
    plt.subplot(5, 1, 5)
    for pair_key, spread in simple_spreads.items():
        stock1, stock2 = pair_key
        plt.plot(spread.index, spread, label=f'{stock1}/{stock2}', alpha=0.7)
    plt.axhline(y=2, color='r', linestyle='--', label='Entry Threshold')
    plt.axhline(y=-2, color='r', linestyle='--')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Simple Pair Spread Z-Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cointegration_statarb_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_trades(trades):
    """Analyze trading performance"""
    if not trades:
        return {}
    
    trade_analysis = {
        'total_trades': len(trades),
        'cointegration_trades': len([t for t in trades if t['method'] == 'cointegration']),
        'correlation_trades': len([t for t in trades if t['method'] == 'correlation']),
        'avg_zscore': np.mean([t['zscore'] for t in trades]),
        'unique_pairs': len(set([t['pair'] for t in trades]))
    }
    
    return trade_analysis

def main():
    # Get data from local CSV files
    print("Reading stock data from CSV files...")
    data = get_stock_data()
    
    # Test cointegration for all pairs
    print("\nTesting cointegration for all stock pairs...")
    all_pairs = list(combinations(data.columns, 2))
    cointegration_results = {}
    
    for stock1, stock2 in all_pairs:
        is_cointegrated, p_value, beta = test_cointegration(data[stock1], data[stock2])
        cointegration_results[(stock1, stock2)] = {
            'cointegrated': is_cointegrated,
            'p_value': p_value,
            'beta': beta
        }
        print(f"{stock1}/{stock2}: Cointegrated={is_cointegrated}, p-value={p_value:.4f}, beta={beta:.4f}" if beta else f"{stock1}/{stock2}: Cointegrated={is_cointegrated}, p-value={p_value:.4f}, beta=N/A")
    
    # Run enhanced backtest with lower correlation threshold
    print("\nRunning cointegration-based statistical arbitrage backtest...")
    results, positions, z_scores, cointegration_spreads, simple_spreads, trades = backtest_cointegration_statarb(
        data, min_correlation=0.5  # Lower threshold to capture more pairs
    )
    
    # Calculate performance metrics
    total_return = results['Portfolio_Value'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(results)) - 1
    sharpe_ratio = np.sqrt(252) * results['Returns'].mean() / results['Returns'].std()
    
    # Calculate additional metrics
    max_drawdown = (results['Portfolio_Value'] / results['Portfolio_Value'].cummax() - 1).min()
    win_rate = (results['Returns'] > 0).mean()
    volatility = results['Returns'].std() * np.sqrt(252)
    
    # Analyze trades
    trade_analysis = analyze_trades(trades)
    
    print(f"\n=== Cointegration-Based Statistical Arbitrage Results ===")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Annual Volatility: {volatility:.2%}")
    
    print(f"\n=== Trading Analysis ===")
    print(f"Total Trades: {trade_analysis.get('total_trades', 0)}")
    print(f"Cointegration Trades: {trade_analysis.get('cointegration_trades', 0)}")
    print(f"Correlation Trades: {trade_analysis.get('correlation_trades', 0)}")
    print(f"Average Z-Score: {trade_analysis.get('avg_zscore', 0):.2f}")
    print(f"Unique Pairs Traded: {trade_analysis.get('unique_pairs', 0)}")
    
    # Plot results
    plot_enhanced_results(data, results, positions, z_scores, cointegration_spreads, simple_spreads)

if __name__ == "__main__":
    main()
