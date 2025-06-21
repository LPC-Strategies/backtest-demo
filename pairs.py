import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def get_stock_data():
    """Read KO and PEP data from local CSV files"""
    # Read and process each file
    KO_df = pd.read_csv('KO data 2015-2025 - Sheet1.csv')
    PEP_df = pd.read_csv('PEP data 2015-2025 - Sheet1.csv')
    
    # Convert dates and set as index
    KO_df['Date'] = pd.to_datetime(KO_df['Date']).dt.date
    PEP_df['Date'] = pd.to_datetime(PEP_df['Date']).dt.date
    KO_df.set_index('Date', inplace=True)
    PEP_df.set_index('Date', inplace=True)
    KO_df = KO_df[~KO_df.index.duplicated(keep='first')]
    PEP_df = PEP_df[~PEP_df.index.duplicated(keep='first')]
    # Create DataFrame with both stocks
    data = pd.DataFrame({
        'KO': KO_df['Close'],
        'PEP': PEP_df['Close']
    })
    data = data.dropna()
    
    return data

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

def calculate_cointegration_spread(data, window=252, min_window=60):
    """
    Calculate cointegration-based spread with rolling window
    Uses Engle-Granger test to ensure cointegration relationship
    """
    spread = pd.Series(index=data.index, dtype=float)
    betas = pd.Series(index=data.index, dtype=float)
    cointegration_status = pd.Series(index=data.index, dtype=bool)
    
    for i in range(min_window, len(data)):
        # Use rolling window for cointegration testing
        start_idx = max(0, i - window)
        window_data = data.iloc[start_idx:i+1]
        
        # Test for cointegration in the window
        is_cointegrated, p_value, beta = test_cointegration(
            window_data['KO'], window_data['PEP']
        )
        
        cointegration_status.iloc[i] = is_cointegrated
        
        if is_cointegrated and beta is not None:
            betas.iloc[i] = beta
            # Calculate spread: KO - beta * PEP
            spread.iloc[i] = data['KO'].iloc[i] - beta * data['PEP'].iloc[i]
        else:
            # If not cointegrated, use simple price ratio as fallback
            betas.iloc[i] = data['KO'].iloc[i] / data['PEP'].iloc[i]
            spread.iloc[i] = np.log(data['KO'].iloc[i] / data['PEP'].iloc[i])
    
    return spread, betas, cointegration_status

def calculate_zscore(series, window=20):
    """Calculate rolling z-score of a series"""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / std

def calculate_optimal_thresholds(spread, window=252):
    """
    Calculate optimal entry and exit thresholds based on spread volatility
    """
    rolling_std = spread.rolling(window=window).std()
    
    # Dynamic thresholds based on volatility
    entry_threshold = 2.0  # Base threshold
    exit_threshold = 0.5   # Base exit threshold
    
    # Adjust thresholds based on recent volatility
    if len(rolling_std) > 0:
        recent_vol = rolling_std.iloc[-1]
        mean_vol = rolling_std.mean()
        if not np.isnan(recent_vol) and not np.isnan(mean_vol) and recent_vol > 0 and mean_vol > 0:
            # Scale thresholds by volatility
            vol_ratio = recent_vol / mean_vol
            entry_threshold = max(1.5, min(3.0, 2.0 * vol_ratio))
            exit_threshold = max(0.3, min(1.0, 0.5 * vol_ratio))
    
    return entry_threshold, exit_threshold

def backtest_cointegration_pairs_trading(data, window=252, min_window=60):
    """Backtest cointegration-based pairs trading strategy for KO/PEP"""
    returns = data.pct_change()
    results = pd.DataFrame(index=data.index)
    results['Portfolio_Value'] = 1.0
    positions = pd.DataFrame(0.0, index=data.index, columns=data.columns, dtype=float)
    
    # Calculate cointegration-based spread
    spread, betas, cointegration_status = calculate_cointegration_spread(data, window, min_window)
    
    # Calculate z-score of the spread
    zscore = calculate_zscore(spread, window=60)
    
    # Track strategy performance
    trades = []
    current_position = 0
    
    for i in range(min_window, len(data)):
        current_z = zscore.iloc[i]
        is_cointegrated = cointegration_status.iloc[i]
        
        # Only trade when cointegrated
        if is_cointegrated and not np.isnan(current_z):
            # Dynamic thresholds
            entry_threshold, exit_threshold = calculate_optimal_thresholds(spread.iloc[:i+1], window=60)
            
            # Entry signals
            if current_z > entry_threshold and current_position <= 0:
                positions.loc[data.index[i], 'KO'] = -1
                positions.loc[data.index[i], 'PEP'] = betas.iloc[i]
                current_position = 1
                trades.append({
                    'date': data.index[i],
                    'type': 'short_ko_long_pep',
                    'zscore': current_z,
                    'threshold': entry_threshold
                })
            elif current_z < -entry_threshold and current_position >= 0:
                positions.loc[data.index[i], 'KO'] = 1
                positions.loc[data.index[i], 'PEP'] = -betas.iloc[i]
                current_position = -1
                trades.append({
                    'date': data.index[i],
                    'type': 'long_ko_short_pep',
                    'zscore': current_z,
                    'threshold': -entry_threshold
                })
            # Exit signals
            elif abs(current_z) < exit_threshold and current_position != 0:
                positions.loc[data.index[i], 'KO'] = 0
                positions.loc[data.index[i], 'PEP'] = 0
                current_position = 0
                trades.append({
                    'date': data.index[i],
                    'type': 'exit',
                    'zscore': current_z,
                    'threshold': exit_threshold
                })
            else:
                # Maintain current position
                positions.loc[data.index[i]] = positions.iloc[i-1]
        else:
            # No position when not cointegrated
            positions.loc[data.index[i], 'KO'] = 0
            positions.loc[data.index[i], 'PEP'] = 0
            current_position = 0
        
        # Calculate daily returns
        if i > 0:
            daily_returns = (positions.iloc[i-1] * returns.iloc[i]).sum()
            results.loc[data.index[i], 'Portfolio_Value'] = results.loc[data.index[i-1], 'Portfolio_Value'] * (1 + daily_returns)
    
    results['Returns'] = results['Portfolio_Value'].pct_change()
    results['Cumulative_Returns'] = results['Portfolio_Value']
    
    return results, positions, zscore, spread, cointegration_status, trades

def plot_enhanced_results(data, results, positions, zscore, spread, cointegration_status, entry_threshold=2.0, exit_threshold=0.5):
    """Plot enhanced strategy results with cointegration analysis"""
    plt.figure(figsize=(15, 20))
    
    # Plot portfolio value
    plt.subplot(5, 1, 1)
    plt.plot(results.index, results['Portfolio_Value'], label='Strategy', linewidth=2)
    plt.title('Portfolio Value - Cointegration-Based Pairs Trading')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot stock prices
    plt.subplot(5, 1, 2)
    plt.plot(data.index, data['KO'], label='KO', alpha=0.7, linewidth=1.5)
    plt.plot(data.index, data['PEP'], label='PEP', alpha=0.7, linewidth=1.5)
    plt.title('Stock Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot cointegration spread
    plt.subplot(5, 1, 3)
    plt.plot(spread.index, spread, label='Cointegration Spread', color='purple', alpha=0.8)
    plt.title('KO/PEP Cointegration Spread')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot spread z-score with cointegration status
    plt.subplot(5, 1, 4)
    plt.plot(zscore.index, zscore, label='Spread Z-Score', alpha=0.7, linewidth=1.5)
    
    # Highlight cointegrated periods
    cointegrated_periods = cointegration_status[cointegration_status == True]
    if len(cointegrated_periods) > 0:
        plt.scatter(cointegrated_periods.index, zscore[cointegrated_periods.index], 
                   color='green', alpha=0.6, s=10, label='Cointegrated')
    
    plt.axhline(y=entry_threshold, color='r', linestyle='--', label='Entry Threshold')
    plt.axhline(y=-entry_threshold, color='r', linestyle='--')
    plt.axhline(y=exit_threshold, color='g', linestyle='--', label='Exit Threshold')
    plt.axhline(y=-exit_threshold, color='g', linestyle='--')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('KO/PEP Cointegration Spread Z-Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot cointegration status
    plt.subplot(5, 1, 5)
    plt.plot(cointegration_status.index, cointegration_status.astype(int), 
             label='Cointegration Status', color='orange', linewidth=2)
    plt.title('Cointegration Status (1=Cointegrated, 0=Not Cointegrated)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cointegration_pairs_trading_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_trades(trades, data):
    """Analyze trading performance"""
    if not trades:
        return {}
    
    trade_analysis = {
        'total_trades': len(trades),
        'long_ko_trades': len([t for t in trades if t['type'] == 'long_ko_short_pep']),
        'short_ko_trades': len([t for t in trades if t['type'] == 'short_ko_long_pep']),
        'exit_trades': len([t for t in trades if t['type'] == 'exit']),
        'avg_zscore_entry': np.mean([t['zscore'] for t in trades if t['type'] != 'exit']),
        'avg_threshold': np.mean([t['threshold'] for t in trades])
    }
    
    return trade_analysis

def main():
    # Get data from local CSV files
    print("Reading stock data from CSV files...")
    data = get_stock_data()
    
    # Test initial cointegration
    print("\nTesting cointegration between KO and PEP...")
    is_cointegrated, p_value, beta = test_cointegration(data['KO'], data['PEP'])
    print(f"Overall cointegration test:")
    print(f"  Cointegrated: {is_cointegrated}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Beta (KO = β * PEP): {beta:.4f}" if beta else "  Beta: N/A")
    
    # Calculate correlation
    returns = data.pct_change()
    correlation = returns['KO'].corr(returns['PEP'])
    print(f"\nCorrelation between KO and PEP returns: {correlation:.4f}")
    
    # Strategy parameters
    window = 252  # One year for cointegration testing
    min_window = 60  # Minimum window for initial calculations
    
    # Run enhanced backtest
    print("\nRunning cointegration-based backtest...")
    results, positions, zscore, spread, cointegration_status, trades = backtest_cointegration_pairs_trading(
        data, window, min_window
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
    trade_analysis = analyze_trades(trades, data)
    
    print(f"\n=== Cointegration-Based Pairs Trading Results ===")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Annual Volatility: {volatility:.2%}")
    
    print(f"\n=== Trading Analysis ===")
    print(f"Total Trades: {trade_analysis.get('total_trades', 0)}")
    print(f"Long KO/Short PEP: {trade_analysis.get('long_ko_trades', 0)}")
    print(f"Short KO/Long PEP: {trade_analysis.get('short_ko_trades', 0)}")
    print(f"Exit Trades: {trade_analysis.get('exit_trades', 0)}")
    print(f"Average Entry Z-Score: {trade_analysis.get('avg_zscore_entry', 0):.2f}")
    print(f"Average Threshold: {trade_analysis.get('avg_threshold', 0):.2f}")
    
    # Cointegration statistics
    cointegration_rate = cointegration_status.mean()
    print(f"\n=== Cointegration Statistics ===")
    print(f"Cointegration Rate: {cointegration_rate:.2%}")
    print(f"Trading Days: {len(cointegration_status)}")
    print(f"Cointegrated Days: {cointegration_status.sum()}")
    
    # Plot results
    plot_enhanced_results(data, results, positions, zscore, spread, cointegration_status)

if __name__ == "__main__":
    main()
