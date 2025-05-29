import yfinance as yf
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp

# Download data
oil_df = yf.download('CL=F', start='2000-08-01')
vix_df = yf.download('^VIX', start='2000-08-01')

# Combine and clean
df = pd.concat([oil_df['Close'], vix_df['Close']], axis=1)
df.columns = ['oil', 'vix']
df.dropna(inplace=True)

# Calculate returns
df['oil_return'] = df['oil'].pct_change()
df['vix_return'] = df['vix'].pct_change()

print(df)

# Identify oil spike events (z-score > 2)
df['zscore'] = (df['oil_return'] - df['oil_return'].rolling(30).mean()) / df['oil_return'].rolling(30).std()
print(df['zscore'].dropna())  # Check z-score statistics
df['is_peak'] = df['zscore'] > 2
df['is_trough'] = df['zscore'] < -2

# Prepare to collect p-values and statistics
results = []

# Loop through lag days
for lag in range(0, 91):
    df[f'vix_lag_{lag}'] = df['vix_return'].shift(-lag)
    peak_returns = df[df['is_peak']][f'vix_lag_{lag}'].dropna()
    trough_returns = df[df['is_trough']][f'vix_lag_{lag}'].dropna()

    if len(peak_returns) >= 10:  # Only test if we have enough data
        t_stat_peak, p_val_peak = ttest_1samp(peak_returns, 0)
        t_stat_trough, p_val_trough = ttest_1samp(trough_returns, 0)

        results.append({
            'lag': lag,
            'number of peak events': len(peak_returns),
            'mean of peak events': peak_returns.mean(),
            'std of peak events': peak_returns.std(),
            't_stat of peak events': t_stat_peak,
            'p_val of peak events': p_val_peak,
            'number of trough events': len(trough_returns),
            'mean of trough events': trough_returns.mean(),
            'std of trough events': trough_returns.std(),
            't_stat of trough events': t_stat_trough,
            'p_val of trough events': p_val_trough
        })

# Convert to DataFrame and display
results_df = pd.DataFrame(results)
# print(results_df)

# Optionally export
results_df.to_csv("vix_lag_pvalues_paired.csv", index=False)
