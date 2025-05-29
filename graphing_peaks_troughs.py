import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Download data
oil_df = yf.download('CL=F', start='2000-08-01')
vix_df = yf.download('^VIX', start='2000-08-01')

# Step 2: Extract Close prices and combine
df = pd.concat([oil_df['Close'], vix_df['Close']], axis=1)
df.columns = ['oil', 'vix']
df.dropna(inplace=True)

# Step 3: Calculate returns and z-scores
df['oil_return'] = df['oil'].pct_change()
df['vix_return'] = df['vix'].pct_change()

# 30-day rolling z-score of oil returns
df['zscore'] = (df['oil_return'] - df['oil_return'].rolling(30).mean()) / df['oil_return'].rolling(30).std()

# Identify spike events (z-score > 2)
df['is_peak'] = df['zscore'] > 2
df['is_trough'] = df['zscore'] < -2

# Step 4: Create lagged VIX return columns
max_lag = 90
for lag in range(0, max_lag + 1):
    df[f'vix_lag_{lag}'] = df['vix_return'].shift(-lag)

# Step 5: Subset to only oil spike days
spike_df = df[df['is_peak']].copy() # <--------------------------- edit this line to use 'is_trough' for trough
vix_lag_cols = [f'vix_lag_{lag}' for lag in range(0, max_lag + 1)]
spike_vix_lags = spike_df[vix_lag_cols]

# Step 6: Compute average VIX return for each lag
mean_vix_returns = spike_vix_lags.mean()

# Step 7: Plot results
plt.figure(figsize=(10, 6))
mean_vix_returns.plot()
plt.title('Average VIX Return Following Oil Price Peaks (z-score > 2)') # <------------- edit title to reflect plot
plt.xlabel('Lag (days after oil spike)')
plt.ylabel('Mean VIX Return')
plt.axhline(0, color='gray', linestyle='--')

x_values = [81,76,13,71] # <------------------------------------------ edit x values to reflect plot
for x in x_values:
    plt.axvline(x=x, color='red', linestyle='--')

plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('peaks_vix_returns_plot.png', dpi=300) # <-------------- edit this line to use 'trough' for trough

plt.show()