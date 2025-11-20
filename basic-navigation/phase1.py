import shared
import statsmodels.api as sm
import matplotlib.pyplot as plt


def head_of_master_table():
    return shared.markets.head()

def data_collection_and_preprocessing_code():
    return '''
import yfinance as yf
import pandas as pd
import numpy as np

# Define tickers and friendly names
tickers = {
    "^NSEI": "Nifty",
    "^DJI": "DowJones",
    "^IXIC": "Nasdaq",
    "^HSI": "HangSeng",
    "^N225": "Nikkei",
    "^GDAXI": "DAX",
    "^VIX": "VIX"
}

start_date = "2019-01-01"
end_date = "2025-04-30"

# Download and keep only Open and Close
data_frames = {}
for ticker, name in tickers.items():
    data = yf.download(ticker, start=start_date, end=end_date)[['Open', 'Close']]
    data.columns = [f"{name}_Open", f"{name}_Close"]
    data_frames[name] = data

# Get full date range including all days (weekends and holidays included)
full_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Reindex each DataFrame to this full range and interpolate missing Open/Close
for name, df in data_frames.items():
    df = df.reindex(full_range)  # this introduces NaNs for missing dates
    df[f"{name}_Open"] = df[f"{name}_Open"].interpolate(method='linear', limit_direction='both')
    df[f"{name}_Close"] = df[f"{name}_Close"].interpolate(method='linear', limit_direction='both')
    data_frames[name] = df

# Merge all into one master table
master_table = pd.concat(data_frames.values(), axis=1)

# Optional: Reset index and rename
master_table.index.name = "Date"
master_table.reset_index(inplace=True)

print(master_table.head())

master_table.to_csv("markets.csv", index=False)
'''

def qq_plots_for_returns():
    # Create a figure with 2 rows and 3 columns for 5 plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Counter for subplot position
    plot_idx = 0

    for ticker in shared.tickers.values():
        if ticker != "VIX":
            # Create QQ plot in the specific subplot
            sm.graphics.qqplot(
                shared.markets[f"{ticker}_Return"],
                line='45',
                fit=True,
                ax=axes[plot_idx ]
            )
            axes[plot_idx].set_title(f"QQ Plot for {ticker} Returns")
            plot_idx += 1

    # # Hide the 6th subplot (we only have 5 plots)
    # axes[5].set_visible(False)

    plt.tight_layout()
    return fig

def qq_plots_code():
    return '''
    # Create a figure with 2 rows and 3 columns for 5 plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Counter for subplot position
    plot_idx = 0

    for ticker in tickers.values():
        if ticker != "VIX":
            # Create QQ plot in the specific subplot
            sm.graphics.qqplot(
                shared.markets[f"{ticker}_Return"],
                line='45',
                fit=True,
                ax=axes[plot_idx]
            )
            axes[plot_idx].set_title(f"QQ Plot for {ticker} Returns")
            plot_idx += 1

    # Hide the 6th subplot (we only have 5 plots)
    axes[5].set_visible(False)

    plt.tight_layout()'''

def consolidate_columns_code():
    return '''
markets["Quarter"] = pd.to_datetime(markets["Date"]).dt.quarter
markets["Month"] = pd.to_datetime(markets["Date"]).dt.month
markets["Year"] = pd.to_datetime(markets["Date"]).dt.year
markets["dayofweek"] = pd.to_datetime(markets["Date"]).dt.dayofweek
markets["day"] = pd.to_datetime(markets["Date"]).dt.day

# Remove weekends from the dataset
markets = markets[markets.dayofweek < 5]
# Remove the first of Jan of each year
markets = markets[~((markets["Month"] == 1) & (markets["day"] == 1))]

markets.drop(columns=["dayofweek", "day"], inplace=True)
markets.reset_index(inplace=True, drop=True, names="Date")

print(markets.info())

markets.to_csv("markets_with_returns.csv", index=False)
'''

def normality_tests():
    """Perform Shapiro-Wilk and Lilliefors tests for normality on returns"""
    from scipy.stats import shapiro
    from statsmodels.stats.diagnostic import lilliefors

    results = []

    # Shapiro-Wilk test for normality
    results.append("=== SHAPIRO-WILK TEST ===\n")
    for ticker in shared.tickers.values():
        stat, p = shapiro(shared.markets[f"{ticker}_Return"])
        result = "follows" if p > 0.05 else "does not follow"
        results.append(f"{ticker}_Return: Statistics={stat:.4f}, p-value={p:.4f}")
        results.append(f"  → {result} a normal distribution\n")

    results.append("\n=== LILLIEFORS TEST ===\n")
    # Lilliefors test for normality
    for ticker in shared.tickers.values():
        stat, p = lilliefors(shared.markets[f"{ticker}_Return"])
        result = "follows" if p > 0.05 else "does not follow"
        results.append(f"{ticker}_Return: Statistics={stat:.4f}, p-value={p:.4f}")
        results.append(f"  → {result} a normal distribution\n")

    return "\n".join(results)

def normality_tests_code():
    """Return the code for normality tests"""
    return '''
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import lilliefors

# Shapiro-Wilk test for normality
for ticker in tickers.values():
    stat, p = shapiro(markets[f"{ticker}_Return"])
    print(f"Shapiro-Wilk Test for {ticker}_Return: Statistics={stat}, p-value={p}")
    if p > 0.05:
        print(f"{ticker}_Return follows a normal distribution (fail to reject H0)")
    else:
        print(f"{ticker}_Return does not follow a normal distribution (reject H0)")

# Lilliefors test for normality
for ticker in tickers.values():
    stat, p = lilliefors(markets[f"{ticker}_Return"])
    print(f"Lilliefors Test for {ticker}_Return: Statistics={stat}, p-value={p}")
    if p > 0.05:
        print(f"{ticker}_Return follows a normal distribution (fail to reject H0)")
    else:
        print(f"{ticker}_Return does not follow a normal distribution (reject H0)")
'''