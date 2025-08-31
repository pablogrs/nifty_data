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

master_table.to_csv("four-markets.csv", index=False)