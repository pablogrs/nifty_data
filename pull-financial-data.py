import yfinance as yf
import pandas as pd

# Download data for Nifty 50 index
ticker = "^VIX"
nifty_data = yf.download(ticker, start="2019-01-01", end="2025-04-30")

# Keep only Open and Close columns
nifty_oc = nifty_data[['Open', 'Close']]


# Save to CSV
nifty_oc.to_csv("nifty_oc.csv", index_label="Date")

# Clean up columns and save to CSV again
data = pd.read_csv("nifty_oc.csv", header=None, names=["Date", "Open", "Close"], skiprows=3, parse_dates=True)

data.to_csv("nifty_oc.csv", index=False)
