from pathlib import Path
import pandas as pd


print("Loading dataset from shared.py...")
app_dir = Path(__file__).parent
markets = pd.read_csv(app_dir / "Datasets/markets_with_returns.csv")
print("Dataset loaded successfully.")

# List of market return columns to be used in methods
market_return_columns =['Nifty_Return', 'DowJones_Return', 'Nasdaq_Return',
        'HangSeng_Return', 'Nikkei_Return']
