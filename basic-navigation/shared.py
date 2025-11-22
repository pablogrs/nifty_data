from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("Loading dataset from shared.py...")
app_dir = Path(__file__).parent
markets = pd.read_csv(app_dir / "Datasets/markets_with_returns.csv")
print("Dataset loaded successfully.")

# List of market return columns to be used in methods
market_return_columns =['Nifty_Return', 'DowJones_Return', 'Nasdaq_Return',
        'HangSeng_Return', 'Nikkei_Return', 'DAX_Return']

# List market close columns
market_price_columns = ['Nifty_Close', 'DowJones_Close', 'Nasdaq_Close',
                       'HangSeng_Close', 'Nikkei_Close', 'DAX_Close']


tickers = {
    "^NSEI": "Nifty",
    "^DJI": "DowJones",
    "^IXIC": "Nasdaq",
    "^HSI": "HangSeng",
    "^N225": "Nikkei",
    "^GDAXI": "DAX",
    "^VIX": "VIX"
}

def plot_aoc_roc_curve(fpr, tpr, roc_auc, title='ROC Curve', model_name=''):
    """Generate AOC ROC curve for combined dataset."""
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    # # Print AUC
    dataset_type = "Train Dataset" if "Train" in title else "Test Dataset"
    print(f"{model_name} - {dataset_type}: {roc_auc:.2f}")

# define a method to plot confusion matrix
def plot_confusion_matrix(cm, cr, title='Confusion Matrix', model_name=''):
    # Plot confusion matrix heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Use model_name in the print statement
    dataset_type = "Train Dataset" if "Train" in title else "Test Dataset"
    print(f"Classification report - {model_name} - {dataset_type}:\n", cr)