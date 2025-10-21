import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shared

# Prepare data for boxplot: melt the dataframe
# id_vars=['Year']: Columns to keep as identifiers (not melted)
# value_vars=market_return_columns: Columns to melt into rows
# var_name='Market': Name for the new column containing the original column names
# value_name='Return': Name for the new column containing the values
melted_returns = shared.markets.melt(id_vars=['Year'], value_vars=shared.market_return_columns,
                var_name='Market', value_name='Return')
# Rename 'Market' values to remove '_Return' suffix
melted_returns['Market'] = melted_returns['Market'].apply(lambda x: x.split('_')[0])

def boxplot_by_year():
    # Box plot of all markets per year, with market names only (no '_Return')
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Year', y='Return', hue='Market', data=melted_returns)
    plt.title('Box Plot of Market Returns by Year')
    plt.xlabel('Year')
    plt.ylabel('Daily Return')
    plt.legend(title='Market')
    plt.tight_layout()

def boxplot_code():
    return """
        # Box plot of all markets per year
        import matplotlib.pyplot as plt
        import seaborn as sns

        melted_returns = shared.markets.melt(id_vars=['Year'], value_vars=market_return_columns,
                        var_name='Market', value_name='Return')
        melted_returns['Market'] = melted_returns['Market'].apply(lambda x: x.split('_')[0])

        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Year', y='Return', hue='Market', data=melted_returns)
        plt.title('Box Plot of Market Returns by Year')
        plt.xlabel('Year')
        plt.ylabel('Daily Return')
        plt.legend(title='Market')
        plt.tight_layout()
        plt.show()
                        """

def table_of_returns_by_year():
    # Table of daily returns by YEAR (n, mean, std. deviation), grouped by year, market names cleaned
    returns_stats = melted_returns.groupby(['Year', 'Market'])['Return'].agg(['count', 'mean', 'std']).reset_index()
    returns_stats.columns = ['Year', 'Market', 'n', 'Mean', 'StdDev']

    # Create a comprehensive statistics table with multi-level columns
    stats_pivot = returns_stats.pivot(index='Market', columns='Year', values=['n', 'Mean', 'StdDev'])
    stats_pivot = stats_pivot.reorder_levels([1, 0], axis=1).sort_index(axis=1)

    # Format display columns to 2 decimal places
    numeric_cols = [col for col in stats_pivot.columns if col[1] in ['Mean', 'StdDev']]
    stats_pivot[numeric_cols] = stats_pivot[numeric_cols].round(2)

    # Reset index to make Market a regular column
    stats_pivot = stats_pivot.reset_index()

    return stats_pivot



def table_of_returns_by_year_code():
    return """
        # Table of daily returns by YEAR (n, mean, std. deviation), grouped by year, market names cleaned
        returns_stats = melted_returns.groupby(['Year', 'Market'])['Return'].agg(['count', 'mean', 'std']).reset_index()
        returns_stats.columns = ['Year', 'Market', 'n', 'Mean', 'StdDev']

        # Create a comprehensive statistics table with multi-level columns
        stats_pivot = returns_stats.pivot(index='Market', columns='Year', values=['n', 'Mean', 'StdDev'])
        stats_pivot = stats_pivot.reorder_levels([1, 0], axis=1).sort_index(axis=1)

        # Format display columns to 2 decimal places
        numeric_cols = [col for col in stats_pivot.columns if col[1] in ['Mean', 'StdDev']]
        stats_pivot[numeric_cols] = stats_pivot[numeric_cols].round(2)

        display(stats_pivot)
    """

def bar_plot_median_by_year():
    # Bar Plot of median daily return for all markets per year
    median_returns = melted_returns.groupby(['Year', 'Market'])['Return'].median().reset_index()
    pivot_median = median_returns.pivot(index='Year', columns='Market', values='Return')

    pivot_median.plot(kind='bar')
    plt.title('Median Daily Return by Market and Year')
    plt.xlabel('Year')
    plt.ylabel('Median Daily Return')
    plt.legend(title='Market')
    plt.tight_layout()

def bar_plot_median_by_year_code():
    return """
        # Bar Plot of median daily return for all markets per year
        median_returns = melted_returns.groupby(['Year', 'Market'])['Return'].median().reset_index()
        pivot_median = median_returns.pivot(index='Year', columns='Market', values='Return')

        pivot_median.plot(kind='bar', figsize=(14, 8))
        plt.title('Median Daily Return by Market and Year')
        plt.xlabel('Year')
        plt.ylabel('Median Daily Return')
        plt.legend(title='Market')
        plt.tight_layout()
        plt.show()
    """

def heat_map_by_year_and_quarter():
    # Heatmap of median returns for all markets by YEAR and QUARTER (combined)
    melted_quarter = shared.markets.melt(id_vars=['Year', 'Quarter'], value_vars=shared.market_return_columns,
                            var_name='Market', value_name='Return')
    melted_quarter['Market'] = melted_quarter['Market'].apply(lambda x: x.split('_')[0])

    # Create Year-Quarter combination for better labeling
    melted_quarter['Year_Quarter'] = melted_quarter['Year'].astype(str) + '-Q' + melted_quarter['Quarter'].astype(str)

    # Calculate median returns by Market and Year-Quarter
    median_year_quarter = melted_quarter.groupby(['Year_Quarter', 'Market'])['Return'].median().reset_index()
    pivot_year_quarter = median_year_quarter.pivot(index='Market', columns='Year_Quarter', values='Return')

    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_year_quarter, annot=True, fmt=".3f", cmap='coolwarm',
                cbar_kws={'label': 'Median Daily Return'})
    plt.title('Median Daily Return Heatmap (Markets vs Year-Quarter)')
    plt.xlabel('Year-Quarter')
    plt.ylabel('Market')
    plt.xticks(rotation=45)
    plt.tight_layout()

def heat_map_by_year_and_quarter_code():
    return """
        # Heatmap of median returns for all markets by YEAR and QUARTER (combined)
        melted_quarter = shared.markets.melt(id_vars=['Year', 'Quarter'], value_vars=market_return_columns,
                                var_name='Market', value_name='Return')
        melted_quarter['Market'] = melted_quarter['Market'].apply(lambda x: x.split('_')[0])

        # Create Year-Quarter combination for better labeling
        melted_quarter['Year_Quarter'] = melted_quarter['Year'].astype(str) + '-Q' + melted_quarter['Quarter'].astype(str)

        # Calculate median returns by Market and Year-Quarter
        median_year_quarter = melted_quarter.groupby(['Year_Quarter', 'Market'])['Return'].median().reset_index()
        pivot_year_quarter = median_year_quarter.pivot(index='Market', columns='Year_Quarter', values='Return')

        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot_year_quarter, annot=True, fmt=".3f", cmap='coolwarm',
                    cbar_kws={'label': 'Median Daily Return'})
        plt.title('Median Daily Return Heatmap (Markets vs Year-Quarter)')
        plt.xlabel('Year-Quarter')
        plt.ylabel('Market')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        """

def correlation_matrix():
    # Create correlation matrix for all 6 years of data
    correlation_6years = shared.markets[shared.market_return_columns].corr()

    # Clean up column and index names (remove '_Return' suffix)
    correlation_6years.columns = [col.split('_')[0] for col in correlation_6years.columns]
    correlation_6years.index = [idx.split('_')[0] for idx in correlation_6years.index]

    # Filter data for 2024 only
    markets_2024 = shared.markets[shared.markets['Year'] == 2024]

    # Create correlation matrix for 2024 data
    correlation_2024 = markets_2024[shared.market_return_columns].corr()

    # Clean up column and index names (remove '_Return' suffix)
    correlation_2024.columns = [col.split('_')[0] for col in correlation_2024.columns]
    correlation_2024.index = [idx.split('_')[0] for idx in correlation_2024.index]


    # Create subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))


    # Display correlation matrix for all 6 years of data
    sns.heatmap(correlation_6years, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, ax=ax1)
    ax1.set_title('Correlation Matrix - 6 Years Daily Returns (All Markets)')

    #  Display correlation matrix for 2024 as heatmap
    sns.heatmap(correlation_2024, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, ax=ax2, cbar_kws={'label': 'Correlation Coefficient'})
    ax2.set_title('Correlation Matrix - 2024 Daily Returns (All Markets)')

    plt.tight_layout()

def correlation_matrix_code():
    return """
        # Create correlation matrix for all 6 years of data
        correlation_6years = shared.markets[market_return_columns].corr()

        # Clean up column and index names (remove '_Return' suffix)
        correlation_6years.columns = [col.split('_')[0] for col in correlation_6years.columns]
        correlation_6years.index = [idx.split('_')[0] for idx in correlation_6years.index]

        # Filter data for 2024 only
        markets_2024 = shared.markets[shared.markets['Year'] == 2024]

        # Create correlation matrix for 2024 data
        correlation_2024 = markets_2024[market_return_columns].corr()

        # Clean up column and index names (remove '_Return' suffix)
        correlation_2024.columns = [col.split('_')[0] for col in correlation_2024.columns]
        correlation_2024.index = [idx.split('_')[0] for idx in correlation_2024.index]


        # Create subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))


        # Display correlation matrix for all 6 years of data
        sns.heatmap(correlation_6years, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, square=True, ax=ax1)
        ax1.set_title('Correlation Matrix - 6 Years Daily Returns (All Markets)')

        #  Display correlation matrix for 2024 as heatmap
        sns.heatmap(correlation_2024, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, square=True, ax=ax2, cbar_kws={'label': 'Correlation Coefficient'})
        ax2.set_title('Correlation Matrix - 2024 Daily Returns (All Markets)')

        plt.tight_layout()
        plt.show()
    """

def get_pre_covid_peaks(markets, market_price_columns, covid_crash_start):
    pre_covid_period = markets[markets['Date'] < covid_crash_start]
    pre_covid_peaks = {}
    for col in market_price_columns:
        if col in markets.columns:
            pre_covid_peaks[col] = pre_covid_period[col].max()
    return pre_covid_peaks

def calculate_recovery_analysis(markets, market_price_columns, pre_covid_peaks, covid_crash_start):
    recovery_analysis = []
    for col in market_price_columns:
        if col not in markets.columns:
            continue

        market_name = col.replace('_Close', '')
        peak_level = pre_covid_peaks[col]

        covid_period = markets[(markets['Date'] >= covid_crash_start) & (markets['Date'] <= '2020-04-30')]
        covid_low = covid_period[col].min()
        covid_low_date = covid_period[covid_period[col] == covid_low]['Date'].iloc[0]

        max_drawdown = ((covid_low - peak_level) / peak_level) * 100

        post_covid = markets[markets['Date'] > covid_low_date]
        recovery_data = post_covid[post_covid[col] >= peak_level]

        if not recovery_data.empty:
            recovery_date = recovery_data['Date'].iloc[0]
            recovery_days = (recovery_date - covid_low_date).days
            recovered = True
        else:
            recovery_date = None
            recovery_days = None
            recovered = False

        recovery_analysis.append({
            'Market': market_name,
            'Pre_COVID_Peak': peak_level,
            'COVID_Low': covid_low,
            'COVID_Low_Date': covid_low_date.strftime('%Y-%m-%d'),
            'Max_Drawdown_%': round(max_drawdown, 2),
            'Recovered': recovered,
            'Recovery_Date': recovery_date.strftime('%Y-%m-%d') if recovery_date else 'Not yet',
            'Recovery_Days': recovery_days,
            'Current_Level': markets[col].iloc[-1] if col in markets.columns else None
        })
    return recovery_analysis

def plot_covid_recovery(markets, market_price_columns, pre_covid_peaks, covid_crash_start):
    plt.figure(figsize=(10, 20))
    for i, col in enumerate(market_price_columns):
        if col not in markets.columns:
            continue

        plt.subplot(2, 3, i+1)
        market_name = col.replace('_Close', '')

        plt.plot(markets['Date'], markets[col], label=market_name, alpha=0.7)

        peak_level = pre_covid_peaks[col]
        plt.axhline(y=peak_level, color='green', linestyle='--', alpha=0.8,
                    label=f'Pre-COVID Peak: {peak_level:.0f}')

        covid_start_idx = markets[markets['Date'] >= covid_crash_start].index[0] if not markets[markets['Date'] >= covid_crash_start].empty else 0
        covid_end_idx = markets[markets['Date'] >= '2020-04-30'].index[0] if not markets[markets['Date'] >= '2020-04-30'].empty else len(markets)

        plt.axvspan(markets['Date'].iloc[covid_start_idx], markets['Date'].iloc[covid_end_idx],
                    alpha=0.2, color='red', label='COVID Crash Period')

        plt.title(f'{market_name} - COVID Recovery')
        plt.xlabel('Date')
        plt.ylabel('Price Level')
        plt.legend(fontsize=8)
        plt.xticks(rotation=45)
    plt.tight_layout()

## 3. Pre-Post Covid Performance Analytics
# Time required for a market to come back to pre-covid level
def covid_analysis():
    covid_crash_start = '2020-02-24'  # When markets started falling

    shared.markets['Date'] = pd.to_datetime(shared.markets['Date'])

    market_price_columns = ['Nifty_Close', 'DowJones_Close', 'Nasdaq_Close',
                            'HangSeng_Close', 'Nikkei_Close', 'DAX_Close']

    pre_covid_peaks = get_pre_covid_peaks(shared.markets, market_price_columns, covid_crash_start)
    plot_covid_recovery(shared.markets, market_price_columns, pre_covid_peaks, covid_crash_start)


def nifty_open_dir_year():
    # Sort dataframe by date to ensure proper order for lag calculation
    shared.markets = shared.markets.sort_values('Date').reset_index(drop=True)

    # Create Nifty_Open_Dir variable
    shared.markets['Nifty_Open_Dir'] = (shared.markets['Nifty_Open'] > shared.markets['Nifty_Close'].shift(1)).astype(int)

    # Calculate percentage of up days (Nifty_Open_Dir=1) by year
    nifty_dir_by_year = shared.markets.groupby('Year')['Nifty_Open_Dir'].agg(['count', 'sum', 'mean']).reset_index()
    # Rename columns
    nifty_dir_by_year.columns = ['Year', 'Total_Days', 'Up_Days', 'Percentage_Up']

    # Convert percentage to percentage format
    nifty_dir_by_year['Percentage_Up'] = (nifty_dir_by_year['Percentage_Up'] * 100).round(2)

    # Display the table
    plt.figure(figsize=(10, 6))
    plt.bar(nifty_dir_by_year['Year'], nifty_dir_by_year['Percentage_Up'])
    plt.title('Percentage of Up Opening Days (Nifty_Open_Dir=1) by Year')
    plt.xlabel('Year')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    for i, v in enumerate(nifty_dir_by_year['Percentage_Up']):
        plt.text(nifty_dir_by_year['Year'].iloc[i], v + 1, f'{v}%', ha='center')
    plt.tight_layout()

def nifty_open_dir_year_code():
    return """
        # Sort dataframe by date to ensure proper order for lag calculation
        shared.markets = shared.markets.sort_values('Date').reset_index(drop=True)

        # Create Nifty_Open_Dir variable
        shared.markets['Nifty_Open_Dir'] = (shared.markets['Nifty_Open'] > shared.markets['Nifty_Close'].shift(1)).astype(int)

        # Calculate percentage of up days (Nifty_Open_Dir=1) by year
        nifty_dir_by_year = shared.markets.groupby('Year')['Nifty_Open_Dir'].agg(['count', 'sum', 'mean']).reset_index()
        # Rename columns
        nifty_dir_by_year.columns = ['Year', 'Total_Days', 'Up_Days', 'Percentage_Up']

        # Convert percentage to percentage format
        nifty_dir_by_year['Percentage_Up'] = (nifty_dir_by_year['Percentage_Up'] * 100).round(2)

        # Display the table
        plt.figure(figsize=(10, 6))
        plt.bar(nifty_dir_by_year['Year'], nifty_dir_by_year['Percentage_Up'])
        plt.title('Percentage of Up Opening Days (Nifty_Open_Dir=1) by Year')
        plt.xlabel('Year')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)
        for i, v in enumerate(nifty_dir_by_year['Percentage_Up']):
            plt.text(nifty_dir_by_year['Year'].iloc[i], v + 1, f'{v}%', ha='center')
        plt.tight_layout()
    """

def global_indices_against_nifty():
    # Create box plots for each market's returns grouped by Nifty_Open_Dir
    plt.figure(figsize=(20, 12))

    # Define the markets to analyze
    markets_to_analyze = ['DowJones_Return', 'Nasdaq_Return', 'HangSeng_Return', 'Nikkei_Return', 'DAX_Return']

    for i, market in enumerate(markets_to_analyze, 1):
        plt.subplot(3, 2, i)

        # Create box plot
        sns.boxplot(x='Nifty_Open_Dir', y=market, data=shared.markets,
                    palette=["#BE6969", "#5EBCB6"])

        # Clean up market name for title
        market_name = market.replace('_Return', '')
        plt.title(f'{market_name} Returns by Nifty Opening Direction')
        plt.ylabel(f'{market_name} Daily Return')

        # Add grid for better readability
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

def global_indices_against_nifty_code():
    return """
        # Create box plots for each market's returns grouped by Nifty_Open_Dir
        plt.figure(figsize=(20, 12))

        # Define the markets to analyze
        markets_to_analyze = ['DowJones_Return', 'Nasdaq_Return', 'HangSeng_Return', 'Nikkei_Return', 'DAX_Return']

        for i, market in enumerate(markets_to_analyze, 1):
            plt.subplot(3, 2, i)

            # Create box plot
            sns.boxplot(x='Nifty_Open_Dir', y=market, data=shared.markets, hue='Nifty_Open_Dir', palette=["#BE6969", "#5EBCB6"])

            # Clean up market name for title
            market_name = market.replace('_Return', '')
            plt.title(f'{market_name} Returns by Nifty Opening Direction')
            plt.xlabel('Nifty Opening Direction (0: Down, 1: Up)')
            plt.ylabel(f'{market_name} Daily Return')

            # Add grid for better readability
            plt.grid(True, alpha=0.3)
    """