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

        melted_returns = df.melt(id_vars=['Year'], value_vars=market_return_columns,
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

    pivot_median.plot(kind='bar', figsize=(14, 8))
    plt.title('Median Daily Return by Market and Year')
    plt.xlabel('Year')
    plt.ylabel('Median Daily Return')
    plt.legend(title='Market')
    plt.tight_layout()





def correlation_matrix():
    # Create correlation matrix for all 6 years of data
    correlation_6years = shared.markets[shared.market_return_columns].corr()

    # Clean up column and index names (remove '_Return' suffix)
    correlation_6years.columns = [col.split('_')[0] for col in correlation_6years.columns]
    correlation_6years.index = [idx.split('_')[0] for idx in correlation_6years.index]


    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_6years, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Matrix - 6 years (All Markets)')


    plt.tight_layout()

def correlation_matrix_1_year():
    # Filter data for 2024 only
    df_2024 = shared.markets[shared.markets['Year'] == 2024]

    # Create correlation matrix for 2024 data
    correlation_2024 = df_2024[shared.market_return_columns].corr()

    # Clean up column and index names (remove '_Return' suffix)
    correlation_2024.columns = [col.split('_')[0] for col in correlation_2024.columns]
    correlation_2024.index = [idx.split('_')[0] for idx in correlation_2024.index]

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_2024, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Matrix - 2024 Daily Returns (All Markets)')
    plt.tight_layout()
