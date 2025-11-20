from shiny.express import ui, render

import phase1
import phase2
import phase3
import phase4

ui.head_content(
    ui.tags.link(rel="stylesheet", type="text/css", href="styles.css")
)

ui.page_opts(title="Market Analysis Dashboard")

SOURCE_CODE = "Source Code"

with ui.nav_panel("Phase 1"):
    with ui.navset_card_underline(title="Data Collection and Preprocessing"):
        with ui.nav_panel("Head of Master Table"):
            ui.markdown("""
            ### Master Dataset Overview

            This table displays the first few rows of our consolidated master dataset, which combines
            data from six major global market indices (Nifty 50, Dow Jones, Nasdaq, Hang Seng, Nikkei 225,
            and DAX) along with the VIX volatility index.

            The dataset includes:
            - **Date**: Trading date
            - **Open/Close prices**: Daily opening and closing values for each index
            - **Returns**: Calculated daily returns for each market
            - **Time indicators**: Year, Quarter, and Month columns for temporal analysis

            All data spans from January 2019 to April 2025, with missing values handled using
            linear interpolation to ensure continuity across different market holidays.

            Given the charactistics of the VIX index, it is included in the dataset but not used in
            return calculations for other markets.
            Cboe Volatility Index® (VIX®)
            The Cboe Voltility Index® (VIX® ) is considered by many to be the world's premier barometer
            of equity market volatility. The VIX Index is based on real-time prices of options on the
            S&P 500® Index (SPX) and is designed to reflect investors' consensus view of future
            (30-day) expected stock market volatility. The VIX Index is often referred to as the market's "fear gauge".

            """)
            ui.br()
            @render.table
            def master_table_head():
                return phase1.head_of_master_table()

        with ui.nav_panel(SOURCE_CODE):
            ui.tags.pre(
                ui.code(phase1.data_collection_and_preprocessing_code())
            )

    ui.hr()

    with ui.navset_card_underline(title="Normality Check of Daily Returns"):
        with ui.nav_panel("QQ Plots for Daily Returns"):
            ui.markdown("""
            ### QQ Plots for Daily Returns
            The QQ (Quantile-Quantile) plots below visualize the distribution of daily returns for each
            global market index, showing a pretty close alignment with the theoretical normal distribution.
            This suggests that the daily returns for these markets approximate a normal distribution, even if the
            Shapiro-Wilk test indicates otherwise.
            """)
            ui.br()
            @render.plot(height=800)
            def qq_plots():
                return phase1.qq_plots_for_returns()

        with ui.nav_panel(SOURCE_CODE):
            ui.tags.pre(
                ui.code(phase1.qq_plots_code())
            )

    ui.hr()

    with ui.navset_card_underline(title="Normality Check of Daily Returns with Shapiro-Wilk Test and Lilliefors Test"):
        with ui.nav_panel("Shapiro-Wilk Test Results"):
            ui.markdown("""
            ### Shapiro-Wilk and Lilliefors Test Results for Daily Returns
            The test results below indicate that the daily returns for all global market indices
            significantly deviate from a normal distribution (p-value < 0.05).
            This statistical test contradicts the null hypothesis of normality, suggested by the QQ plots,
            the actual return distributions have characteristics that differ from a true normal distribution.
            """)
            ui.br()
            ui.tags.pre(
                ui.code(phase1.normality_tests())
            )

        with ui.nav_panel(SOURCE_CODE):

            ui.tags.pre(
                ui.code(phase1.normality_tests_code())
            )

    with ui.navset_card_underline(title="Consolidate Columns Snippet"):
        with ui.nav_panel(SOURCE_CODE):
            ui.markdown("""
            ### Code Snippet for Consolidating Date Columns and Cleaning Data
            The following code snippet demonstrates how to extract temporal features from the 'Date' column,
            remove weekends and specific holidays, and save the cleaned dataset to a CSV file.
            """)
            ui.tags.pre(
                ui.code(phase1.consolidate_columns_code())
            )


with ui.nav_panel("Phase 2"):
    with ui.accordion(id="acc", open="Global Indices - 5 Years Performance"):
        with ui.accordion_panel("Global Indices - 5 Years Performance"):
            with ui.navset_card_underline():
                with ui.nav_panel("Box-Whisker Plot"):
                    ui.markdown("""
                    # Box-Whisker Plot of Daily Returns by YEAR
                    This section presents box-whisker plots illustrating the distribution of daily returns
                    for the global market indices over the past six years. Each plot breaks down the daily returns
                    by year, allowing for a clear comparison of market performance and volatility across different time periods.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def boxplot():
                        return phase2.boxplot_by_year()
                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase2.boxplot_code())
                    )
            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel("Table of Returns Summary"):
                    ui.markdown("""
                    # Table of Daily Returns by YEAR
                    This table shows the mean and std deviation of global market indices over the past six years.
                    one row per market, and columns for each year showing the mean and std deviation of daily returns.""")
                    ui.br()
                    @render.table
                    def returns_summary():
                        return phase2.table_of_returns_by_year()
                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase2.table_of_returns_by_year_code())
                        )

            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel('Bar Plot of median daily return by "YEAR"'):
                    ui.markdown("""
                    # Bar Plot of Median Daily Return by YEAR
                    This bar plot illustrates the median daily returns of global market indices over the past six years.
                    Each bar represents the median return for a specific year, providing insights into the central tendency
                    of market performance during that period.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def median_daily():
                        return phase2.bar_plot_median_by_year()
                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase2.bar_plot_median_by_year_code())
                    )
            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel('Heat Map by "YEAR" and "QUARTER" showing median returns'):
                    ui.markdown("""
                    # Heat Map by YEAR and QUARTER Showing Median Returns
                    This heat map visualizes the median returns of global market indices across different years and quarters.
                    The color intensity indicates the level of returns, allowing for quick identification of periods with higher or
                    lower median returns.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def heat_map():
                        return phase2.heat_map_by_year_and_quarter()
                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase2.heat_map_by_year_and_quarter_code())
                    )

        with ui.accordion_panel("Global Indices - Correlation Analysis"):
            with ui.navset_card_underline():
                with ui.nav_panel("Correlation Matrix"):
                    ui.markdown("""
                    # Correlation Matrix of Daily Returns
                    This correlation matrix displays the relationships between the daily returns of various global market indices.
                    Strong positive or negative correlations can indicate how closely the markets move in relation to each other.
                    """)
                    ui.br()
                    @render.plot(height=800)
                    def correlation_plot():
                        return phase2.correlation_matrix()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase2.correlation_matrix_code())
                    )

        with ui.accordion_panel("Global Indices - Covid Analysis"):
            with ui.navset_card_underline():
                with ui.nav_panel("Covid Analysis"):
                    ui.markdown("""
                    # Covid Analysis of Global Indices
                    This analysis examines the impact of the Covid-19 pandemic on global market indices.
                    The plots illustrate market behavior during key phases of the pandemic, highlighting significant
                    fluctuations and recovery patterns.
                    """)
                    ui.br()
                    @render.plot(height=1024)
                    def covid_analysis():
                        return phase2.plot_covid_recovery()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase2.covid_recovery_code())
                    )
            ui.br()
            ui.hr()
        with ui.accordion_panel("Global Indices - Nifty 50 Daily Movement"):
            with ui.navset_card_underline():
                with ui.nav_panel("Nifty up per year"):
                    ui.markdown("""
                    # Nifty positive movement per year
                    This plot illustrates the number of days the Nifty 50 index closed higher than its opening
                    price for each year over the past six years. It provides insights into the frequency of positive
                    trading days and overall market sentiment during this period.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def nifty_movement():
                        return phase2.nifty_open_dir_year()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase2.nifty_open_dir_year_code())
                    )
            ui.br()
            ui.hr()
            with ui.navset_card_underline():
                with ui.nav_panel("Nifty Direction Correlation with Global Indices"):
                    ui.markdown("""
                    # Nifty Direction Correlation with Global Indices
                    This plot examines the relationship between the daily opening direction of the Nifty 50 index
                    and the daily returns of major global market indices. It provides insights into how movements in
                    these international markets may be influenced or correlate with the Nifty 50's performance.
                    """)
                    ui.br()
                    @render.plot(height=1000)
                    def nifty_open_dir():
                        return phase2.global_indices_against_nifty()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase2.global_indices_against_nifty_code())
                    )

with ui.nav_panel("Phase 3"):
    with ui.accordion(id="phase3", open="Binary Logistic Regression Performance"):
        with ui.accordion_panel("Binary Logistic Regression"):
            with ui.navset_card_underline():
                with ui.nav_panel("Binary Logistic Regression Model"):
                    ui.markdown("""
                    # Binary Logistic Regression Model Summary
                    Statistical summary of the logistic regression model predicting Nifty 50 opening direction.
                    """)
                    ui.br()
                    @render.table
                    def blr_model():
                        model = phase3.fit_binary_classification_model()
                        return phase3.get_model_summary_table(model)
                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase3.data_splitting_code()),
                        ui.code(phase3.logistic_regression_code())
                    )
            ui.hr()

with ui.nav_panel("Phase 4"):

    ui.markdown("""

    1. Create data partition into train and test data sets ( 80/20) (already done for Logistic Regression)
    1. Apply Naïve Bayes Method on train data with Nifty Direction as dependent variable and previous day global market returns (and VIX) as independent variables.
    1. Obtain ROC curve and AUC for train data
    1. Obtain Confusion Matrix for train data
    1. Obtain ROC curve and AUC for test data
    1. Obtain Confusion Matrix for test data
    1. Repeat steps 2 to 6 for Decision Tree
    1. Repeat steps 2 to 6 for Random Forest Method
    1. Compare AUC for test data in case of 4/5 methods and finalize the method
    """)