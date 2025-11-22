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

    with ui.accordion(id="acc_phase3", open="Initial Model - All Variables"):
        with ui.accordion_panel("Initial Model - All Variables"):
            with ui.navset_card_underline():
                with ui.nav_panel("Model Summary"):
                    ui.markdown("""
                    # Initial Logistic Regression Model - All Variables
                    This is the initial logistic regression model using all available independent variables:
                    DowJones_Return, Nasdaq_Return, HangSeng_Return, Nikkei_Return, DAX_Return, and VIX_Return.

                    The model summary below shows the coefficients, p-values, and other statistical measures
                    for each variable. We'll use VIF analysis to check for multicollinearity before
                    refining the model.

                    **Significance codes:** *** p<0.001, ** p<0.01, * p<0.05
                    """)
                    ui.br()

                    ui.h4("Model Statistics")
                    @render.table
                    def initial_model_stats():
                        _, stats_df = phase3.logistic_regression_summary()
                        return stats_df

                    ui.br()
                    ui.h4("Coefficients")
                    @render.table
                    def initial_model_coef():
                        results_df, _ = phase3.logistic_regression_summary()
                        return results_df

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase3.logistic_regression_summary_code())
                    )

        with ui.accordion_panel("VIF Analysis"):
            with ui.navset_card_underline():
                with ui.nav_panel("VIF Table"):
                    ui.markdown("""
                    # Variance Inflation Factor (VIF) Analysis
                    This table displays the VIF scores for each independent variable in the initial logistic regression model.
                    Variables with high VIF scores indicate redundancy and should be considered for removal
                    to improve model stability and interpretability. Since all variables have VIF lower than 5, we can keep them.
                    """)
                    ui.br()
                    @render.table
                    def vif_table():
                        return phase3.display_vif_analysis()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase3.display_vif_analysis_code())
                    )

        with ui.accordion_panel("Adjusted Model - Significant Variables Only"):
            with ui.navset_card_underline():
                with ui.nav_panel("Model Summary"):
                    ui.markdown("""
                    # Adjusted Logistic Regression Model - Significant Variables
                    Based on the VIF analysis and statistical significance, we've refined the model to include
                    only the following variables: 'HangSeng_Return', 'Nikkei_Return', 'DAX_Return', 'VIX_Return'

                    This adjusted model removes variables with p-value higher than 0.05 and focuses on
                    the most predictive features.

                    **Significance codes:** *** p<0.001, ** p<0.01, * p<0.05
                    """)
                    ui.br()

                    ui.h4("Model Statistics")
                    @render.table
                    def adjusted_model_stats():
                        _, stats_df = phase3.logistic_regression_significant_vars_summary()
                        return stats_df

                    ui.br()
                    ui.h4("Coefficients")
                    @render.table
                    def adjusted_model_coef():
                        results_df, _ = phase3.logistic_regression_significant_vars_summary()
                        return results_df

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase3.logistic_regression_significant_vars_summary_code())
                    )

        with ui.accordion_panel("ROC Curve Analysis"):
            with ui.navset_card_underline():
                with ui.nav_panel("ROC Curve - Training Data"):
                    ui.markdown("""
                    # ROC Curve - Training Dataset
                    The Receiver Operating Characteristic (ROC) curve illustrates the performance of the logistic regression
                    model on the training dataset. The Area Under Curve (AUC) provides a single metric to evaluate model performance,
                    where values closer to 1.0 indicate better classification ability.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def roc_train():
                        return phase3.plot_roc_curve_train()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase3.plot_roc_curve_train_code())
                    )

            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel("ROC Curve - Test Data"):
                    ui.markdown("""
                    # ROC Curve - Test Dataset
                    This ROC curve evaluates the logistic regression model's performance on the test dataset.
                    Comparing this with the training ROC curve helps assess whether the model generalizes well
                    to unseen data or if there are signs of overfitting.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def roc_test():
                        return phase3.plot_roc_curve_test()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase3.plot_roc_curve_test_code())
                    )

        with ui.accordion_panel("Confusion Matrix Analysis"):
            with ui.navset_card_underline():
                with ui.nav_panel("Confusion Matrix - Training Data"):
                    ui.markdown(f"""
                    # Confusion Matrix - Training Dataset
                    The confusion matrix visualizes the classification performance on the training dataset using
                    an optimal threshold of {phase3.OPTIMAL_THRESHOLD}. It shows the counts of true positives, true negatives,
                    false positives, and false negatives, along with detailed precision, recall, and F1-score metrics.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def cm_train():
                        return phase3.plot_confusion_matrix_train()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase3.plot_confusion_matrix_train_code())
                    )

            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel("Confusion Matrix - Test Data"):
                    ui.markdown(f"""
                    # Confusion Matrix - Test Dataset
                    This confusion matrix evaluates the model's predictions on the test dataset using the same
                    threshold of {phase3.OPTIMAL_THRESHOLD}. Comparing these metrics with the training results helps
                    determine if the model maintains consistent performance on new, unseen data.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def cm_test():
                        return phase3.plot_confusion_matrix_test()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(
                        ui.code(phase3.plot_confusion_matrix_test_code())
                    )


with ui.nav_panel("Phase 4"):

    with ui.accordion(id="acc_phase4", open="Naive Bayes Classifier"):
        with ui.accordion_panel("Naive Bayes Classifier"):
            ui.markdown("""
            ## Gaussian Naive Bayes
            Naive Bayes is a probabilistic classifier based on Bayes' theorem with the assumption of independence
            between features. The Gaussian variant assumes that continuous features follow a normal distribution.

            This method is particularly useful for:
            - Fast training and prediction
            - Working well with small datasets
            - Handling high-dimensional data effectively
            """)

            with ui.navset_card_underline():
                with ui.nav_panel("ROC Curve - Training Data"):
                    ui.markdown("""
                    ### ROC Curve - Training Dataset
                    The ROC curve shows the trade-off between true positive rate and false positive rate
                    for the Naive Bayes model on training data.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def nb_roc_train():
                        phase4.train_naive_bayes()
                        return phase4.plot_roc_curve_nb_train()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(ui.code(phase4.train_naive_bayes_code()))

            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel("ROC Curve - Test Data"):
                    ui.markdown("""
                    ### ROC Curve - Test Dataset
                    Evaluates the Naive Bayes model's performance on unseen test data.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def nb_roc_test():
                        return phase4.plot_roc_curve_nb_test()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(ui.code(phase4.train_naive_bayes_code()))

            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel("Confusion Matrix - Training Data"):
                    ui.markdown("""
                    ### Confusion Matrix - Training Dataset
                    Shows the classification performance with actual vs predicted values for training data.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def nb_cm_train():
                        return phase4.plot_confusion_matrix_nb_train()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(ui.code(phase4.train_naive_bayes_code()))

            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel("Confusion Matrix - Test Data"):
                    ui.markdown("""
                    ### Confusion Matrix - Test Dataset
                    Shows the classification performance with actual vs predicted values for test data.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def nb_cm_test():
                        return phase4.plot_confusion_matrix_nb_test()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(ui.code(phase4.train_naive_bayes_code()))

        with ui.accordion_panel("Decision Tree Classifier"):
            ui.markdown("""
            ## Decision Tree Classifier
            Decision Trees are non-parametric models that learn decision rules from data features.
            They create a tree-like structure of decisions to classify instances.

            Key characteristics:
            - Easy to interpret and visualize
            - Can handle both numerical and categorical data
            - Prone to overfitting (controlled with max_depth parameter)
            """)

            with ui.navset_card_underline():
                with ui.nav_panel("ROC Curve - Training Data"):
                    ui.markdown("""
                    ### ROC Curve - Training Dataset
                    The ROC curve for Decision Tree model on training data.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def dt_roc_train():
                        phase4.train_decision_tree()
                        return phase4.plot_roc_curve_dt_train()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(ui.code(phase4.train_decision_tree_code()))

            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel("ROC Curve - Test Data"):
                    ui.markdown("""
                    ### ROC Curve - Test Dataset
                    Evaluates the Decision Tree model's performance on unseen test data.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def dt_roc_test():
                        return phase4.plot_roc_curve_dt_test()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(ui.code(phase4.train_decision_tree_code()))

            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel("Confusion Matrix - Training Data"):
                    ui.markdown("""
                    ### Confusion Matrix - Training Dataset
                    Shows the classification performance for training data.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def dt_cm_train():
                        return phase4.plot_confusion_matrix_dt_train()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(ui.code(phase4.train_decision_tree_code()))

            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel("Confusion Matrix - Test Data"):
                    ui.markdown("""
                    ### Confusion Matrix - Test Dataset
                    Shows the classification performance for test data.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def dt_cm_test():
                        return phase4.plot_confusion_matrix_dt_test()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(ui.code(phase4.train_decision_tree_code()))

        with ui.accordion_panel("Random Forest Classifier"):
            ui.markdown("""
            ## Random Forest Classifier
            Random Forest is an ensemble method that creates multiple decision trees and combines
            their predictions through voting. This reduces overfitting and improves generalization.

            Advantages:
            - More robust than single decision trees
            - Handles large datasets with high dimensionality well
            - Provides feature importance rankings
            - Less prone to overfitting
            """)

            with ui.navset_card_underline():
                with ui.nav_panel("ROC Curve - Training Data"):
                    ui.markdown("""
                    ### ROC Curve - Training Dataset
                    The ROC curve for Random Forest model on training data.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def rf_roc_train():
                        phase4.train_random_forest()
                        return phase4.plot_roc_curve_rf_train()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(ui.code(phase4.train_random_forest_code()))

            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel("ROC Curve - Test Data"):
                    ui.markdown("""
                    ### ROC Curve - Test Dataset
                    Evaluates the Random Forest model's performance on unseen test data.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def rf_roc_test():
                        return phase4.plot_roc_curve_rf_test()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(ui.code(phase4.train_random_forest_code()))

            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel("Confusion Matrix - Training Data"):
                    ui.markdown("""
                    ### Confusion Matrix - Training Dataset
                    Shows the classification performance for training data.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def rf_cm_train():
                        return phase4.plot_confusion_matrix_rf_train()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(ui.code(phase4.train_random_forest_code()))

            ui.hr()

            with ui.navset_card_underline():
                with ui.nav_panel("Confusion Matrix - Test Data"):
                    ui.markdown("""
                    ### Confusion Matrix - Test Dataset
                    Shows the classification performance for test data.
                    """)
                    ui.br()
                    @render.plot(height=600)
                    def rf_cm_test():
                        return phase4.plot_confusion_matrix_rf_test()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(ui.code(phase4.train_random_forest_code()))

        with ui.accordion_panel("Model Comparison"):
            with ui.navset_card_underline():
                with ui.nav_panel("AUC Comparison Table"):
                    ui.markdown("""
                    ## Model Performance Comparison

                    This table compares the Area Under the ROC Curve (AUC) for all three models
                    on both training and test datasets.

                    **Key Insights:**
                    - Higher AUC indicates better model performance
                    - Large differences between Train and Test AUC suggest overfitting
                    - The model with the best Test AUC and smallest Train-Test difference is preferred
                    """)
                    ui.br()
                    @render.table
                    def model_comparison():
                        return phase4.compare_models()

                with ui.nav_panel(SOURCE_CODE):
                    ui.tags.pre(ui.code(phase4.compare_models_code()))