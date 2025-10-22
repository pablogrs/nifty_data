# filepath: /Users/pablosan/Documents/Ireland/study/DSI/PROJECT/basic-navigation/phase3.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shared

# Market return columns for modeling
independent_vars = [
    'DowJones_Return', 'Nasdaq_Return',
    'HangSeng_Return', 'Nikkei_Return',
    'DAX_Return', 'VIX_Return'
]

def define_target_variable():
    """Create binary target variable for Nifty direction prediction"""
    shared.markets['Nifty_Dir_Open'] = (shared.markets['Nifty_Return'] > 0).astype(int)
    return shared.markets

df = define_target_variable()

# Create data partition into train and test data sets (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)

def fit_binary_classification_model():
    # Add a constant to the independent variables (for the intercept term)
    X_train = sm.add_constant(train_df[independent_vars])
    X_test = sm.add_constant(test_df[independent_vars])
    # Define the dependent variable
    y_train = train_df['Nifty_Dir_Open']
    y_test = test_df['Nifty_Dir_Open']

    # Build the logistic regression model
    model = sm.Logit(y_train.astype(int), X_train.astype(int)).fit()

    return model

def get_model_summary_table(model):
    # Extract key statistics from model summary
    summary_data = {
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'Std Error': model.bse.values,
        'z-score': model.tvalues.values,
        'P-value': model.pvalues.values,
        'CI Lower': model.conf_int()[0].values,
        'CI Upper': model.conf_int()[1].values
    }

    return pd.DataFrame(summary_data).round(4)

def calculate_variance_inflation_factors(feature_matrix):
    """Assess multicollinearity using VIF scores"""
    vif_scores = pd.DataFrame()
    vif_scores["predictor"] = feature_matrix.columns
    vif_scores["VIF_score"] = [
        variance_inflation_factor(feature_matrix.values, idx)
        for idx in range(feature_matrix.shape[1])
    ]

    # Filter predictors with acceptable VIF
    acceptable_predictors = []
    for predictor in feature_columns:
        vif_value = vif_scores.loc[vif_scores['predictor'] == predictor, 'VIF_score'].values[0]
        if vif_value <= 5:
            acceptable_predictors.append(predictor)

    return vif_scores, acceptable_predictors

def generate_training_roc_analysis(model, feature_matrix, target_vector):
    """Create ROC curve and calculate AUC for training data"""
    predicted_probabilities = model.predict(feature_matrix.astype(int))
    false_pos_rate, true_pos_rate, decision_thresholds = roc_curve(target_vector, predicted_probabilities)
    area_under_curve = auc(false_pos_rate, true_pos_rate)

    plt.figure(figsize=(8, 6))
    plt.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {area_under_curve:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Training Data - ROC Analysis')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return false_pos_rate, true_pos_rate, decision_thresholds, area_under_curve

def create_confusion_matrix_train(model, feature_matrix, target_vector):
    """Generate confusion matrix and classification metrics for training set"""
    predictions = model.predict(feature_matrix.astype(int)) >= 0.5
    conf_matrix = confusion_matrix(target_vector, predictions)
    class_report = classification_report(target_vector, predictions)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Down', 'Predicted Up'],
                yticklabels=['Actual Down', 'Actual Up'])
    plt.title('Training Set - Confusion Matrix')
    plt.xlabel('Predicted Direction')
    plt.ylabel('Actual Direction')
    plt.tight_layout()

    return conf_matrix, class_report

def find_optimal_decision_threshold(false_pos_rate, true_pos_rate, decision_thresholds):
    """Determine optimal threshold using Youden's Index"""
    youden_scores = true_pos_rate - false_pos_rate
    optimal_index = np.argmax(youden_scores)
    optimal_threshold = decision_thresholds[optimal_index]

    return optimal_threshold, optimal_index

def evaluate_test_set_performance(model, test_data):
    """Assess model performance on unseen test data"""
    test_features = sm.add_constant(test_data[feature_columns])
    test_targets = test_data['Nifty_Dir_Open']

    test_probabilities = model.predict(test_features.astype(int))
    fpr_test, tpr_test, thresh_test = roc_curve(test_targets, test_probabilities)
    auc_test = auc(fpr_test, tpr_test)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2,
             label=f'Test ROC (AUC = {auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Data - ROC Analysis')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fpr_test, tpr_test, thresh_test, auc_test

def create_test_confusion_matrix(model, test_data, threshold=0.5):
    """Generate confusion matrix for test set predictions"""
    test_features = sm.add_constant(test_data[feature_columns])
    test_targets = test_data['Nifty_Dir_Open']

    test_predictions = model.predict(test_features.astype(int)) >= threshold
    conf_matrix_test = confusion_matrix(test_targets, test_predictions)
    class_report_test = classification_report(test_targets, test_predictions)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Down', 'Predicted Up'],
                yticklabels=['Actual Down', 'Actual Up'])
    plt.title('Test Set - Confusion Matrix')
    plt.xlabel('Predicted Direction')
    plt.ylabel('Actual Direction')
    plt.tight_layout()

    return conf_matrix_test, class_report_test

# Code display functions for UI

def data_splitting_code():
    return """
        from sklearn.model_selection import train_test_split

        # Create chronological split (80/20) without shuffling
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=123, shuffle=False
        )
    """

def logistic_regression_code():
    return """
        import statsmodels.api as sm

        # Prepare features and target
        X_train = sm.add_constant(train_df[feature_columns])
        y_train = train_df['Nifty_Dir_Open']

        # Fit logistic regression model
        model = sm.Logit(y_train.astype(int), X_train.astype(int)).fit()
        print(model.summary())
    """

def vif_analysis_code():
    return """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # Calculate VIF for each predictor
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_train.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X_train.values, i)
            for i in range(X_train.shape[1])
        ]

        # Filter features with VIF <= 5
        acceptable_features = []
        for feature in feature_columns:
            vif_score = vif_data.loc[vif_data['feature'] == feature, 'VIF'].values[0]
            if vif_score <= 5:
                acceptable_features.append(feature)
    """

def roc_curve_code():
    return """
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt

        # Generate predictions and ROC curve
        probabilities = model.predict(X_train.astype(int))
        fpr, tpr, thresholds = roc_curve(y_train, probabilities)
        auc_score = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
    """