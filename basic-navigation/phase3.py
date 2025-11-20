
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import shared

# Constants
LR_MODEL_NAME = 'Logistic Regression Model'
INDEPENDENT_VARS = ['DowJones_Return', 'Nasdaq_Return', 'HangSeng_Return', 'Nikkei_Return', 'DAX_Return', 'VIX_Return']
SIGNIFICANT_VARS = ['HangSeng_Return', 'Nikkei_Return', 'DAX_Return', 'VIX_Return']
OPTIMAL_THRESHOLD = 0.4753

def prepare_predictive_data():
    """Prepare data by adding Nifty_Dir_Open and creating train/test split."""
    df_copy = shared.markets.copy()
    df_copy['Nifty_Dir_Open'] = (df_copy['Nifty_Return'] > 0).astype(int)
    training_set, testing_set = train_test_split(df_copy, test_size=0.2, random_state=123)
    return df_copy, training_set, testing_set

def plot_roc_curve_train():
    """Generate ROC curve for training dataset."""
    _, training_set, _ = prepare_predictive_data()

    X_train = sm.add_constant(training_set[SIGNIFICANT_VARS])
    y_train = training_set['Nifty_Dir_Open']

    fitted_model = sm.Logit(y_train.astype(int), X_train.astype(int)).fit(disp=0)
    predicted_probs = fitted_model.predict(X_train.astype(float))

    false_pos_rate, true_pos_rate, _ = roc_curve(y_train, predicted_probs)
    area_under_curve = auc(false_pos_rate, true_pos_rate)

    plt.figure(figsize=(8, 6))
    plt.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {area_under_curve:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Training Dataset')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

def plot_roc_curve_train_code():
    return """
# Generate ROC curve for training dataset
_, training_set, _ = prepare_predictive_data()

X_train = sm.add_constant(training_set[SIGNIFICANT_VARS])
y_train = training_set['Nifty_Dir_Open']

fitted_model = sm.Logit(y_train.astype(int), X_train.astype(int)).fit(disp=0)
predicted_probs = fitted_model.predict(X_train.astype(float))

false_pos_rate, true_pos_rate, _ = roc_curve(y_train, predicted_probs)
area_under_curve = auc(false_pos_rate, true_pos_rate)

plt.figure(figsize=(8, 6))
plt.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {area_under_curve:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Training Dataset')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
    """

def plot_roc_curve_test():
    """Generate ROC curve for test dataset."""
    _, training_set, testing_set = prepare_predictive_data()

    X_train = sm.add_constant(training_set[SIGNIFICANT_VARS])
    X_test = sm.add_constant(testing_set[SIGNIFICANT_VARS])
    y_train = training_set['Nifty_Dir_Open']
    y_test = testing_set['Nifty_Dir_Open']

    fitted_model = sm.Logit(y_train.astype(int), X_train.astype(int)).fit(disp=0)
    predicted_probs_test = fitted_model.predict(X_test.astype(float))

    false_pos_rate, true_pos_rate, _ = roc_curve(y_test, predicted_probs_test)
    area_under_curve = auc(false_pos_rate, true_pos_rate)

    plt.figure(figsize=(8, 6))
    plt.plot(false_pos_rate, true_pos_rate, color='darkgreen', lw=2,
             label=f'ROC curve (AUC = {area_under_curve:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test Dataset')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

def plot_roc_curve_test_code():
    return """
# Generate ROC curve for test dataset
_, training_set, testing_set = prepare_predictive_data()

X_train = sm.add_constant(training_set[SIGNIFICANT_VARS])
X_test = sm.add_constant(testing_set[SIGNIFICANT_VARS])
y_train = training_set['Nifty_Dir_Open']
y_test = testing_set['Nifty_Dir_Open']

fitted_model = sm.Logit(y_train.astype(int), X_train.astype(int)).fit(disp=0)
predicted_probs_test = fitted_model.predict(X_test.astype(float))

false_pos_rate, true_pos_rate, _ = roc_curve(y_test, predicted_probs_test)
area_under_curve = auc(false_pos_rate, true_pos_rate)

plt.figure(figsize=(8, 6))
plt.plot(false_pos_rate, true_pos_rate, color='darkgreen', lw=2,
         label=f'ROC curve (AUC = {area_under_curve:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test Dataset')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
    """

def plot_confusion_matrix_train():
    """Generate confusion matrix for training dataset."""
    _, training_set, _ = prepare_predictive_data()

    X_train = sm.add_constant(training_set[SIGNIFICANT_VARS])
    y_train = training_set['Nifty_Dir_Open']

    fitted_model = sm.Logit(y_train.astype(int), X_train.astype(int)).fit(disp=0)
    predictions = fitted_model.predict(X_train.astype(float)) >= OPTIMAL_THRESHOLD

    conf_matrix = confusion_matrix(y_train, predictions)
    class_report = classification_report(y_train, predictions)

    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Down', 'Predicted Up'],
                yticklabels=['Actual Down', 'Actual Up'])
    plt.title(f'Confusion Matrix - Training Dataset\n(Threshold: {OPTIMAL_THRESHOLD})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    print(f"\nClassification Report - {LR_MODEL_NAME} - Training Dataset:\n{class_report}")

def plot_confusion_matrix_train_code():
    return """
# Generate confusion matrix for training dataset
_, training_set, _ = prepare_predictive_data()

X_train = sm.add_constant(training_set[SIGNIFICANT_VARS])
y_train = training_set['Nifty_Dir_Open']

fitted_model = sm.Logit(y_train.astype(int), X_train.astype(int)).fit(disp=0)
predictions = fitted_model.predict(X_train.astype(float)) >= OPTIMAL_THRESHOLD

conf_matrix = confusion_matrix(y_train, predictions)
class_report = classification_report(y_train, predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Down', 'Predicted Up'],
            yticklabels=['Actual Down', 'Actual Up'])
plt.title(f'Confusion Matrix - Training Dataset\\n(Threshold: {OPTIMAL_THRESHOLD})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

print(f"\\nClassification Report - {LR_MODEL_NAME} - Training Dataset:\\n{class_report}")
    """

def plot_confusion_matrix_test():
    """Generate confusion matrix for test dataset."""
    _, training_set, testing_set = prepare_predictive_data()

    X_train = sm.add_constant(training_set[SIGNIFICANT_VARS])
    X_test = sm.add_constant(testing_set[SIGNIFICANT_VARS])
    y_train = training_set['Nifty_Dir_Open']
    y_test = testing_set['Nifty_Dir_Open']

    fitted_model = sm.Logit(y_train.astype(int), X_train.astype(int)).fit(disp=0)
    predictions_test = fitted_model.predict(X_test.astype(float)) >= OPTIMAL_THRESHOLD

    conf_matrix = confusion_matrix(y_test, predictions_test)
    class_report = classification_report(y_test, predictions_test)

    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Predicted Down', 'Predicted Up'],
                yticklabels=['Actual Down', 'Actual Up'])
    plt.title(f'Confusion Matrix - Test Dataset\n(Threshold: {OPTIMAL_THRESHOLD})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    print(f"\nClassification Report - {LR_MODEL_NAME} - Test Dataset:\n{class_report}")

def plot_confusion_matrix_test_code():
    return """
# Generate confusion matrix for test dataset
_, training_set, testing_set = prepare_predictive_data()

X_train = sm.add_constant(training_set[SIGNIFICANT_VARS])
X_test = sm.add_constant(testing_set[SIGNIFICANT_VARS])
y_train = training_set['Nifty_Dir_Open']
y_test = testing_set['Nifty_Dir_Open']

fitted_model = sm.Logit(y_train.astype(int), X_train.astype(int)).fit(disp=0)
predictions_test = fitted_model.predict(X_test.astype(float)) >= OPTIMAL_THRESHOLD

conf_matrix = confusion_matrix(y_test, predictions_test)
class_report = classification_report(y_test, predictions_test)

plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Predicted Down', 'Predicted Up'],
            yticklabels=['Actual Down', 'Actual Up'])
plt.title(f'Confusion Matrix - Test Dataset\\n(Threshold: {OPTIMAL_THRESHOLD})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

print(f"\\nClassification Report - {LR_MODEL_NAME} - Test Dataset:\\n{class_report}")
    """

def display_vif_analysis():
    """Display VIF analysis table."""
    _, training_set, _ = prepare_predictive_data()
    X_train = sm.add_constant(training_set[INDEPENDENT_VARS])

    vif_analysis = pd.DataFrame()
    vif_analysis["Feature"] = X_train.columns
    vif_analysis["VIF Score"] = [variance_inflation_factor(X_train.values, idx)
                                   for idx in range(X_train.shape[1])]

    return vif_analysis

def display_vif_analysis_code():
    return """
# Display VIF analysis table
_, training_set, _ = prepare_predictive_data()
X_train = sm.add_constant(training_set[INDEPENDENT_VARS])

vif_analysis = pd.DataFrame()
vif_analysis["Feature"] = X_train.columns
vif_analysis["VIF Score"] = [variance_inflation_factor(X_train.values, idx)
                               for idx in range(X_train.shape[1])]

display(vif_analysis)
    """