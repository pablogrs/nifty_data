
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import shared

# Constants
NB_MODEL_NAME = "Gaussian Naive Bayes"
DT_MODEL_NAME = "Decision Tree Classifier"
RF_MODEL_NAME = "Random Forest Classifier"
INDEPENDENT_VARS = ['DowJones_Return', 'Nasdaq_Return', 'HangSeng_Return', 'Nikkei_Return', 'DAX_Return', 'VIX_Return']

# Global variables
train_df, test_df = None, None
X_train, X_test, y_train, y_test = None, None, None, None
nb_model, dt_model, rf_model = None, None, None
roc_auc_train_nb, roc_auc_test_nb = None, None
roc_auc_train_dt, roc_auc_test_dt = None, None
roc_auc_train_rf, roc_auc_test_rf = None, None

# Prepare data by adding Nifty_Dir_Open and creating train/test split
df_copy = shared.markets.copy()
df_copy['Nifty_Dir_Open'] = (df_copy['Nifty_Return'] > 0).astype(int)
train_df, test_df = train_test_split(df_copy, test_size=0.2, random_state=123)

# Separate dependent and independent variables
y_train = train_df['Nifty_Dir_Open']
y_test = test_df['Nifty_Dir_Open']
X_train = train_df[INDEPENDENT_VARS]
X_test = test_df[INDEPENDENT_VARS]


# ============================================================================
# NAIVE BAYES FUNCTIONS
# ============================================================================

def train_naive_bayes():
    """Train Naive Bayes model."""
    global nb_model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    return nb_model


def plot_roc_curve_nb_train():
    """Generate ROC curve for Naive Bayes training dataset."""
    global roc_auc_train_nb

    # Predict probabilities on the train set
    train_predprob_nb = nb_model.predict_proba(X_train)[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_train, train_predprob_nb)
    roc_auc_train_nb = auc(fpr, tpr)

    shared.plot_aoc_roc_curve(fpr, tpr, roc_auc_train_nb,
                              title='ROC Curve - Train Dataset',
                              model_name=NB_MODEL_NAME)


def plot_roc_curve_nb_test():
    """Generate ROC curve for Naive Bayes test dataset."""
    global roc_auc_test_nb

    # Predict probabilities on the test set
    test_predprob_nb = nb_model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, test_predprob_nb)
    roc_auc_test_nb = auc(fpr, tpr)

    shared.plot_aoc_roc_curve(fpr, tpr, roc_auc_test_nb,
                              title='ROC Curve - Test Dataset',
                              model_name=NB_MODEL_NAME)


def plot_confusion_matrix_nb_train():
    """Generate confusion matrix for Naive Bayes training dataset."""
    # Predict probabilities and classes
    train_predprob_nb = nb_model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, train_predprob_nb)

    # Use optimal threshold
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred_nb_train = (train_predprob_nb >= optimal_threshold)

    cm = confusion_matrix(y_train, y_pred_nb_train)
    cr = classification_report(y_train, y_pred_nb_train)

    shared.plot_confusion_matrix(cm, cr,
                                 title=f'Confusion Matrix - Train Dataset\n(Threshold: {optimal_threshold:.4f})',
                                 model_name=NB_MODEL_NAME)


def plot_confusion_matrix_nb_test():
    """Generate confusion matrix for Naive Bayes test dataset."""
    # Predict probabilities and classes
    test_predprob_nb = nb_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, test_predprob_nb)

    # Use optimal threshold
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred_nb_test = (test_predprob_nb >= optimal_threshold)

    cm = confusion_matrix(y_test, y_pred_nb_test)
    cr = classification_report(y_test, y_pred_nb_test)

    shared.plot_confusion_matrix(cm, cr,
                                 title=f'Confusion Matrix - Test Dataset\n(Threshold: {optimal_threshold:.4f})',
                                 model_name=NB_MODEL_NAME)


# ============================================================================
# DECISION TREE FUNCTIONS
# ============================================================================

def train_decision_tree():
    """Train Decision Tree model."""
    global dt_model
    dt_model = DecisionTreeClassifier(
        random_state=123,
        max_depth=5,
        ccp_alpha=0
    )
    dt_model.fit(X_train, y_train)
    return dt_model


def plot_roc_curve_dt_train():
    """Generate ROC curve for Decision Tree training dataset."""
    global roc_auc_train_dt

    # Predict probabilities on the train set
    train_predprob_dt = dt_model.predict_proba(X_train)[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_train, train_predprob_dt)
    roc_auc_train_dt = auc(fpr, tpr)

    shared.plot_aoc_roc_curve(fpr, tpr, roc_auc_train_dt,
                              title='ROC Curve - Train Dataset',
                              model_name=DT_MODEL_NAME)


def plot_roc_curve_dt_test():
    """Generate ROC curve for Decision Tree test dataset."""
    global roc_auc_test_dt

    # Predict probabilities on the test set
    test_predprob_dt = dt_model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, test_predprob_dt)
    roc_auc_test_dt = auc(fpr, tpr)

    shared.plot_aoc_roc_curve(fpr, tpr, roc_auc_test_dt,
                              title='ROC Curve - Test Dataset',
                              model_name=DT_MODEL_NAME)


def plot_confusion_matrix_dt_train():
    """Generate confusion matrix for Decision Tree training dataset."""
    # Predict probabilities and classes
    train_predprob_dt = dt_model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, train_predprob_dt)

    # Use optimal threshold
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred_dt_train = (train_predprob_dt >= optimal_threshold)

    cm = confusion_matrix(y_train, y_pred_dt_train)
    cr = classification_report(y_train, y_pred_dt_train)

    shared.plot_confusion_matrix(cm, cr,
                                 title=f'Confusion Matrix - Train Dataset\n(Threshold: {optimal_threshold:.4f})',
                                 model_name=DT_MODEL_NAME)


def plot_confusion_matrix_dt_test():
    """Generate confusion matrix for Decision Tree test dataset."""
    # Predict probabilities and classes
    test_predprob_dt = dt_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, test_predprob_dt)

    # Use optimal threshold
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred_dt_test = (test_predprob_dt >= optimal_threshold)

    cm = confusion_matrix(y_test, y_pred_dt_test)
    cr = classification_report(y_test, y_pred_dt_test)

    shared.plot_confusion_matrix(cm, cr,
                                 title=f'Confusion Matrix - Test Dataset\n(Threshold: {optimal_threshold:.4f})',
                                 model_name=DT_MODEL_NAME)


# ============================================================================
# RANDOM FOREST FUNCTIONS
# ============================================================================

def train_random_forest():
    """Train Random Forest model."""
    global rf_model
    rf_model = RandomForestClassifier(
        random_state=999,
        oob_score=True,
        max_features='sqrt',
        max_depth=2,
        n_estimators=50
    )
    rf_model.fit(X_train, y_train)
    return rf_model


def plot_roc_curve_rf_train():
    """Generate ROC curve for Random Forest training dataset."""
    global roc_auc_train_rf

    # Predict probabilities on the train set
    train_predprob_rf = rf_model.predict_proba(X_train)[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_train, train_predprob_rf)
    roc_auc_train_rf = auc(fpr, tpr)

    shared.plot_aoc_roc_curve(fpr, tpr, roc_auc_train_rf,
                              title='ROC Curve - Train Dataset',
                              model_name=RF_MODEL_NAME)


def plot_roc_curve_rf_test():
    """Generate ROC curve for Random Forest test dataset."""
    global roc_auc_test_rf

    # Predict probabilities on the test set
    test_predprob_rf = rf_model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, test_predprob_rf)
    roc_auc_test_rf = auc(fpr, tpr)

    shared.plot_aoc_roc_curve(fpr, tpr, roc_auc_test_rf,
                              title='ROC Curve - Test Dataset',
                              model_name=RF_MODEL_NAME)


def plot_confusion_matrix_rf_train():
    """Generate confusion matrix for Random Forest training dataset."""
    # Predict probabilities and classes
    train_predprob_rf = rf_model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, train_predprob_rf)

    # Use optimal threshold
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred_rf_train = (train_predprob_rf >= optimal_threshold)

    cm = confusion_matrix(y_train, y_pred_rf_train)
    cr = classification_report(y_train, y_pred_rf_train)

    shared.plot_confusion_matrix(cm, cr,
                                 title=f'Confusion Matrix - Train Dataset\n(Threshold: {optimal_threshold:.4f})',
                                 model_name=RF_MODEL_NAME)


def plot_confusion_matrix_rf_test():
    """Generate confusion matrix for Random Forest test dataset."""
    # Predict probabilities and classes
    test_predprob_rf = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, test_predprob_rf)

    # Use optimal threshold
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred_rf_test = (test_predprob_rf >= optimal_threshold)

    cm = confusion_matrix(y_test, y_pred_rf_test)
    cr = classification_report(y_test, y_pred_rf_test)

    shared.plot_confusion_matrix(cm, cr,
                                 title=f'Confusion Matrix - Test Dataset\n(Threshold: {optimal_threshold:.4f})',
                                 model_name=RF_MODEL_NAME)


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare_models():
    """Compare AUC scores across all models."""
    # Ensure all models are trained and evaluated
    if None in [roc_auc_train_nb, roc_auc_test_nb, roc_auc_train_dt,
                roc_auc_test_dt, roc_auc_train_rf, roc_auc_test_rf]:
        # Train models if not already trained
        if nb_model is None:
            train_naive_bayes()
            plot_roc_curve_nb_train()
            plot_roc_curve_nb_test()

        if dt_model is None:
            train_decision_tree()
            plot_roc_curve_dt_train()
            plot_roc_curve_dt_test()

        if rf_model is None:
            train_random_forest()
            plot_roc_curve_rf_train()
            plot_roc_curve_rf_test()

    models = [NB_MODEL_NAME, DT_MODEL_NAME, RF_MODEL_NAME]
    train_aucs = [roc_auc_train_nb, roc_auc_train_dt, roc_auc_train_rf]
    test_aucs = [roc_auc_test_nb, roc_auc_test_dt, roc_auc_test_rf]

    comparison_df = pd.DataFrame({
        'Model': models,
        'Train AUC': train_aucs,
        'Test AUC': test_aucs,
        'Difference': [abs(train - test) for train, test in zip(train_aucs, test_aucs)]
    })

    return comparison_df


# ============================================================================
# CODE SNIPPET FUNCTIONS (for Source Code panels)
# ============================================================================

def train_naive_bayes_code():
    return """
# Train Naive Bayes model
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
"""


def train_decision_tree_code():
    return """
# Train Decision Tree model
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    random_state=123,
    max_depth=5,
    ccp_alpha=0
)
dt_model.fit(X_train, y_train)
"""


def train_random_forest_code():
    return """
# Train Random Forest model
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    random_state=999,
    oob_score=True,
    max_features='sqrt',
    max_depth=2,
    n_estimators=50
)
rf_model.fit(X_train, y_train)
"""


def compare_models_code():
    return """
# Compare AUC scores across all models
import pandas as pd

models = [NB_MODEL_NAME, DT_MODEL_NAME, RF_MODEL_NAME]
train_aucs = [roc_auc_train_nb, roc_auc_train_dt, roc_auc_train_rf]
test_aucs = [roc_auc_test_nb, roc_auc_test_dt, roc_auc_test_rf]

comparison_df = pd.DataFrame({
    'Model': models,
    'Train AUC': train_aucs,
    'Test AUC': test_aucs,
    'Difference': [abs(train - test) for train, test in zip(train_aucs, test_aucs)]
})

display(comparison_df)
"""
