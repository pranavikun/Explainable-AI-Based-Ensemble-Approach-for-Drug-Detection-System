#3
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import matthews_corrcoef, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import json

data = pd.read_csv('classification_model_data.csv')
print("Data loaded successfully:")
print(data.head(5))

features = data.drop(columns=['activity_class'])
targets = data.activity_class
print("\nFeatures and targets separated.")
print("Feature data preview:")
print(features.head(5))
print("Target data preview:")
print(targets.head(5))

label_encoder = LabelEncoder()
targets_encoded = label_encoder.fit_transform(targets)

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

features = remove_low_variance(features, threshold=0.1)
print("\nFeatures after removing low variance:")
print(features.head(5))

def get_metrics(predicted, true):
    metrics = dict()
    metrics['accuracy'] = round(accuracy_score(predicted, true), 5)
    metrics['precision'] = round(precision_score(predicted, true, average='weighted', zero_division=0), 5)
    metrics['recall'] = round(recall_score(predicted, true, average='weighted', zero_division=0), 5)
    metrics['f1'] = round(f1_score(predicted, true, average='weighted', zero_division=0), 5)
    return metrics

X_training_set, X_validation_set, y_training_set, y_validation_set = train_test_split(
    features, targets_encoded, test_size=0.2, random_state=42)
print("\nData split into training and validation sets.")
print(f"Training set size: {X_training_set.shape}, Validation set size: {X_validation_set.shape}")

RF_model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
RF_model.fit(X_training_set, y_training_set)
print("\nRandom Forest model trained.")
joblib.dump(RF_model, 'trained_model.joblib')

y_RF_pred = RF_model.predict(X_validation_set)
print("\nRandom Forest Classification Report:")
print(classification_report(y_validation_set, y_RF_pred))

SVM_classifier = SVC(kernel='linear', probability=True, max_iter=10000)
SVM_classifier.fit(X_training_set, y_training_set)
print("\nSVM model trained.")

y_SVM_pred = SVM_classifier.predict(X_validation_set)
print("\nSVM Classification Report:")
print(classification_report(y_validation_set, y_SVM_pred))

XGB_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
XGB_model.fit(X_training_set, y_training_set)
print("\nXGBoost model trained.")

y_XGB_pred = XGB_model.predict(X_validation_set)
print("\nXGBoost Classification Report:")
print(classification_report(y_validation_set, y_XGB_pred))

ensemble_model = VotingClassifier(
    estimators=[('rf', RF_model), ('svm', SVM_classifier), ('xgb', XGB_model)],
    voting='soft'
)
ensemble_model.fit(X_training_set, y_training_set)
print("\nEnsemble model trained (RF + SVM + XGBoost).")

y_ensemble_pred = ensemble_model.predict(X_validation_set)
print("\nEnsemble validation metrics:")
print(pd.DataFrame([get_metrics(y_ensemble_pred, y_validation_set)]))
print("\nEnsemble Classification Report:")
print(classification_report(y_validation_set, y_ensemble_pred))

RF_metrics = pd.DataFrame([get_metrics(y_RF_pred, y_validation_set)])
SVM_metrics = pd.DataFrame([get_metrics(y_SVM_pred, y_validation_set)])
XGB_metrics = pd.DataFrame([get_metrics(y_XGB_pred, y_validation_set)])
Ensemble_metrics = pd.DataFrame([get_metrics(y_ensemble_pred, y_validation_set)])

RF_metrics.to_csv("RF_metrics.csv", index=False)
SVM_metrics.to_csv("SVM_metrics.csv", index=False)
XGB_metrics.to_csv("XGB_metrics.csv", index=False)
Ensemble_metrics.to_csv("Ensemble_metrics.csv", index=False)
print("\nAll metrics saved to CSV files.")


'''
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import json

# Load data
data = pd.read_csv('classification_model_data.csv')
print("Data loaded successfully:")
print(data.head(5))

# Separate features and target
features = data.drop(columns=['activity_class'])
targets = data.activity_class
print("\nFeatures and targets separated.")
print("Feature data preview:")
print(features.head(5))
print("Target data preview:")
print(targets.head(5))

# Remove low-variance features
def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

features = remove_low_variance(features, threshold=0.1)
#features = remove_low_variance(features, threshold=10.0) #changing threshold for testcase

print("\nFeatures after removing low variance:")
print(features.head(5))

# Metrics function
def get_metrics(predicted, true):
    metrics = dict()
    metrics['accuracy'] = round(accuracy_score(predicted, true), 5)
    metrics['precision'] = round(precision_score(predicted, true, average='weighted'), 5)
    metrics['recall'] = round(recall_score(predicted, true, average='weighted'), 5)
    metrics['f1'] = round(f1_score(predicted, true, average='weighted'), 5)

    return metrics

# Split data
X_training_set, X_validation_set, y_training_set, y_validation_set = train_test_split(features, targets, test_size=0.2, random_state=42)
print("\nData split into training and validation sets.")
print(f"Training set size: {X_training_set.shape}, Validation set size: {X_validation_set.shape}")

# Train Random Forest Classifier
RF_model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
RF_model.fit(X_training_set, y_training_set)
print("\nRandom Forest model trained.")

# Save the trained Random Forest model for later use
joblib.dump(RF_model, 'trained_model.joblib')  # Saves model to 'trained_model.joblib'
print("Random Forest model saved as 'trained_model.joblib'.")

# Predictions and Metrics for Random Forest
y_training_pred = RF_model.predict(X_training_set)
y_validation_pred = RF_model.predict(X_validation_set)
RF_mcc_test = matthews_corrcoef(y_validation_set, y_validation_pred)
print("\nRandom Forest MCC (Validation):", RF_mcc_test)

RF_metrics = pd.DataFrame([get_metrics(y_validation_pred, y_validation_set)])
print("\nRandom Forest validation metrics:")
print(RF_metrics)

# Train SVM Classifier
SVM_classifier = LinearSVC(max_iter=10000)
SVM_classifier.fit(X_training_set, y_training_set)
print("\nSVM model trained.")

# Predictions and Metrics for SVM
y_SVM_pred = SVM_classifier.predict(X_validation_set)
SVM_metrics = pd.DataFrame([get_metrics(y_SVM_pred, y_validation_set)])
print("\nSVM validation metrics:")
print(SVM_metrics)

# Save metrics
RF_metrics.to_csv("RF_metrics.csv", index=False)
SVM_metrics.to_csv("SVM_metrics.csv", index=False)
print("\nMetrics saved to RF_metrics.csv and SVM_metrics.csv")


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import matthews_corrcoef, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import json

# Load data
data = pd.read_csv('classification_model_data.csv')
print("Data loaded successfully:")
print(data.head(5))

# Separate features and target
features = data.drop(columns=['activity_class'])
targets = data.activity_class
print("\nFeatures and targets separated.")
print("Feature data preview:")
print(features.head(5))
print("Target data preview:")
print(targets.head(5))

# Encode target labels for XGBoost
label_encoder = LabelEncoder()
targets_encoded = label_encoder.fit_transform(targets)

# Remove low-variance features
def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

features = remove_low_variance(features, threshold=0.1)
print("\nFeatures after removing low variance:")
print(features.head(5))

# Metrics function
def get_metrics(predicted, true):
    metrics = dict()
    metrics['accuracy'] = round(accuracy_score(predicted, true), 5)
    metrics['precision'] = round(precision_score(predicted, true, average='weighted', zero_division=0), 5)
    metrics['recall'] = round(recall_score(predicted, true, average='weighted', zero_division=0), 5)
    metrics['f1'] = round(f1_score(predicted, true, average='weighted', zero_division=0), 5)
    return metrics

# Split data
X_training_set, X_validation_set, y_training_set, y_validation_set = train_test_split(
    features, targets_encoded, test_size=0.2, random_state=42)
print("\nData split into training and validation sets.")
print(f"Training set size: {X_training_set.shape}, Validation set size: {X_validation_set.shape}")

# ----------------------------
# Random Forest
# ----------------------------
RF_model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
RF_model.fit(X_training_set, y_training_set)
print("\nRandom Forest model trained.")
joblib.dump(RF_model, 'trained_model.joblib')

y_RF_pred = RF_model.predict(X_validation_set)
print("\nRandom Forest Classification Report:")
print(classification_report(y_validation_set, y_RF_pred))

# ----------------------------
# SVM
# ----------------------------
SVM_classifier = SVC(kernel='linear', probability=True, max_iter=10000)
SVM_classifier.fit(X_training_set, y_training_set)
print("\nSVM model trained.")

y_SVM_pred = SVM_classifier.predict(X_validation_set)
print("\nSVM Classification Report:")
print(classification_report(y_validation_set, y_SVM_pred))

# ----------------------------
# XGBoost
# ----------------------------
XGB_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
XGB_model.fit(X_training_set, y_training_set)
print("\nXGBoost model trained.")

y_XGB_pred = XGB_model.predict(X_validation_set)

print("\nXGBoost Classification Report:")
print(classification_report(y_validation_set, y_XGB_pred))

# ----------------------------
# Ensemble (RF + SVM + XGBoost)
# ----------------------------
ensemble_model = VotingClassifier(
    estimators=[('rf', RF_model), ('svm', SVM_classifier), ('xgb', XGB_model)],
    voting='soft'
)
ensemble_model.fit(X_training_set, y_training_set)
print("\nEnsemble model trained (RF + SVM + XGBoost).")

y_ensemble_pred = ensemble_model.predict(X_validation_set)
print("\nEnsemble validation metrics:")
print(pd.DataFrame([get_metrics(y_ensemble_pred, y_validation_set)]))
print("\nEnsemble Classification Report:")
print(classification_report(y_validation_set, y_ensemble_pred))

# ----------------------------
# Save metrics
# ----------------------------
RF_metrics = pd.DataFrame([get_metrics(y_RF_pred, y_validation_set)])
SVM_metrics = pd.DataFrame([get_metrics(y_SVM_pred, y_validation_set)])
XGB_metrics = pd.DataFrame([get_metrics(y_XGB_pred, y_validation_set)])
Ensemble_metrics = pd.DataFrame([get_metrics(y_ensemble_pred, y_validation_set)])

RF_metrics.to_csv("RF_metrics.csv", index=False)
SVM_metrics.to_csv("SVM_metrics.csv", index=False)
XGB_metrics.to_csv("XGB_metrics.csv", index=False)
Ensemble_metrics.to_csv("Ensemble_metrics.csv", index=False)
print("\nAll metrics saved to CSV files.")
'''