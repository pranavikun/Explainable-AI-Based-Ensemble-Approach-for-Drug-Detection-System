#5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
import joblib

data = pd.read_csv('classification_model_data.csv')

features = data.drop(columns=['activity_class'])
targets = data['activity_class']

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

features = remove_low_variance(features, threshold=0.1)

np.save('selected_features.npy', features.columns)

model = RandomForestClassifier(random_state=42)  
model.fit(features, targets)

joblib.dump(model, 'trained_model.joblib')

predictions = model.predict(features)

accuracy = accuracy_score(targets, predictions)
precision = precision_score(targets, predictions, average='weighted')

print("Training Accuracy:", accuracy)
print("Training Precision:", precision)
print("\nDetailed Classification Report:\n", classification_report(targets, predictions))

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(features)
    avg_probabilities = np.mean(probabilities, axis=0)
    drug_types = model.classes_

    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(drug_types, avg_probabilities, color='skyblue')
    plt.title("Average Probability for Each Drug Type")
    plt.xlabel("Drug Type")
    plt.ylabel("Probability")
    plt.show()
else:
    print("The model does not support probability predictions.")
