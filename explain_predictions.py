#6
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer

model = joblib.load('trained_model.joblib')

features = pd.read_csv('classification_model_data.csv')

categorical_features = features.select_dtypes(include=['object']).columns
for col in categorical_features:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])

selected_features = np.load('selected_features.npy', allow_pickle=True)
features = features[selected_features]

shap_values = shap.TreeExplainer(model).shap_values(features)
shap_values_sum = shap_values.sum(axis=2)

shap.summary_plot(shap_values_sum, features)
plt.savefig('shap_summary_plot.png')
plt.close() 

feature_name = 'SubFP3'  

if feature_name in features.columns:
    shap.dependence_plot(feature_name, shap_values_sum, features)
    plt.savefig('shap_dependence_plot.png')
    plt.close()  
else:
    print(f"Feature '{feature_name}' not found in the dataset.")

explainer = LimeTabularExplainer(
    training_data=features.values,
    feature_names=features.columns.tolist(),
    mode='classification',
    class_names=model.classes_  
)

sample_index = 0  # Change this index to explain different samples
sample = features.iloc[sample_index:sample_index + 1]  

import matplotlib.pyplot as plt

# Generate LIME explanation
lime_exp = explainer.explain_instance(
    sample.values[0],
    lambda x: model.predict_proba(pd.DataFrame(x, columns=features.columns)),
    num_features=min(10, features.shape[1])
)

# Plot the explanation
fig = lime_exp.as_pyplot_figure()
plt.savefig('lime_explanation_plot.png')
plt.close()
print("LIME explanation saved as 'lime_explanation_plot.png'")
