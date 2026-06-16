# Explainable AI-Based Ensemble Approach for Drug Detection System

## Overview
This project presents an Explainable AI (XAI)-based ensemble learning framework for predicting drug-target interactions. The system combines Random Forest, Support Vector Machine (SVM), and XGBoost models to improve prediction accuracy while using SHAP and LIME to provide transparent and interpretable explanations.

## Key Features
- Drug-target interaction prediction
- Ensemble Learning (Random Forest, SVM, XGBoost)
- Explainable AI using SHAP and LIME
- Molecular descriptor generation with PaDEL and RDKit
- Bioactivity classification
- Visualization of model explanations

## Workflow
1. Data Collection from ChEMBL
2. Data Preprocessing
3. Molecular Descriptor Generation
4. Model Training (RF, SVM, XGBoost)
5. Ensemble Prediction
6. Explainable AI Analysis
7. Result Visualization

## Project Structure

```text
Explainable-AI-Based-Ensemble-Approach-for-Drug-Detection-System/
│
├── data/
│
├── chembl_data_processing.py
├── padel_descriptors_calculation.py
├── RF_SVM_Classification.py
├── drug_prediction.py
├── explain_predictions.py
├── EDA_Normalization.py
├── fingerprints_xml.zip
│
├── images/
│
├── README.md
└── requirements.txt
```

## Technologies Used
- Python
- Scikit-learn
- XGBoost
- RDKit
- PaDEL Descriptor
- SHAP
- LIME
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Results
The ensemble model improves drug-target interaction prediction by combining multiple machine learning algorithms and providing interpretable explanations through SHAP and LIME visualizations.

## Future Work
- Deep Learning-based prediction models
- Multi-omics data integration
- Web-based deployment
- Real-time drug recommendation system

## Authors
- Catherine Joanna Mathews
- Nagireddy Jagruthi
- Pranavi Kunchanapalli

## Institution
Gokaraju Rangaraju Institute of Engineering and Technology (GRIET)


