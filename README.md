# Predictive Model for Treatment Adherence in Non-Small Cell Lung Cancer (NSCLC) Patients using Osimertinib
Predictive analytics model leveraging a neural network to identify patients at risk of prematurely ending Osimertinib treatment for non-small cell lung cancer (NSCLC). Achieved an AUC of 0.93 on validation data through focused feature selection. This project was part of the Humana-Mays Healthcare Analytics Case Competition, where our team placed in the top 50 of 150 teams.

## Overview
Osimertinib, a targeted therapy for non-small cell lung cancer (NSCLC), has shown efficacy but is associated with adverse effects such as nausea, diarrhea, and seizures. Patient adherence to Osimertinib treatment is crucial for optimal outcomes, but many patients discontinue prematurely due to these side effects. Our project aims to develop a predictive model to identify patients at risk of treatment discontinuation and provide recommendations to enhance adherence.

## Model & Data Processing
We utilized a neural network to predict patients at risk of ending Osimertinib treatment prematurely. The model considered patient demographics, physician visits, and prescription data. Through feature selection, we achieved an impressive Area Under the Curve (AUC) of 0.93 on validation data, demonstrating the model's ability to identify high-risk patients.

The dataset includes information on patient demographics, pharmacy claims, and medical diagnoses. We preprocessed the data, handling missing values, encoding categorical variables, and engineering features to create a comprehensive input for the predictive model.

## Key Features
- Neural Network Model: Utilized a neural network for predicting treatment adherence.
- Feature Selection: Identified 11 key variables associated with treatment discontinuation.
- Visualizations: Generated insightful visualizations, including age vs. drop-off rates and race vs. drop-off rates.
- Model Evaluation: Achieved an AUC of 0.93 on validation data, showcasing the model's effectiveness.

## Environment and Dependencies
This project was developed using Python version 3.7. The following Python packages and libraries were utilized for data analysis and statistical modeling:
- `**pandas**`
- `**scikit-learn**`
- `**numpy**`
- `**matplotlib**`
- `**seaborn**`
- `**tensorflow**`
