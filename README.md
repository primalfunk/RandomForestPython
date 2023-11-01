# Customer Churn Prediction: An End-to-End Python Project

## Overview

This repository contains a Python-based end-to-end project focusing on predicting customer churn in a hypothetical scenario. The project covers the entire machine learning pipeline, from data generation to model evaluation and visualization.

## Table of Contents

- [Features](#features)
- [Dependencies](#dependencies)
- [Code Structure](#code-structure)
  - [Data Generation](#data-generation)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Feature Importance Visualization](#feature-importance-visualization)
- [Usage](#usage)
- [Metrics](#metrics)

## Features

The project simulates a customer churn prediction scenario with the following features:
  
- **Age**: The age of the customer (between 18 to 65).
- **Gender**: The gender of the customer (Male/Female).
- **Tenure**: The tenure of the customer with the service (in months, between 1 to 72).
- **Monthly Charges**: The amount the customer is charged monthly (between $20 to $100).
- **Feedback**: The customer's feedback about the service (Good/Average/Poor).

The target variable is **Churn**, indicating whether the customer churned (1) or not (0).

## Dependencies

The following libraries are used:
- Matplotlib
- Pandas
- NumPy
- Scikit-learn
- Seaborn

## Code Structure

### Data Generation

The `ChurnPredictionDataSpoofer` class is responsible for generating a synthetic dataset. It creates a data frame with random data for the features and the target variable.

\`\`\`python
data_gen = ChurnPredictionDataSpoofer(n_samples=1000)
data = data_gen.generate_data()
\`\`\`

### Data Preprocessing

The `ChurnDataPreprocessing` class handles preprocessing tasks such as encoding categorical variables and scaling numerical features.

\`\`\`python
preprocessor = ChurnDataPreprocessing(data)
preprocessor.encode_categorical()
preprocessor.scale_numerical()
\`\`\`

### Model Training

The `ChurnModel` class trains a Random Forest Classifier on the preprocessed data.

\`\`\`python
churn_model = ChurnModel(X_train, y_train)
churn_model.train_model()
\`\`\`

### Model Evaluation

The model's performance is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

\`\`\`python
y_pred = churn_model.predict(X_test)
metrics = churn_model.evaluate(y_test, y_pred)
\`\`\`

### Feature Importance Visualization

The `ChurnDataVisualization` class visualizes the feature importances using a bar plot.

\`\`\`python
viz = ChurnDataVisualization(churn_model.model, feature_names)
viz.plot_feature_importance()
\`\`\`

## Usage

The entire pipeline can be executed in a Python environment with the required dependencies installed.

\`\`\`python
# Generate data, preprocess it, train model, evaluate, and visualize
\`\`\`

## Metrics

The example model metrics can be printed as follows:

\`\`\`python
print(f"Model Metrics: {metrics}")
\`\`\`

The metrics are a dictionary containing the Accuracy, Precision, Recall, and F1 Score of the model.

This project serves as a comprehensive example of building a customer churn prediction model, right from data generation to feature importance visualization. Feel free to adapt and extend this code for your specific use-cases!
