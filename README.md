# 🌾 KRISHI.AI: An Explainable AI Framework for Precision Crop Recommendation

KRISHI.AI is an intelligent crop recommendation system that leverages machine learning and explainable AI techniques to suggest the most suitable crops based on soil nutrients and environmental conditions. The system is designed to provide not only accurate predictions but also interpretable insights, enabling users to understand the reasoning behind each recommendation.

---

## 📌 Table of Contents

- Introduction
- Problem Statement
- Proposed Solution
- Objectives
- System Overview
- Dataset Description
- Methodology
- Machine Learning Model
- Explainable AI (SHAP)
- Results and Performance
- Project Structure
- Installation and Setup
- Usage Instructions
- Example Output
- Applications
- Limitations
- Future Scope
- Contributors

---

## 🌍 Introduction

Agriculture plays a vital role in sustaining human life and economic stability, especially in countries like India. The productivity of crops is highly dependent on multiple factors such as soil composition, climate conditions, and water availability. Selecting the appropriate crop for a given set of conditions is a complex decision that directly impacts yield and profitability.

With the advancement of artificial intelligence, machine learning-based systems have been introduced to support crop selection. However, most existing solutions focus only on prediction accuracy and lack interpretability, making it difficult for users to trust and understand the recommendations.

---

## ❗ Problem Statement

Traditional crop selection methods face several challenges:

- Lack of data-driven decision-making
- Poor accessibility to expert agricultural advice
- Inability to handle complex environmental interactions
- Lack of transparency in AI-based systems

Most machine learning models act as "black boxes," providing predictions without explaining the reasoning behind them. This reduces trust among users, especially farmers and non-technical stakeholders.

---

## 💡 Proposed Solution

KRISHI.AI addresses these challenges by:

- Using machine learning (LightGBM) for accurate crop prediction
- Incorporating explainable AI (SHAP) to interpret model decisions
- Providing a user-friendly interface using Streamlit
- Enabling real-time prediction based on user inputs

---

## 🎯 Objectives

- To develop a machine learning model for crop recommendation
- To achieve high predictive accuracy using ensemble methods
- To enhance interpretability using SHAP
- To build an interactive system accessible to users

---

## ⚙️ System Overview

The system takes the following inputs:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

Based on these inputs, the system predicts the most suitable crop and explains the contributing factors.

---

## 📊 Dataset Description

The dataset used is a crop recommendation dataset containing multiple agronomic features.

| Feature      | Description                          |
|-------------|--------------------------------------|
| N           | Nitrogen content (kg/ha)             |
| P           | Phosphorus content (kg/ha)           |
| K           | Potassium content (kg/ha)            |
| temperature | Temperature in Celsius               |
| humidity    | Relative humidity (%)                |
| ph          | Soil pH value                        |
| rainfall    | Rainfall (mm)                        |
| label       | Crop type                            |

---

## 🔬 Methodology

The overall workflow of the system includes:

1. Data Collection and Loading
2. Data Preprocessing
3. Feature Selection
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Explainability Analysis
8. Deployment

---

## 🧠 Machine Learning Model

The system uses:

### 🔹 LightGBM (Gradient Boosting)

LightGBM is an efficient gradient boosting framework that:

- Handles large datasets efficiently
- Captures nonlinear relationships
- Provides high accuracy
- Works well with structured data

---

## 🔍 Explainable AI (SHAP)

To make predictions interpretable, SHAP (SHapley Additive Explanations) is used.

### SHAP provides:

- Feature importance
- Contribution of each input feature
- Visualization of decision-making

### Example Insights:

- High rainfall → Rice recommendation
- High potassium → Banana suitability
- Low rainfall → Chickpea preference

---

## 📈 Results and Performance

- Accuracy: **~98–99%**
- Model: LightGBM
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score

The model performs better than traditional algorithms like:
- Random Forest
- SVM
- KNN

---

## 📂 Project Structure
