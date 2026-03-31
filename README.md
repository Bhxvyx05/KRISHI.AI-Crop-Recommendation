# KRISHI.AI: An Explainable AI Framework for Precision Crop Recommendation

KRISHI.AI is an explainable machine learning-based crop recommendation system that suggests suitable crops using key agronomic and environmental parameters such as nitrogen, phosphorus, potassium, pH, temperature, humidity, and rainfall. The project combines predictive modeling with explainable AI techniques to provide transparent and practical decision support for agricultural use.

---

## Table of Contents

- Overview
- Objectives
- Features
- Dataset Description
- Technology Stack
- Project Structure
- Installation
- Usage
- Model Workflow
- Explainability Approach
- Results
- Future Scope
- Contributors
- License

---

## Overview

Agriculture is highly dependent on soil quality, weather conditions, and climatic variability. Selecting an appropriate crop is therefore a critical decision for farmers. KRISHI.AI addresses this need by using machine learning to recommend crops based on soil and environmental inputs, while also explaining the reasoning behind each prediction.

The system is designed to be interactive, interpretable, and suitable for demonstration in academic and practical settings.

---

## Objectives

- To recommend suitable crops using agronomic and climatic features
- To achieve high predictive performance using machine learning
- To improve transparency through explainable AI techniques
- To present recommendations through an interactive user interface

---

## Features

- Crop recommendation using LightGBM
- Explainability using SHAP
- Interactive Streamlit-based interface
- Real-time crop prediction from user inputs
- Support for model interpretation and feature analysis
- Suitable for academic demonstration and research

---

## Dataset Description

The project uses a standard crop recommendation dataset containing the following features:

- **N**: Nitrogen content in soil
- **P**: Phosphorus content in soil
- **K**: Potassium content in soil
- **temperature**: Temperature in degrees Celsius
- **humidity**: Relative humidity in percentage
- **ph**: Soil pH value
- **rainfall**: Rainfall in mm
- **label**: Recommended crop category

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- SHAP
- Streamlit
- Matplotlib
- Seaborn
- Plotly

---

## Project Structure

```text
PBL_PROJ/
├── app.py
├── Crop_recommendation.csv
├── crop-recommendation-system-using-lightgbm.ipynb
├── requirements.txt
└── README.md