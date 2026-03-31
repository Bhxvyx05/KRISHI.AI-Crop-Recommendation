# 🌾 KRISHI.AI: Explainable Crop Recommendation System

KRISHI.AI is a machine learning-based system that recommends the most suitable crop based on soil nutrients and environmental conditions. The system also uses Explainable AI (SHAP) to provide insights into why a particular crop is recommended.

---

## 📌 Introduction

Agriculture productivity depends on several factors such as soil nutrients, climate, and water availability. Selecting the right crop is essential for maximizing yield. Traditional methods rely on experience and may not always provide optimal results.

KRISHI.AI provides a data-driven solution using machine learning to assist in crop selection, along with explanation support to improve transparency and trust.

---

## ❗ Problem Statement

- Lack of data-driven crop recommendation systems  
- Limited access to expert agricultural advice  
- Difficulty in understanding AI-based predictions  
- Existing systems act as black boxes  

---

## 💡 Proposed Solution

This project uses:
- LightGBM for crop prediction  
- SHAP for explainability  
- Streamlit for user interaction  

The system predicts crops based on input parameters and explains the reasoning behind predictions.

---

## 🎯 Objectives

- Build a crop recommendation model  
- Achieve high prediction accuracy  
- Provide interpretable results using SHAP  
- Develop an interactive application  

---

## ⚙️ System Overview

Inputs:
- Nitrogen (N)  
- Phosphorus (P)  
- Potassium (K)  
- Temperature  
- Humidity  
- pH  
- Rainfall  

Output:
- Recommended crop  

---

## 📊 Dataset Description

The dataset includes:

- N: Nitrogen content  
- P: Phosphorus content  
- K: Potassium content  
- temperature: Temperature (°C)  
- humidity: Humidity (%)  
- ph: Soil pH  
- rainfall: Rainfall (mm)  
- label: Crop name  

---

## 🔬 Methodology

1. Load dataset  
2. Preprocess data  
3. Split into train and test sets  
4. Train LightGBM model  
5. Evaluate model performance  
6. Apply SHAP for explainability  
7. Deploy using Streamlit  

---

## 🧠 Model Used

LightGBM (Gradient Boosting):
- High accuracy  
- Fast training  
- Handles nonlinear relationships  

---

## 🔍 Explainable AI

SHAP is used to:
- Show feature importance  
- Explain model predictions  
- Improve trust in results  

Example:
- High rainfall → Rice  
- High potassium → Banana  

---

## 📈 Results

- Accuracy: ~98–99%  
- Better than traditional models (SVM, KNN, Random Forest)  

---

## 💡 Example

Input:
N = 90  
P = 42  
K = 43  
temperature = 20  
humidity = 82  
ph = 6.5  
rainfall = 202  

Output:
Recommended Crop: Rice  

---

## 🌍 Applications

- Precision agriculture  
- Smart farming  
- Agricultural advisory systems  
- Educational projects  

---

## ⚠️ Limitations

- Depends on dataset quality  
- No real-time weather integration  
- Limited regional adaptability  

---

## 🚀 Future Scope

- Mobile application  
- IoT integration  
- Region-specific recommendations  
- Multi-language support  

---

## ⭐ Conclusion

KRISHI.AI combines machine learning with explainable AI to provide accurate and interpretable crop recommendations. It helps bridge the gap between prediction and understanding in agricultural decision-making.

---
