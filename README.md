# ⚡ Electricity Price Prediction | Deep Learning Model

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Library](https://img.shields.io/badge/Library-Keras%20%2F%20TensorFlow-orange.svg)](https://keras.io/)
[![Status](https://img.shields.io/badge/Status-Completed-green.svg)]()

## 📋 Overview
This project implements a **Deep Learning solution** to forecast electricity prices based on historical data. By analyzing time-series patterns and consumption metrics, the model aims to predict future pricing trends, providing actionable insights for energy market analysis.

This repository contains the complete source code, data processing pipelines, and a detailed analysis report.

## 📂 Repository Structure
* **`Electricity_Price_Prediction.ipynb`**: The core Jupyter Notebook containing:
  * Data Loading & Cleaning (Pandas)
  * Exploratory Data Analysis (Visualization)
  * Neural Network Construction (Keras)
  * Model Training & Evaluation
* **`Project_Report_Analysis.pdf`**: A comprehensive report detailing the methodology, mathematical concepts, and interpretation of results.
* **`data/`**: Directory containing the training and testing datasets (`X_train.csv`, `y_train.csv`, etc.).

## 🛠 Tech Stack & Libraries
This project relies on a robust Python ecosystem for Data Science:
* **Deep Learning:** `Keras`, `TensorFlow` (Neural Network Architecture)
* **Data Manipulation:** `Pandas`, `NumPy` (Preprocessing & Normalization)
* **Visualization:** `Matplotlib`, `Seaborn` (Data insights & Loss curves)
* **Machine Learning:** `Scikit-Learn` (Train/Test splitting, Metrics)

## ⚙️ Key Features
1.  **Data Preprocessing Pipeline:**
    * Handling missing values and outliers.
    * Feature scaling/normalization to optimize Neural Network convergence.
2.  **Neural Network Architecture:**
    * Implementation of a Multi-Layer Perceptron (MLP).
    * Tuning of hyperparameters (Epochs, Batch Size, Learning Rate).
3.  **Performance Evaluation:**
    * Analysis of Loss Functions (MSE/MAE) to validate model accuracy.
    * Visual comparison between Predicted vs. Actual prices.

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/DeepLearning-Electricity-Prediction.git](https://github.com/YOUR_USERNAME/DeepLearning-Electricity-Prediction.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy tensorflow matplotlib seaborn scikit-learn
    ```
3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook "Electricity_Price_Prediction.ipynb"
    ```

## 📊 Results & Analysis
The model demonstrates the ability to capture non-linear relationships in the electricity market data. For a deep dive into the mathematical models and detailed performance metrics, please refer to the **[Project Report (PDF)](./Project_Report_Analysis.pdf)** included in this repository.

---
*Author: Yassine Jenjare*