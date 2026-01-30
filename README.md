# Electricity Price Prediction - Deep Learning Model

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

## Project Overview

This project implements a **Deep Learning solution** designed to forecast electricity prices based on historical time-series data. By leveraging neural network architectures, the model analyzes consumption metrics and temporal patterns to predict future pricing trends, providing actionable insights for energy market analysis.

## Repository Structure

* **`Electricity_Price_Prediction.ipynb`**: The core Jupyter Notebook containing the end-to-end pipeline:
    * Data ingestion and cleaning (Pandas).
    * Exploratory Data Analysis (EDA) and visualization.
    * Neural Network construction and training (Keras/TensorFlow).
    * Model evaluation and metric analysis.
* **`Project_Report_Analysis.pdf`**: A comprehensive report detailing the methodology, mathematical foundations, and interpretation of the results.
* **`data/`**: Directory containing the training and testing datasets.

## Technical Stack

This project relies on a standard Data Science ecosystem:
* **Deep Learning:** `Keras`, `TensorFlow` (Multi-Layer Perceptron architecture).
* **Data Manipulation:** `Pandas`, `NumPy` (Preprocessing & Normalization).
* **Visualization:** `Matplotlib`, `Seaborn` (Loss curves & regression plots).
* **Machine Learning:** `Scikit-Learn` (Train/Test splitting, Metrics).

## Methodology & Features

The solution follows a structured machine learning pipeline:

1.  **Data Preprocessing:**
    * Imputation of missing values and outlier detection.
    * Feature scaling/normalization to optimize gradient descent convergence.
2.  **Neural Network Architecture:**
    * Implementation of a Dense Neural Network (MLP).
    * Hyperparameter tuning (Epochs, Batch Size, Learning Rate).
3.  **Performance Evaluation:**
    * rigorous analysis using Loss Functions (MSE/MAE).
    * Visual comparison between Predicted values vs. Actual market prices.

## Getting Started

**1. Clone the repository**
```bash
git clone [https://github.com/YassineJRE/DeepLearning-Electricity-Prediction.git](https://github.com/YassineJRE/DeepLearning-Electricity-Prediction.git)
