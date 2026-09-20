# Heart Failure Clinical Records EDA & Machine Learning

Reproducible Python portfolio project for **exploratory data analysis** and **binary classification** on the Heart Failure Clinical Records dataset.

## Overview

This project takes a clinical heart-failure dataset and walks through a complete analytics workflow:

- Data validation & descriptive statistics
- Exploratory data analysis and visualisation
- Multiple classification models with proper evaluation
- Hierarchical clustering

It is designed as an educational / portfolio project only and is **not** a clinical decision-making system.

## Techniques Used

- Data validation and descriptive statistics
- EDA and visualisation (boxplots, count plots, correlation heatmap)
- Models: Logistic Regression, Random Forest, Decision Tree, LDA, Perceptron
- Stratified train/test split + 5-fold cross-validation
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrices, ROC curves, Precision–Recall curves
- Hierarchical clustering (Ward linkage) by gender

## Project Structure

```
heart-failure-ml-analysis/
├── data/                  # Place the CSV dataset here
├── src/
│   ├── data.py            # Loading & data quality checks
│   ├── eda.py             # Exploratory plots
│   ├── models.py          # Classification pipeline
│   └── clustering.py      # Hierarchical clustering
├── main.py                # Entry point
├── requirements.txt
└── README.md
```

## How to Run

1. Place the dataset at:
   ```
   data/heart_failure_clinical_records_dataset.csv
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis:
   ```bash
   python main.py
   ```

Results (tables + plots) are written to the `outputs/` folder.

## Skills Demonstrated

- Python (pandas, scikit-learn, matplotlib, seaborn, scipy)
- Exploratory Data Analysis
- Supervised Machine Learning (classification)
- Model evaluation & cross-validation
- Hierarchical clustering
- Reproducible project structure



**Heart Failure Clinical Records EDA & Machine Learning | Python**  
Performed end-to-end exploratory analysis and binary classification on heart failure clinical data. Built multiple models (Logistic Regression, Random Forest, Decision Tree, LDA, Perceptron), evaluated them with stratified CV and standard metrics, and added hierarchical clustering for patient segmentation.
