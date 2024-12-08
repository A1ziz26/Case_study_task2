# Machine Learning Segmentation Task

This repository contains the solution for a segmentation prediction task using various machine learning models and techniques. The project involves data preprocessing, feature engineering, model training, and evaluation.

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Code Description](#code-description)
- [Running the Code](#running-the-code)
- [Model Comparison](#model-comparison)
- [Results and Analysis](#results-and-analysis)
- [Kaggle Notebook Link](#kaggle-notebook-link)

---

## Setup Instructions

1. Clone this repository to your local machine:
   ```bash
   git clone <repository-link>
   cd <repository-directory>

Code Description
The code is divided into the following sections:

Data Preprocessing: Handling missing values, encoding categorical features, and scaling numerical features.
Feature Engineering: Creating new features to improve model performance.
Model Training and Evaluation: Training multiple models like Random Forest, LightGBM, SVM, and Neural Networks, and evaluating their performance on validation datasets.
Visualization: Comparison of model accuracies using bar plots.
Test Predictions: Using the best model to predict the segmentation on the test dataset.
Running the Code
To run the code, execute the main_notebook.ipynb or the corresponding .py files using the following command:

bash
Copy code
jupyter notebook main_notebook.ipynb
Alternatively, you can run the .py file:

bash
Copy code
python main_script.py
Make sure the dataset files are in the correct directory as mentioned in the setup.

The predictions for the test dataset will be saved as submission.csv.

Model Comparison
The models were trained on three datasets:

Complete Dataset (train_data): Contains all segments (0, 1, 2, and 3).
Subset Dataset 1 (train_data1): Contains segments 2 and 3.
Subset Dataset 2 (train_data2): Contains segments 0 and 1.
Validation Accuracies
Model	Complete Dataset	Segments 2 & 3	Segments 0 & 1
Random Forest	0.51	0.8372	0.6383
LightGBM	0.5366	0.8436	0.6514
SVM (RBF kernel)	0.5101	0.8301	0.635
XGBoost	0.5111	0.8349	0.60
Neural Networks	0.5279	0.8431	0.6083
KNN	0.5049	0.827	0.66
GaussianNB	0.488	0.8231	0.6409
Logistic Regression	0.4851	0.8207	0.635
Results and Analysis
The subset datasets (train_data1 and train_data2) performed significantly better than the complete dataset.
The best validation accuracy for train_data1 (2 & 3) was achieved with LightGBM: 84.36%.
The best validation accuracy for train_data2 (0 & 1) was achieved with KNN: 66.00%.
Kaggle Notebook Link
The clean and final notebook can be found on Kaggle: Kaggle Notebook Link

