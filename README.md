# Iris Flower Classification using Machine Learning

# 📌 Project Overview

This project is a Machine Learning classification system that predicts the species of Iris flowers based on their physical measurements. It uses the famous Iris dataset and multiple ML algorithms to classify flowers into three species:

Iris-setosa

Iris-versicolor

Iris-virginica

A Streamlit web application is also built to provide an interactive dashboard for predictions and data visualization.

# 🎯 Objective

Use flower measurements as input data

Train machine learning models for classification

Compare multiple ML algorithms

Evaluate model performance using accuracy metrics

Build an interactive web dashboard using Streamlit

Visualize data insights and predictions

# 📂 Dataset Information

The dataset contains 150 samples with the following features:

# Feature	Description

Sepal Length (cm)	Length of sepal

Sepal Width (cm)	Width of sepal

Petal Length (cm)	Length of petal

Petal Width (cm)	Width of petal

Species	Target class (flower type)

# Species:

Iris-setosa

Iris-versicolor

Iris-virginica

# ⚙️ Technologies Used

Python 🐍

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Streamlit

Joblib

# 🤖 Machine Learning Models Used

The following models were trained and compared:


Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Decision Tree

Random Forest

# 📊 Project Workflow

Data Loading & Exploration

Data Cleaning & Preprocessing

Feature Selection

Train-Test Split

Model Training

Model Evaluation

Accuracy Comparison

Visualization (Graphs & Heatmaps)

Model Deployment using Streamlit

# 📈 Results

Models achieved high accuracy due to clean dataset

Random Forest and SVM performed best

Confusion matrix used for performance evaluation

Final model selected based on accuracy comparison

# 🖥️ Streamlit Dashboard Features

The web application includes:


Interactive sliders for input

Real-time prediction

Dataset preview

Accuracy comparison graph


Confusion matrix visualization

Feature distribution plots

Clean UI with custom styling

# 🚀 How to Run the Project

1. Clone Repository

git clone  https://github.com/your-username/iris-flower-classification.git
cd iris-flower-classification

3. Install Dependencies
   
pip install -r requirements.txt

5. Train Model (if needed)
   
python src/train_model.py

7. Run Prediction Script
   
python src/predict.py

9. Run Streamlit App
    
python -m streamlit run src/app.py

# 📁 Project Structure

iris-flower-classification/
│
├── dataset/
│   └── iris.csv
│
├── images/
│
├── models/
│   └── iris_model.pkl
│
├── outputs/
│   ├── accuracy_graph.png
│   ├── confusion_matrix.png
│   ├── pairplot.png
│   └── heatmap.png
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── app.py
│
├── requirements.txt
└── README.md

# 📌 Key Learnings

Understanding classification algorithms

Model training and evaluation

Data visualization techniques

Building ML web apps using Streamlit

End-to-end ML project workflow

# 🎯 Future Improvements

Add more ML models

Deploy on cloud (Heroku / Streamlit Cloud)

Add user login system

Improve UI with animations

Add downloadable prediction report

# 👨‍💻 Author

Kumkum Manjhi

Internship Project – Data Science


