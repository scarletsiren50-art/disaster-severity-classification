# Comparative Analysis of Machine Learning Models for Disaster Severity Classification

## 📌 Overview

This project presents a systematic comparative analysis of classical
Machine Learning (ML) and Deep Learning (DL) models for multi-class
disaster severity classification using satellite imagery from the xView2
dataset.

The study evaluates model performance under moderate dataset conditions
(1200 balanced building-level image crops) and investigates how dataset
scale influences deep learning generalization.

------------------------------------------------------------------------

## 🛰 Dataset

**Dataset Used:** xView2 Disaster Damage Assessment Dataset\
Official Source: https://xview2.org/dataset

### Classes:

-   No Damage
-   Minor Damage
-   Major Damage
-   Destroyed

### Dataset Preparation:

-   Building-level crop extraction using WKT polygon parsing
-   Image resizing (128×128)
-   Balanced dataset (300 samples per class)
-   80/20 Train-Validation split

Total samples used: **1200**

------------------------------------------------------------------------

## 🧠 Models Implemented

### Classical Machine Learning:

-   Logistic Regression
-   Support Vector Machine (RBF Kernel)
-   Random Forest (100 estimators)

### Deep Learning:

-   Custom Convolutional Neural Network (CNN)
-   ResNet50 (Transfer Learning -- ImageNet Pretrained)

------------------------------------------------------------------------

## ⚙️ Experimental Setup

-   Language: Python
-   Libraries: Scikit-learn, TensorFlow/Keras, OpenCV, NumPy, Matplotlib
-   Hardware: AMD Ryzen 5 3500U, 16GB RAM (CPU-based training)
-   Evaluation Metrics:
    -   Accuracy
    -   Precision
    -   Recall
    -   F1-score
    -   Confusion Matrix

------------------------------------------------------------------------

## 📊 Results

### 🔹 Final Model Comparison

![Final Comparison](results/graphs/final_model_comparison.png)

Random Forest achieved the highest validation accuracy (63.33%),
outperforming CNN and ResNet50 under moderate dataset size.

------------------------------------------------------------------------

### 🔹 Random Forest Confusion Matrix

![Random Forest CM](results/graphs/random_forest_confusion_matrix.png)

Strong diagonal dominance indicates stable per-class prediction
performance.

------------------------------------------------------------------------

### 🔹 ResNet50 Confusion Matrix

![ResNet CM](results/graphs/resnet_confusion_matrix.png)

Inter-class confusion observed between minor_damage and major_damage,
indicating dataset-scale limitations for deep transfer learning.

------------------------------------------------------------------------

## 📈 Validation Accuracy Summary

  Model                 Validation Accuracy
  --------------------- ---------------------
  Logistic Regression   48.75%
  SVM                   59.17%
  Random Forest         **63.33%**
  CNN                   51.25%
  ResNet50              55.83%

------------------------------------------------------------------------

## 🚀 How to Run

### Install Dependencies

pip install -r requirements.txt

### Run Baseline ML Models

python main_week5.py

### Run CNN Model

python main_week6.py

### Run ResNet50

python main_week8.py

### Generate Final Comparison Graph

python final_comparison.py

------------------------------------------------------------------------

## 👩‍💻 Author

Md. Sadiya Tabassum\
BTech Special Project\
Comparative Analysis of ML Models for Disaster Severity Classification

------------------------------------------------------------------------

## 📜 License

This project is for academic research purposes.
