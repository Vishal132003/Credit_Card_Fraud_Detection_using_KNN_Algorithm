# 🕵️‍♂️ Fraud or Risk Detection using KNN

## 📘 Overview
This project focuses on detecting fraudulent transactions or risky financial activities using the **K-Nearest Neighbors (KNN)** classification algorithm.  
The dataset is taken from **Kaggle**, and the main goal is to classify whether a transaction is *fraudulent* or *legitimate* based on given features.

---

## 🎯 Objective
- To identify potentially **fraudulent transactions** using KNN.
- To apply **data preprocessing** and evaluate model performance.
- To understand how **distance-based classification** helps in fraud detection.

---

## 🧩 Dataset
**Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Description:**
- The dataset contains transactions made by European credit cardholders in September 2013.
- Features are numerical values obtained through **PCA transformation**.
- Target variable:  
  - `0` → Normal Transaction  
  - `1` → Fraudulent Transaction  

---

## ⚙️ Steps Involved
### 1. Importing Libraries
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn  

### 2. Data Preprocessing
- Checked and removed **null values**
- Removed **duplicate rows**
- Performed **feature scaling (StandardScaler)**

### 3. Data Splitting
- 80% training data  
- 20% testing data  

### 4. Model Training
Used **KNeighborsClassifier** from `sklearn.neighbors`:
```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

## 📂 Dataset
Download the dataset from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
