## **📌 Rain Prediction in Australia — Machine Learning & Data Mining Project**

![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Classification-green)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blueviolet)
![License](https://img.shields.io/badge/License-MIT-success)

A complete **Knowledge Discovery & Data Mining (CS-513)** course project that predicts **whether it will rain tomorrow in Australia** using a real-world meteorological dataset and several machine learning classification models.

This project performs:

* Exploratory Data Analysis
* Missing value handling
* One-hot encoding
* Balancing imbalanced classes using **RandomOverSampler**
* Training & evaluation of multiple ML models
* Final accuracy comparison

---

# 🌦️ **Project Overview**

Australia’s weather is extremely diverse and difficult to model. Using historical climate records from **Kaggle’s “WeatherAUS” dataset**, this project builds predictive models to classify whether it will rain the next day.

The final system is designed to:

* Support informed agricultural & environmental planning
* Demonstrate ML classification model performance
* Show the full KDD pipeline from raw data → knowledge

---

# 📁 **Dataset**

Dataset used: **weatherAUS.csv**
Source: Kaggle — *Weather Dataset from Rattle Package*

Key features include:

* Temperature (min/max)
* Rainfall
* Sunshine & evaporation
* Wind direction & speed
* Humidity
* Cloud cover
* Atmospheric pressure
* Binary rain labels

---

# 🔎 **Data Preprocessing**

| Step                       | Description                                                             |
| -------------------------- | ----------------------------------------------------------------------- |
| Remove date column         | Not useful for prediction                                               |
| Encode categorical columns | One-hot encoding for Location, WindGustDir, WindDir9am, WindDir3pm      |
| Convert labels             | “RainToday” and “RainTomorrow” → 0/1                                    |
| Handle missing values      | Removed all rows with NAs                                               |
| Balance dataset            | Used **RandomOverSampler** from imbalanced-learn to fix class imbalance |

---

# 🤖 **Models Implemented**

| Model               | Accuracy                  |
| ------------------- | ------------------------- |
| Logistic Regression | **0.8089**                |
| KNN (k = 3)         | **0.8446** ⭐ *Best Model* |
| KNN (k = 5)         | 0.8063                    |
| KNN (k = 10)        | 0.7824                    |
| Naive Bayes         | 0.7215                    |
| Random Forest       | 0.7923                    |
| Decision Tree       | 0.7287                    |
| XGBoost Classifier  | 0.8031                    |

**Highest Accuracy:**

### ⭐ **K-Nearest Neighbors (k = 3) — 84.46%**

---

# 📊 **Model Comparison Visualization**

*(Add this image after uploading your PPT images to GitHub)*
You can export the model comparison chart from your presentation and add it here.

---

# 🧪 **Technologies Used**

* Python
* Jupyter Notebook
* Scikit-Learn
* Imbalanced-Learn
* Pandas / NumPy
* Matplotlib / Seaborn
* XGBoost
* Kaggle Dataset

---

# 🚀 **How to Run the Notebook**

```bash
# Install dependencies
pip install -r requirements.txt

# Open Jupyter Notebook
jupyter notebook KDD_Project_Aus_Rain_Prediction.ipynb
```
