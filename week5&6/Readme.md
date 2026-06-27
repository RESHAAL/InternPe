# 📧 Spam Mail Detection Dashboard

A Machine Learning-based web application that detects whether an email or SMS message is **Spam** or **Ham (Legitimate)**. This project is developed using **Python**, **Scikit-learn**, and **Streamlit** with an interactive dashboard for prediction and model analysis.

---

## 🚀 Features

* Interactive and responsive Streamlit dashboard
* Email/SMS spam prediction
* Confidence score for each prediction
* Spam risk meter
* Email statistics (words, characters, links, numbers, special symbols, etc.)
* Explainable AI showing words influencing the prediction
* Suspicious keyword detection
* Prediction probability visualization
* Model comparison (Logistic Regression, Naive Bayes, Linear SVM)
* Confusion Matrix visualization
* Feature importance analysis
* Prediction history
* Download prediction report as CSV

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Plotly
* TF-IDF Vectorizer

---

## 📂 Dataset

The project uses the **SMS Spam Collection Dataset**, which contains labeled SMS messages categorized as:

* Ham (Legitimate Messages)
* Spam (Unwanted Messages)

---

## 🤖 Machine Learning Models

* Logistic Regression
* Multinomial Naive Bayes
* Linear Support Vector Machine (SVM)

Logistic Regression is used as the final prediction model based on its performance.

---

## 📊 Dashboard Highlights

* Dataset Overview
* Email Category Distribution
* Model Accuracy Comparison
* Prediction Confidence
* Spam Risk Meter
* Email Statistics
* Explainable AI
* Confusion Matrix
* Feature Importance
* Prediction History
* Downloadable Prediction Report

---

## ▶️ How to Run

1. Clone the repository

```bash
git clone <repository-link>
```

2. Install the required libraries

```bash
pip install -r requirements.txt
```

3. Run the application

```bash
streamlit run spam_email_detection.py
```

---

## 📌 Project Outcome

This project demonstrates how Machine Learning can be integrated with an interactive web interface to accurately classify spam messages while also providing visual insights into model performance and prediction reasoning.

---

### Internship

**InternPe – Week 5 & 6 Project**

Developed as part of the Machine Learning Internship Program.
