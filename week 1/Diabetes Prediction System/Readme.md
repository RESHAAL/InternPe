# 🩺 Diabetes Risk Screening System

A web-based diabetes screening application developed using Python, Streamlit, and Machine Learning.

This project was created to provide a simple and structured way to analyze basic health parameters and estimate diabetes risk using predictive analytics. The application follows a clean healthcare-style interface inspired by modern clinical dashboards.

---

## Overview

The system allows users to enter patient health details such as glucose level, BMI, insulin level, age, and blood pressure. Based on these parameters, the model performs a preliminary diabetes risk assessment and displays the result along with basic health recommendations.

The project focuses not only on prediction accuracy but also on creating a professional and user-friendly healthcare interface.

---

## Features

- Professional medical dashboard UI
- Diabetes risk prediction using Machine Learning
- Patient screening workflow
- Risk probability indicator
- Preventive health recommendations
- Clinical analytics and visualizations
- Interactive Streamlit web application

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Machine Learning Model

The prediction model is built using:

- Support Vector Machine (SVM)
- StandardScaler for data preprocessing

The model is trained using the PIMA Indians Diabetes Dataset.

---

## Parameters Considered

The system evaluates the following health parameters:

- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
