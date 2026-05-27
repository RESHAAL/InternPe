# 🚗 Car Price Prediction System

A machine learning based web application that predicts the resale price of used cars using different regression models and an interactive Streamlit dashboard.

The project focuses on building a practical and user-friendly prediction system with data cleaning, feature engineering, model comparison, and smart resale insights.

---

## 📌 Features

- Predicts used car prices based on:
  - Company
  - Car Model
  - Manufacturing Year
  - Kilometers Driven
  - Fuel Type

- Interactive Streamlit interface

- Multiple ML models used:
  - Linear Regression
  - Random Forest Regressor
  - Decision Tree Regressor

- Automatic best model selection using R² Score

- Car Age feature engineering

- Smart prediction insights

- Price category classification:
  - Budget Car
  - Mid-Range Car
  - Premium Car
  - Luxury Car

- Similar car recommendation system

- Model performance visualization

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

## ⚙️ How It Works

### 1. Data Cleaning
The dataset is cleaned by:
- removing invalid values
- converting text values into numeric format
- handling missing fuel types
- extracting useful car name information

### 2. Feature Engineering
A new feature called `car_age` is created using:

```python
car_age = current_year - manufacturing_year
```

This helps improve prediction accuracy.

### 3. Model Training
Different regression models are trained and evaluated using R² Score.

The model with the best performance is automatically selected for predictions.

### 4. Prediction System
The user enters car details through the Streamlit dashboard, and the system predicts the estimated resale value instantly.

---

## 📊 Dashboard Features

- Interactive sidebar input system
- Real-time price prediction
- AI-based resale insights
- Similar car suggestions
- Accuracy comparison graph
- Best model display

---

## ▶️ Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python -m streamlit run Car_Price_Prediction.py
```

---

## 📈 Models Comparison

The project compares model performance using R² Score visualization to select the most accurate model for prediction.

---

## 💡 Example Insights

- High kilometers reduce resale value
- Newer cars generally have better resale value
- Petrol cars usually have lower maintenance costs
- Diesel cars are preferred for long-distance travel

---

## 🚀 Future Improvements

- Live market price integration
- Car image upload support
- Deep learning models
- User authentication system
- Deployment on cloud platforms

---

## 📚 Dataset

Dataset used: Quikr Used Cars Dataset

---

## 👨‍💻 Developed By

Reshal Age

---

## ⭐ If you found this project useful

Feel free to star the repository.
