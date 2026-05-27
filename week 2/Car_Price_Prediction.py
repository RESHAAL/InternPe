import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

# PAGE CONFIG

st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# CUSTOM CSS

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    background: linear-gradient(to right, #ff4b2b, #ff416c);
    color: white;
    font-size: 18px;
    border: none;
}

.prediction-box {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(to right, #1f4037, #99f2c8);
    color: black;
    font-size: 25px;
    text-align: center;
    font-weight: bold;
}

.insight-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #262730;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# TITLE

st.title("🚗 AI Powered Used Car Price Prediction System")

# LOAD DATA

car = pd.read_csv(r"C:\Users\ASUS\OneDrive\Pictures\Screenshots\Documents\InternPe\WEEK 2\quikr_car.csv")

# DATA CLEANING

car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)

car = car[car['Price'] != 'Ask For Price']

car['Price'] = car['Price'].str.replace(',', '')
car['Price'] = car['Price'].astype(int)

car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')

car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)

car = car[~car['fuel_type'].isna()]

car['name'] = car['name'].str.split(' ').str.slice(0,3).str.join(' ')

# NEW FEATURE : CAR AGE

current_year = datetime.datetime.now().year
car['car_age'] = current_year - car['year']

# FEATURES

X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type', 'car_age']]
y = car['Price']

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# COLUMN TRANSFORMER

ohe = OneHotEncoder(handle_unknown='ignore')

column_trans = make_column_transformer(
    (ohe, ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# MULTIPLE MODELS

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor()
}

scores = {}
trained_models = {}

for model_name, model in models.items():

    pipe = make_pipeline(column_trans, model)

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    score = r2_score(y_test, y_pred)

    scores[model_name] = score
    trained_models[model_name] = pipe

# BEST MODEL

best_model_name = max(scores, key=scores.get)
best_model = trained_models[best_model_name]

# SIDEBAR

st.sidebar.header("⚙️ Enter Car Details")

companies = sorted(car['company'].unique())
company = st.sidebar.selectbox("Company", companies)

car_models = sorted(car[car['company'] == company]['name'].unique())
name = st.sidebar.selectbox("Car Model", car_models)

year = st.sidebar.slider("Manufacturing Year", 1995, current_year, 2018)

kms_driven = st.sidebar.number_input("Kilometers Driven", 0, 500000)

fuel = st.sidebar.selectbox(
    "Fuel Type",
    car['fuel_type'].unique()
)

# PREDICT BUTTON

if st.sidebar.button("🚀 Predict Price"):

    car_age = current_year - year

    input_df = pd.DataFrame(
        [[name, company, year, kms_driven, fuel, car_age]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type', 'car_age']
    )

    prediction = best_model.predict(input_df)[0]

    prediction = round(prediction)

    # PRICE CATEGORY

    if prediction < 300000:
        category = "💰 Budget Car"

    elif prediction < 700000:
        category = "🚘 Mid-Range Car"

    elif prediction < 1500000:
        category = "🔥 Premium Car"

    else:
        category = "👑 Luxury Car"

    # PREDICTION OUTPUT

    st.markdown(
        f"""
        <div class="prediction-box">
        Predicted Price: ₹ {prediction:,}
        <br><br>
        {category}
        </div>
        """,
        unsafe_allow_html=True
    )

    # SMART INSIGHTS

    insight = ""

    if kms_driven > 100000:
        insight += "⚠️ High kilometers reduce resale value.<br>"

    if car_age < 5:
        insight += "✅ Newer cars generally have better resale value.<br>"

    if fuel == "Diesel":
        insight += "⛽ Diesel cars are preferred for long-distance travel.<br>"

    if fuel == "Petrol":
        insight += "🚗 Petrol cars usually have lower maintenance costs.<br>"

    st.markdown(
        f"""
        <div class="insight-box">
        <h3>🧠 AI Insights</h3>
        {insight}
        </div>
        """,
        unsafe_allow_html=True
    )

    # RECOMMENDATION SYSTEM

    recommendations = car[
        (car['company'] == company)
    ][['name', 'Price']].sort_values(by='Price').head(5)

    st.subheader("🚘 Similar Recommended Cars")

    st.dataframe(recommendations)

# MODEL COMPARISON

st.subheader("📊 Model Accuracy Comparison")

score_df = pd.DataFrame({
    "Model": list(scores.keys()),
    "R2 Score": list(scores.values())
})

st.dataframe(score_df)

# VISUALIZATION

fig, ax = plt.subplots()

ax.bar(score_df['Model'], score_df['R2 Score'])

ax.set_ylabel("Accuracy Score")
ax.set_title("Model Performance")

st.pyplot(fig)

# BEST MODEL

st.success(f"🏆 Best Model Selected: {best_model_name}")

# SAVE MODEL

pickle.dump(best_model, open('model.pkl', 'wb'))