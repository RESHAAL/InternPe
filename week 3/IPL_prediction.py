import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ====================================================
# PAGE CONFIG
# ====================================================

st.set_page_config(
    page_title="IPL Winner Prediction",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 IPL Winning Team Prediction")
st.write("Machine Learning based IPL Match Winner Predictor")

# ====================================================
# LOAD DATA
# ====================================================

@st.cache_data
def load_data():

    matches = pd.read_csv("matches.csv")

    return matches

matches = load_data()

# ====================================================
# DATA CLEANING
# ====================================================

matches = matches[[
    'team1',
    'team2',
    'toss_winner',
    'toss_decision',
    'venue',
    'winner'
]]

matches.dropna(inplace=True)

# ====================================================
# ENCODING
# ====================================================

team_encoder = LabelEncoder()
venue_encoder = LabelEncoder()
decision_encoder = LabelEncoder()
winner_encoder = LabelEncoder()

all_teams = pd.concat([
    matches['team1'],
    matches['team2'],
    matches['toss_winner'],
    matches['winner']
])

team_encoder.fit(all_teams)

matches['team1'] = team_encoder.transform(matches['team1'])
matches['team2'] = team_encoder.transform(matches['team2'])
matches['toss_winner'] = team_encoder.transform(matches['toss_winner'])

matches['venue'] = venue_encoder.fit_transform(matches['venue'])
matches['toss_decision'] = decision_encoder.fit_transform(matches['toss_decision'])

matches['winner'] = winner_encoder.fit_transform(matches['winner'])

# ====================================================
# TRAIN MODEL
# ====================================================

X = matches.drop('winner', axis=1)
y = matches['winner']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

# ====================================================
# SIDEBAR
# ====================================================

st.sidebar.header("Model Information")
st.sidebar.success(f"Accuracy: {accuracy*100:.2f}%")

# ====================================================
# TEAM LIST
# ====================================================

teams = sorted(team_encoder.classes_)

# ====================================================
# USER INPUTS
# ====================================================

st.subheader("Match Details")

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Team 1", teams)

with col2:
    team2 = st.selectbox("Team 2", teams)

toss_winner = st.selectbox(
    "Toss Winner",
    [team1, team2]
)

toss_decision = st.selectbox(
    "Toss Decision",
    decision_encoder.classes_
)

venue = st.selectbox(
    "Venue",
    venue_encoder.classes_
)

# ====================================================
# PREDICTION
# ====================================================

if st.button("Predict Winner"):

    input_data = pd.DataFrame({
        'team1': [team_encoder.transform([team1])[0]],
        'team2': [team_encoder.transform([team2])[0]],
        'toss_winner': [team_encoder.transform([toss_winner])[0]],
        'toss_decision': [decision_encoder.transform([toss_decision])[0]],
        'venue': [venue_encoder.transform([venue])[0]]
    })

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    winner = winner_encoder.inverse_transform([prediction])[0]

    confidence = max(probabilities) * 100

    st.success(f"🏆 Predicted Winner: {winner}")

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

# ====================================================
# DATA PREVIEW
# ====================================================

st.subheader("Dataset Preview")

st.dataframe(matches.head())
