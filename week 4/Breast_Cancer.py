import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from io import BytesIO
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# PAGE CONFIG

st.set_page_config(
    page_title="Breast Cancer Risk Assessment",
    page_icon="🏥",
    layout="wide"
)

# CUSTOM CSS

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}

.title-text {
    font-size:48px;
    font-weight:700;
    color:white;
    text-align:center;
}

.subtitle-text {
    font-size:20px;
    color:#d1d5db;
    text-align:center;
}

.low-risk {
    color:green;
    font-weight:bold;
}

.medium-risk {
    color:orange;
    font-weight:bold;
}

.high-risk {
    color:red;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# HEADER

st.markdown(
    """
    <div class='title-text'>
    🏥 AI-Powered Breast Cancer Risk Assessment System
    </div>

    <div class='subtitle-text'>
    Early Detection • Random Forest Prediction • Clinical Dashboard
    </div>

    <br>
    <hr>
    """,
    unsafe_allow_html=True
)

# LOAD DATASET

@st.cache_data
def load_data():

    df = pd.read_csv("dataset.csv")

    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    if "Unnamed: 32" in df.columns:
        df.drop("Unnamed: 32", axis=1, inplace=True)

    return df

df = load_data()

# DATA PREPROCESSING

df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RANDOM FOREST MODEL

@st.cache_resource
def train_model():

    model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    random_state=42
)

    model.fit(
        X_train_scaled,
        y_train
    )

    return model

model = train_model()

# MODEL PERFORMANCE

pred_prob = model.predict_proba(
    X_test_scaled
)[:,1]

pred_class = model.predict(
    X_test_scaled
)

accuracy = accuracy_score(y_test, pred_class)
precision = precision_score(y_test, pred_class)
recall = recall_score(y_test, pred_class)
f1 = f1_score(y_test, pred_class)

# SIDEBAR

st.sidebar.title("🩺 Navigation")

menu = st.sidebar.radio(
    "Choose Section",
    [
        "Patient Inputs",
        "Risk Dashboard",
        "Analytics",
        "Generate Report"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success(
    f"Model Accuracy: {accuracy*100:.2f}%"
)

# PATIENT INPUTS

def generate_pdf_report(
        prediction,
        risk_level,
        malignant_risk,
        accuracy,
        precision,
        recall,
        f1):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    content = []

    title = Paragraph(
        "Breast Cancer Risk Assessment Report",
        styles['Title']
    )

    content.append(title)
    content.append(Spacer(1, 12))

    report_text = f"""
    <b>Date:</b> {datetime.now().strftime('%d-%m-%Y %H:%M')}<br/><br/>
    <b>Prediction:</b> {prediction}<br/><br/>
    <b>Risk Level:</b> {risk_level}<br/><br/>
    <b>Malignant Risk:</b> {malignant_risk:.2f}%<br/><br/>
    <b>Model Accuracy:</b> {accuracy*100:.2f}%<br/><br/>
    <b>Precision:</b> {precision*100:.2f}%<br/><br/>
    <b>Recall:</b> {recall*100:.2f}%<br/><br/>
    <b>F1 Score:</b> {f1*100:.2f}%<br/><br/>
    """

    content.append(
        Paragraph(
            report_text,
            styles['BodyText']
        )
    )

    doc.build(content)

    pdf = buffer.getvalue()

    buffer.close()

    return pdf

if menu == "Patient Inputs":

    st.header("📝 Patient Clinical Measurements")

    st.write(
        "Enter patient values to assess breast cancer risk."
    )

    patient_data = {}

    cols = st.columns(2)

    for i, feature in enumerate(feature_names):

        with cols[i % 2]:

            patient_data[feature] = st.number_input(
                feature,
                value=float(X[feature].mean())
            )

    if st.button(
        "🔍 Predict Cancer Risk",
        use_container_width=True
    ):

        patient_df = pd.DataFrame(
            [patient_data]
        )

        patient_scaled = scaler.transform(
            patient_df
        )

        probability = model.predict_proba(
    patient_scaled
)[0][1]

        malignant_risk = probability * 100
        benign_probability = (1 - probability) * 100

        prediction = (
            "Malignant"
            if probability >= 0.5
            else "Benign"
        )

        st.session_state["patient_df"] = patient_df
        st.session_state["probability"] = probability
        st.session_state["prediction"] = prediction
        st.session_state["malignant_risk"] = malignant_risk
        st.session_state["benign_probability"] = benign_probability

        st.success(
            "Prediction Generated Successfully!"
        )

# RISK DASHBOARD

elif menu == "Risk Dashboard":

    st.header("📊 Cancer Risk Dashboard")

    if "prediction" not in st.session_state:

        st.warning(
            "Please generate a prediction first from Patient Inputs."
        )

    else:

        prediction = st.session_state["prediction"]
        malignant_risk = st.session_state["malignant_risk"]
        benign_probability = st.session_state["benign_probability"]
        probability = st.session_state["probability"]
        patient_df = st.session_state["patient_df"]

        # TOP METRICS

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Prediction",
                prediction
            )

        with col2:
            st.metric(
                "Malignant Risk %",
                f"{malignant_risk:.2f}%"
            )

        with col3:
            st.metric(
                "Benign Probability %",
                f"{benign_probability:.2f}%"
            )

        st.markdown("---")

        # RISK METER

        st.subheader("🚦 Risk Meter")

        st.progress(float(malignant_risk / 100))

        if malignant_risk < 30:

            st.success(
                f"🟢 LOW RISK ({malignant_risk:.2f}%)"
            )

            risk_level = "LOW"

        elif malignant_risk < 70:

            st.warning(
                f"🟡 MODERATE RISK ({malignant_risk:.2f}%)"
            )

            risk_level = "MODERATE"

        else:

            st.error(
                f"🔴 HIGH RISK ({malignant_risk:.2f}%)"
            )

            risk_level = "HIGH"

        # PATIENT REPORT CARD

        st.markdown("---")

        st.subheader("📋 Patient Health Report Card")

        report_col1, report_col2 = st.columns(2)

        with report_col1:

            st.info(
                f"""
Patient ID: BC-{np.random.randint(1000,9999)}

Prediction: {prediction}

Risk Level: {risk_level}

Date: {datetime.now().strftime('%d-%m-%Y')}
"""
            )

        with report_col2:

            st.info(
                f"""
Model Accuracy: {accuracy*100:.2f}%

Precision: {precision*100:.2f}%

Recall: {recall*100:.2f}%

F1 Score: {f1*100:.2f}%
"""
            )

        # AI HEALTH SUMMARY

        st.markdown("---")

        st.subheader("🤖 AI Health Summary")

        summary = f"""
The Random Forest model predicts **{prediction}**
with a confidence score of **{max(malignant_risk, benign_probability):.2f}%**.

Current cancer risk level is **{risk_level}**.

This assessment should be considered
as a screening support tool and not
a replacement for professional medical diagnosis.
"""

        st.success(summary)

        # COMPARE WITH AVERAGE PATIENT

        st.markdown("---")

        st.subheader("📉 Compare With Average Patient")

        comparison_features = feature_names[:10]

        comparison_df = pd.DataFrame({
            "Feature": comparison_features,
            "Patient Value": [
                patient_df[col].values[0]
                for col in comparison_features
            ],
            "Dataset Average": [
                X[col].mean()
                for col in comparison_features
            ]
        })

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        # FEATURE IMPORTANCE

        st.markdown("---")

        st.subheader("📈 Feature Importance")

        importance_scores = pd.Series(
             model.feature_importances_,
            index=feature_names
        )

        top_features = (
            importance_scores
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(
            figsize=(10,5)
        )

        sns.barplot(
            x=top_features.values,
            y=top_features.index,
            ax=ax
        )

        ax.set_title(
            "Top Influential Features"
        )

        st.pyplot(fig)

# ANALYTICS PAGE

elif menu == "Analytics":

    st.header("📈 Model Analytics")

    # METRICS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision*100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall*100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{f1*100:.2f}%"
    )

    st.markdown("---")

    # CONFUSION MATRIX

    st.subheader("📦 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        pred_class
    )

    fig, ax = plt.subplots(
        figsize=(6,5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # ROC CURVE

    st.markdown("---")

    st.subheader("📊 ROC Curve")

    fpr, tpr, thresholds = roc_curve(
        y_test,
        pred_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    fig, ax = plt.subplots(
        figsize=(7,5)
    )

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.3f}"
    )

    ax.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.legend()

    ax.set_title(
        "Receiver Operating Characteristic"
    )

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("🌳 Model Information")

    st.info(f"""
    Model Type: Random Forest

    Trees: 500

    Features: {len(feature_names)}

    Training Samples: {len(X_train)}

    Testing Samples: {len(X_test)}
    """)


# REPORT PAGE

elif menu == "Generate Report":
    st.header("📄 Medical Report Generator")

    if "prediction" not in st.session_state:

        st.warning(
            "Generate prediction first."
        )

    else:

        prediction = st.session_state["prediction"]
        malignant_risk = st.session_state["malignant_risk"]

        if malignant_risk < 30:
            risk_level = "LOW"

        elif malignant_risk < 70:
            risk_level = "MODERATE"

        else:
            risk_level = "HIGH"

        st.subheader("🩺 Early Detection Recommendation")

        # RECOMMENDATION SYSTEM

        if risk_level == "LOW":

            st.success("""
✅ Low Risk Detected

• Continue regular screening

• Maintain healthy lifestyle

• Schedule routine checkups

• Monitor changes periodically
""")

        elif risk_level == "MODERATE":

            st.warning("""
⚠ Moderate Risk Detected

• Follow-up screening advised

• Consult healthcare provider

• Additional tests may be needed

• Monitor symptoms carefully
""")

        else:

            st.error("""
🚨 High Risk Detected

• Consult Oncologist Immediately

• Mammography Recommended

• Biopsy may be required

• Seek professional diagnosis urgently
""")

        # REPORT SUMMARY

        st.markdown("---")

        st.subheader("📋 Report Summary")

        st.info(
            f"""
Prediction: {prediction}

Risk Level: {risk_level}

Cancer Risk: {malignant_risk:.2f}%

Accuracy: {accuracy*100:.2f}%
"""
        )

        # PDF DOWNLOAD

        pdf = generate_pdf_report(
            prediction,
            risk_level,
            malignant_risk,
            accuracy,
            precision,
            recall,
            f1
        )

        st.download_button(
            label="📥 Download Medical PDF Report",
            data=pdf,
            file_name="Breast_Cancer_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )