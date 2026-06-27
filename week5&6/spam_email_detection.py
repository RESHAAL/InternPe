# ==========================================================
#              SPAM MAIL DETECTION DASHBOARD
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import plotly.express as px
import plotly.graph_objects as go

import re

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Spam Mail Detection",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background:#F8FAFC;
}

.block-container{
    padding-top:2rem;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

.big-font{
    font-size:30px;
    font-weight:bold;
    color:#1E3A8A;
}

.result-card{
    padding:25px;
    border-radius:15px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    "<h1 style='text-align:center;'>📧 Spam Mail Detection Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:gray;'>"
    "Machine Learning Based Email Classification System"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# ==========================================================
# LOAD DATA
# ==========================================================

import os

@st.cache_data
def load_dataset():

    current_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(
        current_dir,
        "mail_data (2) - mail_data (2).csv"
    )

    df = pd.read_csv(csv_path)

    df.fillna("", inplace=True)

    df.loc[df["Category"] == "spam", "Category"] = 0
    df.loc[df["Category"] == "ham", "Category"] = 1

    df["Category"] = df["Category"].astype(int)

    return df

# ==========================================================
# PREPARE DATA
# ==========================================================
mail_data = load_dataset()
X = mail_data["Message"]

Y = mail_data["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================================================
# TRAIN MODELS
# ==========================================================

@st.cache_resource
def train_models():

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Naive Bayes":
            MultinomialNB(),

        "Linear SVM":
            LinearSVC()

    }

    trained_models = {}
    accuracies = {}

    for name, model in models.items():

        model.fit(X_train_vec, y_train)

        pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, pred)

        trained_models[name] = model

        accuracies[name] = round(acc*100,2)

    return trained_models, accuracies

models, accuracies = train_models()

# Default Prediction Model

prediction_model = models["Logistic Regression"]

# ==========================================================
# MODEL COMPARISON TABLE
# ==========================================================

comparison_df = pd.DataFrame({

    "Model": accuracies.keys(),
    "Accuracy (%)": accuracies.values()

})

# ==========================================================
# DATASET STATISTICS
# ==========================================================

total_emails = len(mail_data)

ham_count = len(mail_data[mail_data["Category"]==1])

spam_count = len(mail_data[mail_data["Category"]==0])

spam_percentage = round(
    spam_count/total_emails*100,
    2
)

ham_percentage = round(
    ham_count/total_emails*100,
    2
)

# ==========================================================
# EMAIL STATISTICS FUNCTION
# ==========================================================

def email_statistics(text):

    words = len(text.split())

    characters = len(text)

    uppercase = len(
        re.findall(r"\b[A-Z]{2,}\b", text)
    )

    digits = len(
        re.findall(r"\d", text)
    )

    links = len(
        re.findall(r"http\S+|www\S+", text)
    )

    special = len(
        re.findall(r"[!@#$%^&*(),.?\":{}|<>]", text)
    )

    emails = len(
        re.findall(
            r"\S+@\S+",
            text
        )
    )

    return {

        "Words": words,
        "Characters": characters,
        "Uppercase": uppercase,
        "Numbers": digits,
        "Links": links,
        "Special Characters": special,
        "Email IDs": emails

    }

# ==========================================================
# CONFUSION MATRIX FUNCTION
# ==========================================================

def model_metrics(model):

    pred = model.predict(X_test_vec)

    cm = confusion_matrix(y_test, pred)

    accuracy = accuracy_score(y_test, pred)

    precision = precision_score(y_test, pred)

    recall = recall_score(y_test, pred)

    f1 = f1_score(y_test, pred)

    return cm, accuracy, precision, recall, f1

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def important_words(model, top=15):

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    coef = model.coef_[0]

    spam_words = feature_names[np.argsort(coef)[:top]]

    ham_words = feature_names[np.argsort(coef)[-top:]]

    return spam_words, ham_words

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image("https://cdn-icons-png.flaticon.com/512/561/561127.png", width=80)

    st.title("Spam Detection")

    st.markdown("---")

    st.subheader("Project Details")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Vectorizer:** TF-IDF")
    st.write("**Dataset:** SMS Spam Collection")

    st.markdown("---")

    st.subheader("Navigation")

    page = st.radio(
        "Select Section",
        [
            "Dashboard",
            "Prediction",
            "Model Analysis"
        ]
    )

    st.markdown("---")

    st.success("System Ready")

# ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.header("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "📧 Total Emails",
            total_emails
        )

    with c2:
        st.metric(
            "✅ Ham Emails",
            ham_count
        )

    with c3:
        st.metric(
            "🚨 Spam Emails",
            spam_count
        )

    with c4:
        st.metric(
            "📈 Spam %",
            f"{spam_percentage}%"
        )

    st.divider()

    # ======================================================
    # CHARTS
    # ======================================================

    left, right = st.columns(2)

    with left:

        pie = px.pie(
            values=[ham_count, spam_count],
            names=["Ham", "Spam"],
            title="Dataset Distribution",
            hole=0.45,
            color_discrete_sequence=[
                "#22C55E",
                "#EF4444"
            ]
        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

    with right:

        bar = px.bar(
            x=["Ham", "Spam"],
            y=[ham_count, spam_count],
            color=["Ham", "Spam"],
            text=[ham_count, spam_count],
            title="Email Categories",
            color_discrete_sequence=[
                "#16A34A",
                "#DC2626"
            ]
        )

        bar.update_layout(showlegend=False)

        st.plotly_chart(
            bar,
            use_container_width=True
        )

    st.divider()

    # ======================================================
    # MODEL COMPARISON
    # ======================================================

    st.subheader("🤖 Model Comparison")

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    accuracy_chart = px.bar(

        comparison_df,

        x="Model",

        y="Accuracy (%)",

        color="Model",

        text="Accuracy (%)"

    )

    accuracy_chart.update_layout(

        yaxis_range=[90,100],

        showlegend=False

    )

    st.plotly_chart(

        accuracy_chart,

        use_container_width=True

    )

    st.divider()

    # ======================================================
    # QUICK INSIGHTS
    # ======================================================

    st.subheader("📌 Dataset Insights")

    a, b = st.columns(2)

    with a:

        st.info(f"""
        **Ham Percentage**

        {ham_percentage}%
        """)

    with b:

        st.warning(f"""
        **Spam Percentage**

        {spam_percentage}%
        """)

    st.success(
        "Logistic Regression is selected as the final prediction model."
    )

# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "Prediction":

    st.header("📨 Email Spam Prediction")

    st.write(
        "Enter an email or SMS below to check whether it is **Spam** or **Ham**."
    )

    st.divider()

    # ------------------------------------------------------
    # INPUT BOX
    # ------------------------------------------------------

    email = st.text_area(
        "📧 Enter Email",
        height=220,
        placeholder="Paste your email here..."
)

    st.divider()

    # ------------------------------------------------------
    # EMAIL STATISTICS
    # ------------------------------------------------------

    if email:

        stats = email_statistics(email)

        st.subheader("📊 Email Statistics")

        a, b, c, d = st.columns(4)

        a.metric("Words", stats["Words"])
        b.metric("Characters", stats["Characters"])
        c.metric("Uppercase", stats["Uppercase"])
        d.metric("Numbers", stats["Numbers"])

        e, f, g = st.columns(3)

        e.metric("Links", stats["Links"])
        f.metric("Special Symbols", stats["Special Characters"])
        g.metric("Email IDs", stats["Email IDs"])

    st.divider()

    # ------------------------------------------------------
    # PREDICT BUTTON
    # ------------------------------------------------------

    if st.button("🔍 Predict", use_container_width=True):

        if email.strip() == "":

            st.warning("Please enter an email.")

        else:

            # Vectorize

            vector = vectorizer.transform([email])

            prediction = prediction_model.predict(vector)[0]

            probability = prediction_model.predict_proba(vector)[0]

            ham_probability = probability[1] * 100
            spam_probability = probability[0] * 100

            confidence = max(
                ham_probability,
                spam_probability
            )

            # --------------------------------------------------
            # RESULT
            # --------------------------------------------------

            if prediction == 1:

                st.success("✅ HAM EMAIL")

            else:

                st.error("🚨 SPAM EMAIL")

            st.markdown("### 🎯 Confidence Score")

            st.progress(int(confidence))

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.divider()

            # --------------------------------------------------
            # SPAM RISK
            # --------------------------------------------------

            st.subheader("⚠ Spam Risk Meter")

            risk = spam_probability

            if risk < 20:

                level = "🟢 Very Low"

            elif risk < 40:

                level = "🟢 Low"

            elif risk < 60:

                level = "🟡 Moderate"

            elif risk < 80:

                level = "🟠 High"

            else:

                level = "🔴 Very High"

            left, right = st.columns([3,1])

            with left:
                st.progress(int(risk))

            with right:
                st.metric(
                    "Risk",
                    level
                )

            st.divider()

            # --------------------------------------------------
            # PROBABILITIES
            # --------------------------------------------------

            st.subheader("📈 Prediction Probability")

            probability_df = pd.DataFrame({

                "Category": ["Ham", "Spam"],

                "Probability": [

                    ham_probability,

                    spam_probability

                ]

            })

            probability_chart = px.bar(

                probability_df,

                x="Category",

                y="Probability",

                text="Probability",

                color="Category",

                color_discrete_sequence=[
                    "#22C55E",
                    "#EF4444"
                ]

            )

            probability_chart.update_layout(

                yaxis_title="Probability (%)",

                showlegend=False

            )

            st.plotly_chart(

                probability_chart,

                use_container_width=True

            )

            st.divider()

            # --------------------------------------------------
            # RAW SCORES
            # --------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(

                    "✅ Ham Probability",

                    f"{ham_probability:.2f}%"

                )

            with col2:

                st.metric(

                    "🚨 Spam Probability",

                    f"{spam_probability:.2f}%"

                )

            # Save for Part 3B

            st.session_state.last_prediction = prediction
            st.session_state.last_email = email
            st.session_state.ham_probability = ham_probability
            st.session_state.spam_probability = spam_probability
            st.session_state.confidence = confidence

            # =====================================================
            # EXPLAINABLE AI
            # =====================================================

            st.divider()

            st.subheader("🧠 Why did the model predict this?")

            feature_names = np.array(
                vectorizer.get_feature_names_out()
            )

            tfidf_vector = vector.toarray()[0]

            non_zero = np.where(tfidf_vector > 0)[0]

            if len(non_zero) > 0:

                feature_scores = []

                coefficients = prediction_model.coef_[0]

                for index in non_zero:

                    score = tfidf_vector[index] * coefficients[index]

                    feature_scores.append(
                        (
                            feature_names[index],
                            score
                        )
                    )

                feature_scores = sorted(
                    feature_scores,
                    key=lambda x: abs(x[1]),
                    reverse=True
                )

                top_features = feature_scores[:10]

                explain_df = pd.DataFrame(

                    top_features,

                    columns=[
                        "Word",
                        "Contribution"
                    ]

                )

                explain_df["Impact"] = explain_df[
                    "Contribution"
                ].apply(
                    lambda x: "Spam"
                    if x < 0
                    else "Ham"
                )

                st.dataframe(
                    explain_df,
                    use_container_width=True
                )

                fig = px.bar(

                    explain_df,

                    x="Contribution",

                    y="Word",

                    color="Impact",

                    orientation="h",

                    title="Top Words Affecting Prediction",

                    color_discrete_map={

                        "Spam": "#EF4444",

                        "Ham": "#22C55E"

                    }

                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info("No significant words found.")

            # =====================================================
            # HIGHLIGHT SUSPICIOUS WORDS
            # =====================================================

            st.divider()

            st.subheader("🚩 Suspicious Words Detected")

            suspicious_words = [

                "free",
                "winner",
                "win",
                "claim",
                "offer",
                "cash",
                "bonus",
                "click",
                "urgent",
                "prize",
                "gift",
                "congratulations",
                "selected",
                "reward",
                "limited",
                "discount"

            ]

            found = []

            email_lower = email.lower()

            for word in suspicious_words:

                if word in email_lower:

                    found.append(word)

            if found:

                cols = st.columns(4)

                for i, word in enumerate(found):

                    cols[i % 4].error(word.upper())

            else:

                st.success(
                    "No common spam keywords detected."
                )

            # =====================================================
            # PREDICTION HISTORY
            # =====================================================

            st.divider()

            st.subheader("📜 Prediction History")

            if "history" not in st.session_state:

                st.session_state.history = []

            result = "HAM" if prediction == 1 else "SPAM"

            st.session_state.history.append({

                "Prediction": result,

                "Confidence": round(
                    confidence,
                    2
                ),

                "Spam Probability": round(
                    spam_probability,
                    2
                )

            })

            history_df = pd.DataFrame(
                st.session_state.history
            )

            st.dataframe(
                history_df,
                use_container_width=True
            )

            # =====================================================
            # DOWNLOAD REPORT
            # =====================================================

            st.divider()

            st.subheader("⬇ Download Prediction Report")

            report = pd.DataFrame({

                "Email":[email],

                "Prediction":[result],

                "Confidence":[confidence],

                "Ham Probability":[ham_probability],

                "Spam Probability":[spam_probability]

            })

            csv = report.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                "📥 Download CSV",

                csv,

                "prediction_report.csv",

                "text/csv"

            )

            # =====================================================
            # FINAL RESULT CARD
            # =====================================================

            st.divider()

            st.subheader("📌 Final Summary")

            if prediction == 1:

                st.success(f"""
Prediction : HAM

Confidence : {confidence:.2f}%

Spam Probability : {spam_probability:.2f}%

Risk Level : {level}
""")

            else:

                st.error(f"""
Prediction : SPAM
Confidence : {confidence:.2f}%
Spam Probability : {spam_probability:.2f}%
Risk Level : {level}
""")

# ==========================================================
# MODEL ANALYSIS
# ==========================================================

elif page == "Model Analysis":

    st.header("📊 Model Analysis")

    st.subheader("Model Performance")

    cm, accuracy, precision, recall, f1 = model_metrics(prediction_model)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{accuracy*100:.2f}%")
    c2.metric("Precision", f"{precision*100:.2f}%")
    c3.metric("Recall", f"{recall*100:.2f}%")
    c4.metric("F1 Score", f"{f1*100:.2f}%")

    st.divider()

    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Ham", "Actual Spam"],
        columns=["Predicted Ham", "Predicted Spam"]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Top Important Words")

    spam_words, ham_words = important_words(prediction_model)

    left, right = st.columns(2)

    with left:

        spam_df = pd.DataFrame({
            "Spam Words": spam_words
        })

        st.write("### 🚨 Spam Indicators")

        st.dataframe(spam_df, use_container_width=True)

    with right:

        ham_df = pd.DataFrame({
            "Ham Words": ham_words
        })

        st.write("### ✅ Ham Indicators")

        st.dataframe(ham_df, use_container_width=True)

    st.divider()

    st.subheader("Model Comparison")

    st.dataframe(comparison_df, use_container_width=True)

    fig = px.bar(
        comparison_df,
        x="Model",
        y="Accuracy (%)",
        color="Model",
        text="Accuracy (%)"
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)