import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📩 Spam Message Detection App")
st.write("Detect whether a message is **Spam or Ham (Genuine)** using Machine Learning.")

# Load model once
model = joblib.load("spam_clf.pkl")

# Layout
col1, col3 = st.columns([3, 3], gap="large")

# ------------------ Single Prediction ------------------
with col1:
    st.subheader("🔍 Single Message Prediction")

    text = st.text_input("Enter your message")

    if st.button("Predict", key="single"):
        if text.strip() == "":
            st.warning("⚠️ Please enter a message first!")
        else:
            result = model.predict([text])[0]

            if result == "spam":
                st.error("🚨 Spam Message Detected!")
            else:
                st.success("✅ Genuine (Ham) Message")

# ------------------ Bulk Prediction ------------------
with col3:
    st.subheader("📂 Bulk Message Prediction")

    file = st.file_uploader("Upload CSV file", type=["csv", "txt"])

    if file is not None:
        df = pd.read_csv(file, header=None, names=["Message"])
        st.dataframe(df, use_container_width=True)

        if st.button("Predict Bulk", key="bulk"):
            df["Prediction"] = model.predict(df["Message"])
            st.success("✅ Prediction Completed!")

            st.dataframe(df, use_container_width=True)

            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Results",
                data=csv,
                file_name="predicted_messages.csv",
                mime="text/csv"
            )

# st = columns, headers, button, error, text, success, file_loader, text_input 
