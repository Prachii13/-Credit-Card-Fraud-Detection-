import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
df = pd.read_csv("data/creditcard.csv")

st.title("💳 Credit Card Fraud Detector")
st.markdown("Choose a random transaction from the dataset:")

index = st.slider("Transaction index", 0, len(df)-1, 1)
sample = df.drop("Class", axis=1).iloc[[index]]
true_label = df.iloc[index]["Class"]

if st.button("Predict"):
    prediction = model.predict(sample)[0]
    st.success("✅ Legit" if prediction == 0 else "❌ Fraud")
    st.write(f"🔍 Ground Truth: {'Fraud' if true_label else 'Legit'}")
