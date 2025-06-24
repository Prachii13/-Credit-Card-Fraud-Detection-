import joblib
import pandas as pd

model = joblib.load("model.pkl")

sample = pd.read_csv("data/creditcard.csv").sample(1).drop("Class", axis=1)
prediction = model.predict(sample)[0]

print("💳 Prediction:", "Fraud ❌" if prediction else "Legit ✅")
