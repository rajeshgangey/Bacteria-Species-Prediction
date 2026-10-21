import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

st.set_page_config(page_title="Bacterial Species Demo", page_icon="🦠")
st.title("🦠 Bacterial Species Prediction")
st.caption("Classification from example numeric biological features.")
data = pd.read_csv("data/bacteria_demo.csv")
features = [c for c in data.columns if c.startswith("feature_")]
X, y = data[features], data["species"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
st.warning("The included values are synthetic teaching examples, not measured biological data. Predictions are not scientifically valid.")
with st.form("predict"):
    values = [st.number_input(f"Feature {i+1}", value=float(X.iloc[:, i].mean())) for i in range(len(features))]
    submitted = st.form_submit_button("Predict demo species")
if submitted:
    row = pd.DataFrame([values], columns=features)
    st.success(f"Demo prediction: {model.predict(row)[0]}")
with st.expander("Demo data and validation"):
    st.dataframe(data, use_container_width=True)
    try:
        scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=3)
        st.write("3-fold cross-validation accuracy on tiny synthetic data:", round(float(scores.mean()), 3))
        st.caption("This score only describes the toy dataset and does not measure real-world biological performance.")
    except ValueError as exc:
        st.info(f"Validation unavailable: {exc}")
