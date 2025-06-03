import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and label encoder
with open("disease_model.pkl", "rb") as f:
    model, all_symptoms = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load and clean datasets
desc_df = pd.read_csv("symptom_Description.csv")
desc_df.columns = desc_df.columns.str.strip().str.lower()

precaution_df = pd.read_csv("symptom_precaution.csv")
precaution_df.columns = precaution_df.columns.str.strip().str.lower()

severity_df = pd.read_csv("Symptom-severity.csv")
severity_df.columns = severity_df.columns.str.strip()

# --- UI ---
st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("🩺 Disease Prediction App")
st.markdown("Select symptoms below to predict your disease and get essential health information.")

# --- Sidebar: Select symptoms ---
selected_symptoms = st.multiselect(
    "Select your symptoms", options=all_symptoms, help="Choose symptoms you are experiencing"
)

# --- Predict Button ---
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Create input vector
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]

        # Predict
        prediction = model.predict([input_vector])[0]
        disease = le.inverse_transform([prediction])[0]

        st.success(f"🧠 **Predicted Disease:** {disease}")

        # --- Disease Description ---
        st.subheader("📘 Disease Description")
        disease_desc = desc_df[desc_df['disease'].str.lower() == disease.lower()]
        if not disease_desc.empty:
            st.markdown(f"**{disease}**: {disease_desc.iloc[0]['description']}")
        else:
            st.markdown("No description available for this disease.")

        # --- Precautions ---
        st.subheader("🛡️ Recommended Precautions")
        precaution_row = precaution_df[precaution_df['disease'].str.lower() == disease.lower()]
        if not precaution_row.empty:
            for p in precaution_row.iloc[0, 1:]:
                if pd.notna(p):
                    st.markdown(f"- {p}")
        else:
            st.info("No precautions found for this disease.")

        # --- Symptom Severity ---
        st.subheader("⚠️ Symptom Severity")
        severity_info = severity_df[severity_df['Symptom'].isin(selected_symptoms)]
        if not severity_info.empty:
            st.dataframe(severity_info.set_index("Symptom"))
        else:
            st.markdown("No severity info found for selected symptoms.")
