import pickle
import re
import streamlit as st

# --- Load pickled models ---
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("voting_classifier.pkl", "rb") as f:
    voting_clf = pickle.load(f)

with open("label_encoder .pkl", "rb") as f:
    label_encoder = pickle.load(f)

# --- Text cleaning ---``
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Prediction function ---
def predict_category(short_desc, long_desc):
    combined_desc = str(short_desc) + " " + str(long_desc)
    cleaned_desc = clean_text(combined_desc)
    desc_vectorized = vectorizer.transform([cleaned_desc])
    predicted_label = voting_clf.predict(desc_vectorized)[0]
    predicted_category = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_category

# --- Streamlit App ---
st.set_page_config(page_title="Bug Category Predictor", layout="centered")
st.title("🐞 Bug Category Prediction App")
st.write("Enter the short and long descriptions of the bug to predict its category.")

# Input fields
short_desc = st.text_input("✏️ Short Description")
long_desc = st.text_area("📝 Long Description")

# Prediction
if st.button("🔍 Predict Category"):
    if short_desc.strip() == "" or long_desc.strip() == "":
        st.warning("Please fill in both the short and long descriptions.")
    else:
        category = predict_category(short_desc, long_desc)
        st.success(f"📌 Predicted Category: **{category}**")
