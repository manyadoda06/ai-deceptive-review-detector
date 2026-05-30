import streamlit as st
import pickle

tfidf = pickle.load(open('tfidf.pkl', 'rb'))
text_model = pickle.load(open('text_model.pkl', 'rb'))

st.title("🕵️‍♂️ AI-Based Deceptive Review Detection System")
st.caption("Built by Manya Doda")

text = st.text_area("Enter the statement")

if st.button("Detect"):
    vec = tfidf.transform([text])
    pred = text_model.predict(vec)[0]
    probs = text_model.predict_proba(vec)[0]
    classes = text_model.classes_
    confidence = probs[list(classes).index(pred)]

    st.write("🔍 Result:", "Fake Review" if pred == 1 else "Genuine Review")
    st.write("Confidence:", round(confidence * 100, 2), "%")

    st.caption("⚠️ This model detects deceptive language patterns, not factual truth.")

# 👇 ADD THIS PART AT THE VERY END
st.markdown("""
### 🔎 How it works
- Text is converted into numerical features using **TF-IDF**
- A **Logistic Regression** model classifies reviews as deceptive or genuine
- The prediction is based on **writing style patterns**, not facts
""")
