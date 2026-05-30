import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("deceptive-opinion.csv")

# Map labels: truthful = 0, deceptive = 1
df["label"] = df["deceptive"].map({"truthful": 0, "deceptive": 1})

# Text and labels
X = df["text"]
y = df["label"]

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer
pickle.dump(model, open("text_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf.pkl", "wb"))

print("✅ Model trained successfully!")
