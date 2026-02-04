import pandas as pd
import re
import nltk
import joblib
import streamlit as st

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# ---------------- SETUP ----------------
nltk.download('stopwords')
stop_words = stopwords.words('english')

st.title("Flipkart Product Review Sentiment Analysis")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data.csv")

# Show columns (for verification)
st.write("Dataset Columns:", df.columns.tolist())

# Auto-detect review & rating columns
review_col = None
rating_col = None

for c in df.columns:
    if "review" in c.lower():
        review_col = c
    if "rating" in c.lower() or "star" in c.lower():
        rating_col = c

if review_col is None or rating_col is None:
    st.error("Could not auto-detect Review or Rating columns")
    st.stop()

df = df[[review_col, rating_col]]
df.dropna(inplace=True)

# ---------------- SENTIMENT LABEL ----------------
df['sentiment'] = df[rating_col].apply(lambda x: 1 if float(x) >= 4 else 0)

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-z ]','',text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['clean_review'] = df[review_col].apply(clean_text)

# ---------------- FEATURES ----------------
X = df['clean_review']
y = df['sentiment']

# ---------------- TF-IDF ----------------
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# ---------------- SPLIT ----------------
X_train,X_test,y_train,y_test = train_test_split(
    X_vec,y,test_size=0.2,random_state=42
)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

# ---------------- EVALUATION ----------------
pred = model.predict(X_test)
f1 = f1_score(y_test,pred)

st.write("Model F1 Score:", round(f1,3))

# ---------------- SAVE MODEL ----------------
joblib.dump(model,"model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

# ================= STREAMLIT UI =================

st.subheader("Test Your Own Review")

user_review = st.text_area("Enter Product Review")

if st.button("Analyze Sentiment"):
    if user_review.strip()=="":
        st.warning("Please enter review text")
    else:
        cleaned = clean_text(user_review)
        vec = vectorizer.transform([cleaned])
        result = model.predict(vec)[0]

        if result==1:
            st.success("Positive Review 😀")
        else:
            st.error("Negative Review 😞")
