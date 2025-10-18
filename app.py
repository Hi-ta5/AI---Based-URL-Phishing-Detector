# app_simple.py
import os
import joblib
import streamlit as st
import pandas as pd
from utils import extract_url_features
from scipy import sparse

st.set_page_config(page_title="PhishScan", page_icon="🛡️", layout="centered")

st.markdown("<h1 style='text-align:center'>PhishScan — URL Checker</h1>", unsafe_allow_html=True)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

@st.cache_resource
def load_model():
    """Try to load saved model & vectorizer. Return (model, vectorizer) or (None, None)."""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_PATH)
            vec = joblib.load(VECTORIZER_PATH)
            return model, vec
    except Exception as e:
        st.warning(f"Failed to load model: {e}")
    return None, None

model, vectorizer = load_model()

def predict_with_model(url):
    """Use loaded model & vectorizer to predict (label, prob)."""
    feats = [extract_url_features(url)]
    feat_df = pd.DataFrame(feats).fillna(0)
    X_tfidf = vectorizer.transform([url])
    X = sparse.hstack([X_tfidf, sparse.csr_matrix(feat_df.values)]).tocsr()
    prob = float(model.predict_proba(X)[:,1][0])
    pred = int(prob >= 0.5)
    return pred, prob

def heuristic_predict(url):
    """
    Simple heuristic fallback:
    - If suspicious tokens present, or contains IP, long URL or many digits -> phishing
    - Otherwise legitimate.
    """
    f = extract_url_features(url)
    score = 0
    # suspicious tokens are strong indicator
    score += f.get("suspicious_token_count", 0) * 3
    # IP in domain
    score += f.get("contains_ip", 0) * 3
    # many digits or hyphens
    if f.get("num_digits", 0) > 5:
        score += 1
    if f.get("num_hyphen", 0) > 3:
        score += 1
    # long urls suspicious
    if f.get("url_length", 0) > 80:
        score += 1
    # lack of https is slightly suspicious
    if f.get("has_https", 0) == 0:
        score += 1

    # Convert score to a mock probability (0..1)
    prob = min(1.0, score / 6.0)
    pred = 1 if prob >= 0.5 else 0
    return pred, prob

# UI input
url_input = st.text_input("Enter URL to check", value="", placeholder="e.g. http://example.com/login")

if st.button("Check"):
    if not url_input.strip():
        st.warning("Please enter a URL.")
    else:
        url = url_input.strip()
        if model is not None and vectorizer is not None:
            try:
                pred, prob = predict_with_model(url)
                used = "Trained model"
            except Exception as e:
                st.error(f"Model prediction failed: {e}\nFalling back to heuristic.")
                pred, prob = heuristic_predict(url)
                used = "Heuristic fallback"
        else:
            pred, prob = heuristic_predict(url)
            used = "Heuristic (no model found)"

        label_text = "Scam / Phishing / Fake" if pred == 1 else "Legitimate / Real"
        st.markdown(f"**Result:** {label_text}")
        st.markdown(f"**Probability of phishing:** {prob:.2%}")
        st.info(f"Prediction source: {used}")

        # Optional: show a tiny explanation
        if pred == 1:
            st.write("Likely reasons: suspicious tokens (login/verify/update), IP address in domain, long URL, many digits, or missing HTTPS.")
        else:
            st.write("No strong phishing signals found by the model/heuristic.")

# Show note about model
st.markdown("---")
if model is None or vectorizer is None:
    st.warning(
        "No trained model found in 'models/'.\n"
        "To use a trained XGBoost model, place 'xgb_model.joblib' and 'vectorizer.joblib' into the models/ folder.\n"
        "You can train the model by running: `python train_model.py` (in your project folder)."
    )
else:
    st.success("Know the risk, protect your data. Check before you click.")
