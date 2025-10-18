# train_model.py
"""
Train an XGBoost model on a phishing dataset (phishing_urls.csv).
If the dataset is missing or incomplete, it creates a synthetic one for testing.

Model: XGBoost + TF-IDF + handcrafted URL features
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import joblib
import xgboost as xgb
from utils import extract_url_features
from scipy import sparse
import json

# ==================== PATHS ====================

DATA_PATH = os.path.join("data", "phishing_urls.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ==================== LOAD DATA ====================

def load_dataset(path=DATA_PATH):
    """Load the CSV dataset and normalize column names."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    # Normalize column names (strip spaces, lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    # Try to identify URL and Label columns automatically
    url_col = None
    label_col = None

    for c in df.columns:
        if "url" in c:
            url_col = c
        if "label" in c or "status" in c or "target" in c:
            label_col = c

    if url_col is None:
        raise ValueError("No URL column found. Ensure your CSV has 'url' or similar column name.")
    if label_col is None:
        raise ValueError("No label column found. Ensure your CSV has 'label', 'status', or 'target' column.")

    df = df[[url_col, label_col]].rename(columns={url_col: "url", label_col: "label"})

    # Convert labels to numeric 0/1 if needed
    # Convert all labels to lowercase strings for uniformity
    df["label"] = df["label"].astype(str).str.lower().str.strip()

# Replace common text labels with numeric 0 or 1
    df["label"] = df["label"].replace({
    "benign": 0, "legit": 0, "legitimate": 0, "safe": 0, "good": 0,
    "malicious": 1, "phishing": 1, "phish": 1, "bad": 1, "yes": 1, "no": 0
})

# Now safely convert to int — anything else becomes 0 by default
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)


    print(f"Loaded dataset with {len(df)} rows and columns: {df.columns.tolist()}")
    print(df.head())
    return df

# ==================== FEATURE EXTRACTION ====================

def featurize(df):
    """Convert URLs into numeric + text features."""
    url_feats = []
    for u in tqdm(df["url"].astype(str), desc="Extracting URL features"):
        url_feats.append(extract_url_features(u))
    url_feat_df = pd.DataFrame(url_feats).fillna(0)

    # TF-IDF of URL text
    urls = df["url"].astype(str).tolist()
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), max_features=2000)
    X_tfidf = tfidf.fit_transform(urls)

    X = sparse.hstack([X_tfidf, sparse.csr_matrix(url_feat_df.values)]).tocsr()
    return X, tfidf, url_feat_df.columns.tolist()

# ==================== MODEL TRAINING ====================

def train_model():
    df = load_dataset(DATA_PATH)
    X, tfidf, feat_names = featurize(df)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    # Save model + vectorizer
    joblib.dump(model, os.path.join(MODELS_DIR, "xgb_model.joblib"))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "vectorizer.joblib"))

    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump({"url_feat_names": feat_names}, f)

    print("\n✅ Model and vectorizer saved in 'models/' folder.")

# ==================== MAIN ====================

if __name__ == "__main__":
    train_model()
