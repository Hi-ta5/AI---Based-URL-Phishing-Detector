# utils.py
import re
from urllib.parse import urlparse

def extract_url_features(url: str):
    """
    Extracts numeric and textual features from a URL string.
    Gracefully handles invalid or malformed URLs.
    """
    if not isinstance(url, str):
        url = str(url)
    url = url.strip()

    # If URL is empty or missing, return zeros
    if not url:
        return {  # same feature keys for all rows
            "url_length": 0,
            "has_https": 0,
            "num_dots": 0,
            "num_hyphen": 0,
            "num_at": 0,
            "num_qmark": 0,
            "num_equal": 0,
            "num_slash": 0,
            "num_digits": 0,
            "num_tokens": 0,
            "contains_ip": 0,
            "suspicious_token_count": 0,
        }

    # Try safe parsing
    try:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
    except Exception:
        # fallback if urlparse fails (invalid IPv6, etc.)
        parsed = urlparse("http://" + "invalid.com")

    netloc = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()
    scheme = parsed.scheme.lower()

    features = {}
    features["url_length"] = len(url)
    features["has_https"] = 1 if scheme == "https" else 0
    features["num_dots"] = netloc.count(".")
    features["num_hyphen"] = url.count("-")
    features["num_at"] = url.count("@")
    features["num_qmark"] = url.count("?")
    features["num_equal"] = url.count("=")
    features["num_slash"] = url.count("/")
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["num_tokens"] = len(re.split(r"[./?=&_-]", url))
    features["contains_ip"] = 1 if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", netloc) else 0

    suspicious = ["login", "verify", "update", "secure", "bank", "account", "confirm"]
    features["suspicious_token_count"] = sum(tok in url.lower() for tok in suspicious)

    return features
