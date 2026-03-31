import os
import re
import pickle
import numpy as np

# Spam keywords for highlighting
SPAM_KEYWORDS = [
    'free', 'winner', 'won', 'prize', 'claim', 'urgent', 'congratulations',
    'click here', 'limited time', 'act now', 'offer', 'discount', 'cash',
    'earn money', 'make money', 'work from home', 'risk free', 'guarantee',
    'credit card', 'bank account', 'password', 'verify', 'account suspended',
    'you have been selected', 'dear friend', 'nigerian', 'inheritance',
    'lottery', 'jackpot', 'million dollars', 'billion', 'investment',
    'double your money', 'no cost', '100% free', 'click below', 'unsubscribe',
    'bulk email', 'cheap', 'low price', 'save big', 'exclusive deal',
    'pre-approved', 'no obligation', 'cancel anytime', 'meet singles',
    'hot singles', 'weight loss', 'lose weight', 'incredible deal',
    'special promotion', 'gift card', 'amazon gift', 'paypal', 'bitcoin',
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml_model', 'spam_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'ml_model', 'vectorizer.pkl')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\$[\d,]+', ' money ', text)
    text = re.sub(r'[\d]+%', ' percent ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    return None, None


def rule_based_score(text):
    """Fallback rule-based spam scoring when model is not loaded."""
    text_lower = text.lower()
    score = 0
    total_keywords = len(SPAM_KEYWORDS)

    for keyword in SPAM_KEYWORDS:
        if keyword in text_lower:
            score += 1

    # Additional heuristics
    if len(re.findall(r'!', text)) > 3:
        score += 2
    if len(re.findall(r'\$', text)) > 1:
        score += 2
    if re.search(r'\b(free|FREE)\b', text):
        score += 3
    if re.search(r'urgent|URGENT|immediately|IMMEDIATELY', text):
        score += 2
    if re.search(r'click here|CLICK HERE', text_lower):
        score += 3

    # Normalize score
    normalized = min(score / 10, 1.0)
    return normalized


def predict_email(email_text):
    model, vectorizer = load_model()
    preprocessed = preprocess_text(email_text)

    if model and vectorizer:
        try:
            X = vectorizer.transform([preprocessed])
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]

            if prediction == 1:
                result = 'spam'
                confidence = float(proba[1]) * 100
            else:
                result = 'safe'
                confidence = float(proba[0]) * 100

            return result, round(confidence, 2)
        except Exception:
            pass

    # Fallback to rule-based
    spam_score = rule_based_score(email_text)
    if spam_score >= 0.3:
        return 'spam', round(spam_score * 100, 2)
    else:
        return 'safe', round((1 - spam_score) * 100, 2)


def highlight_suspicious_words(email_text):
    """Return HTML with suspicious keywords highlighted."""
    result = email_text
    for keyword in sorted(SPAM_KEYWORDS, key=len, reverse=True):
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        result = pattern.sub(
            lambda m: f'<mark class="spam-highlight">{m.group()}</mark>',
            result
        )
    return result
