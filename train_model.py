"""
Email Spam Detection - ML Model Training Script
================================================
Dataset: SMS Spam Collection (UCI Repository)
Download: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
Or use the built-in sample data below.

Usage:
    python train_model.py

Output:
    detector/ml_model/spam_model.pkl
    detector/ml_model/vectorizer.pkl
"""

import os
import re
import pickle
import numpy as np

# Try importing ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.pipeline import Pipeline
    print("✅ scikit-learn loaded successfully.")
except ImportError:
    print("❌ scikit-learn not found. Run: pip install scikit-learn")
    exit(1)

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))
    USE_NLTK = True
    print("✅ NLTK loaded successfully.")
except ImportError:
    STOP_WORDS = {'the', 'a', 'an', 'is', 'it', 'in', 'on', 'at', 'to',
                  'for', 'of', 'and', 'or', 'but', 'i', 'you', 'we', 'they',
                  'he', 'she', 'was', 'are', 'be', 'been', 'have', 'has', 'do',
                  'this', 'that', 'with', 'from', 'by', 'as', 'not', 'no'}
    USE_NLTK = False
    print("⚠️  NLTK not found. Using basic stopwords. Install with: pip install nltk")


# ─── SAMPLE TRAINING DATA ────────────────────────────────────────────────────
# A small built-in dataset. For production accuracy, use the UCI SMS Spam
# Collection or Enron spam dataset.
SAMPLE_DATA = [
    # SPAM examples (label=1)
    ("Congratulations! You won a $1000 gift card! Click here to claim NOW!", 1),
    ("FREE prize! You have been selected. Call now 0800-FREE to claim your reward!", 1),
    ("Urgent: Your account has been suspended. Verify your details immediately!", 1),
    ("You are a WINNER! Nigerian prince needs your help transferring $10 billion!", 1),
    ("Limited time offer! Buy now and get 90% discount. No credit card needed!", 1),
    ("Make $5000 per week working from home! Guaranteed income, risk free!", 1),
    ("Your PayPal account is compromised. Click to verify bank account now!", 1),
    ("LOTTERY WINNER! Claim your jackpot prize money today! Exclusive offer!", 1),
    ("Hot singles in your area want to meet you! Click here for FREE access!", 1),
    ("Earn money fast! Bitcoin investment doubles your money in 24 hours!", 1),
    ("IMPORTANT: Your Amazon account will be closed unless you verify immediately!", 1),
    ("Cheap Viagra, Cialis pills. No prescription needed. Order online now!", 1),
    ("You've been pre-approved for a $50,000 loan! No credit check required!", 1),
    ("Congratulations dear friend, you have won the international lottery prize!", 1),
    ("FREE iPhone! Complete this survey to claim your prize. Limited time only!", 1),
    ("Your inheritance fund of $4.5 million is waiting. Contact us urgently!", 1),
    ("Lose 30 pounds in 30 days! Miracle weight loss pill. Order now free trial!", 1),
    ("CLAIM NOW: You are our lucky customer. Win cash prize today for free!", 1),
    ("Act immediately! Your visa has issues. Provide SSN to resolve matter now!", 1),
    ("Make easy money! Our proven system earns $1000 daily. No experience needed!", 1),
    ("Exclusive deal: Click here for cheap medication without doctor prescription!", 1),
    ("You have an unclaimed tax refund. Click to claim your money back now!", 1),
    ("WINNER WINNER! You have been chosen for our grand prize giveaway! Claim!", 1),
    ("Urgent security alert! Your credit card is compromised. Verify details now!", 1),
    ("Get rich quick scheme guaranteed! Invest $100 and earn $10000 in one week!", 1),
    ("Dear lucky winner, you won our sweepstakes. Send your bank details today!", 1),
    ("FREE gift card offer expires tonight! Click here and claim your reward now!", 1),
    ("Investment opportunity! Earn passive income online. 500% guaranteed returns!", 1),
    ("ALERT: Suspicious login detected. Click link to secure your account now!", 1),
    ("You qualify for government grant money! Apply now, no repayment required!", 1),

    # HAM examples (label=0)
    ("Hi, are we still meeting for lunch tomorrow at noon?", 0),
    ("Please find attached the quarterly report for your review.", 0),
    ("Thanks for your help yesterday. The presentation went really well!", 0),
    ("Can you please send me the updated schedule for next week?", 0),
    ("The team meeting has been rescheduled to 3 PM on Friday.", 0),
    ("I wanted to check in on the project status. How is it going?", 0),
    ("Happy Birthday! Hope you have a wonderful day!", 0),
    ("Please review the attached document and share your feedback.", 0),
    ("The invoice for last month's services is attached. Please process payment.", 0),
    ("Let's grab coffee sometime this week and catch up.", 0),
    ("The conference call tomorrow is at 10 AM EST. Dial-in details attached.", 0),
    ("I'll be out of office from Monday to Wednesday. Responses may be delayed.", 0),
    ("Could you please update me on the status of our order #12345?", 0),
    ("Thank you for your inquiry. We will get back to you within 2 business days.", 0),
    ("The product you ordered has been shipped. Tracking number: US12345678.", 0),
    ("Reminder: Your appointment is scheduled for tomorrow at 2:30 PM.", 0),
    ("Please join us for the company all-hands meeting this Friday at 4 PM.", 0),
    ("I've reviewed your proposal and have a few questions. Can we schedule a call?", 0),
    ("Your subscription has been renewed successfully. No action required.", 0),
    ("Hi team, please welcome our new colleague who is joining us on Monday.", 0),
    ("The deadline for the project deliverables is end of this month.", 0),
    ("I hope you and your family are doing well during these challenging times.", 0),
    ("The parking lot will be closed for maintenance next Tuesday from 8 AM to 5 PM.", 0),
    ("Please ensure all timesheets are submitted by Friday 5 PM.", 0),
    ("Your password was successfully changed. If this wasn't you, contact support.", 0),
    ("The library book you reserved is now available for pickup.", 0),
    ("We'd love to get your feedback on your recent experience with us.", 0),
    ("Your flight confirmation: Flight AA123 departs at 6:45 AM on March 15.", 0),
    ("Meeting notes from today's call are attached for your reference.", 0),
    ("Thanks for reaching out! I'll connect you with the right person.", 0),
]


def preprocess_text(text):
    """Clean and normalize email text."""
    text = text.lower()
    # Replace URLs
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    # Replace dollar amounts
    text = re.sub(r'\$[\d,]+', ' moneydollar ', text)
    # Replace percentages
    text = re.sub(r'[\d]+%', ' percent ', text)
    # Replace phone numbers
    text = re.sub(r'[\d\-\(\)]{7,}', ' phonenumber ', text)
    # Remove non-alphabetic chars
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)


def load_dataset(filepath=None):
    """
    Load dataset from file or use built-in sample data.
    Supports SMS Spam Collection format: label<TAB>text
    """
    texts, labels = [], []

    if filepath and os.path.exists(filepath):
        print(f"📂 Loading dataset from: {filepath}")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        label_str, text = parts
                        label = 1 if label_str.strip().lower() == 'spam' else 0
                        texts.append(text)
                        labels.append(label)
        print(f"✅ Loaded {len(texts)} records from file.")
    else:
        print("📊 Using built-in sample dataset...")
        for text, label in SAMPLE_DATA:
            texts.append(text)
            labels.append(label)
        print(f"✅ Loaded {len(texts)} sample records.")

    return texts, labels


def train_model(dataset_path=None):
    """Train and save the spam detection model."""
    print("\n" + "="*55)
    print("  EMAIL SPAM DETECTION - MODEL TRAINING")
    print("="*55)

    # Load data
    texts, labels = load_dataset(dataset_path)

    # Preprocess
    print("\n🔄 Preprocessing text data...")
    processed = [preprocess_text(t) for t in texts]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"   Training samples : {len(X_train)}")
    print(f"   Testing samples  : {len(X_test)}")

    # Vectorize
    print("\n🔢 Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train models
    results = {}

    print("\n🧠 Training Naive Bayes...")
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_vec, y_train)
    nb_pred = nb.predict(X_test_vec)
    nb_acc = accuracy_score(y_test, nb_pred)
    results['naive_bayes'] = (nb, nb_acc)
    print(f"   Accuracy: {nb_acc * 100:.2f}%")

    print("\n🧠 Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train_vec, y_train)
    lr_pred = lr.predict(X_test_vec)
    lr_acc = accuracy_score(y_test, lr_pred)
    results['logistic_regression'] = (lr, lr_acc)
    print(f"   Accuracy: {lr_acc * 100:.2f}%")

    # Pick best model
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc = results[best_name]
    print(f"\n🏆 Best model: {best_name.replace('_', ' ').title()} ({best_acc * 100:.2f}%)")

    # Detailed report
    best_pred = best_model.predict(X_test_vec)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, best_pred, target_names=['Safe', 'Spam']))

    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), 'detector', 'ml_model')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'spam_model.pkl')
    vec_path = os.path.join(model_dir, 'vectorizer.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Vectorizer saved to: {vec_path}")
    print("\n" + "="*55)
    print("  TRAINING COMPLETE! You can now run the Django app.")
    print("="*55 + "\n")

    return best_model, vectorizer


if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    train_model(dataset_path)
