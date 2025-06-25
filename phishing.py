# Import necessary libraries
import os
import pickle
import json
import urllib.parse
import pandas as pd
import numpy as np
import tldextract
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk
from urllib.parse import urlparse
import email
from email import policy
from email.parser import BytesParser
import kagglehub

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Assigns all stop words to a set for easy lookup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define paths
MODEL_PATH = 'models'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


# Custom transformer for extracting phishing features
class PhishingFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.phishing_indicators = {
            'urgent_words': ['urgent', 'immediately', 'alert', 'verify', 'suspended', 'account', 
                           'security', 'verify', 'authenticate', 'login', 'click', 'confirm'],
            'sensitive_requests': ['password', 'credit card', 'ssn', 'social security', 'bank account', 
                                 'pin', 'credentials', 'login', 'username'],
            'suspicious_domains': ['secure', 'login', 'signin', 'verify', 'update', 'support'],
            'action_verbs': ['click', 'download', 'sign in', 'verify', 'update', 'confirm', 'validate']
        }
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = np.zeros((len(X), 13))
        
        for i, text in enumerate(X):
            if not isinstance(text, str):
                continue
                
            # Number of URLs
            urls = re.findall(r'https?://\S+', text)
            features[i, 0] = len(urls)
            
            # URL characteristics
            if urls:
                # Check for URL shortening services
                shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'is.gd', 't.co', 'ow.ly']
                features[i, 1] = any(shortener in url.lower() for url in urls for shortener in shorteners)
                
                # Check for misleading domains
                for url in urls:
                    try:
                        parsed = urlparse(url)
                        domain = parsed.netloc
                        extracted = tldextract.extract(domain)
                        
                        # Check if domain tries to mimic well-known domains
                        known_domains = ['paypal', 'amazon', 'microsoft', 'apple', 'google', 'facebook']
                        for known in known_domains:
                            if known in extracted.domain and known != extracted.domain:
                                features[i, 2] = 1
                                break
                    except:
                        pass
            
            # Count urgent words
            features[i, 3] = sum(word in text.lower() for word in self.phishing_indicators['urgent_words'])
            
            # Count sensitive information requests
            features[i, 4] = sum(phrase in text.lower() for phrase in self.phishing_indicators['sensitive_requests'])
            
            # Count action verbs
            features[i, 5] = sum(verb in text.lower() for verb in self.phishing_indicators['action_verbs'])
            
            # Check for suspicious formatting
            features[i, 6] = 1 if re.search(r'<!DOCTYPE html>|<html>|<body>', text) else 0  # HTML in email
            features[i, 7] = text.count('!')  # Exclamation marks
            features[i, 8] = 1 if re.search(r'[A-Z]{5,}', text) else 0  # All caps text
            
            # Check for spoofing indicators
            features[i, 9] = 1 if re.search(r'From:.*@.*\.[a-z]{2,}.*\n.*Reply-To:.*@.*\.[a-z]{2,}', text, re.IGNORECASE) and \
                           not re.search(r'From:.*@(.*\.[a-z]{2,}).*\n.*Reply-To:.*@\1', text, re.IGNORECASE) else 0
            
            # Check for attachment mentions without actual attachments
            features[i, 10] = 1 if re.search(r'attach(ed|ment)', text, re.IGNORECASE) and not re.search(r'Content-Disposition: attachment', text, re.IGNORECASE) else 0
            
            # Check for misspellings (simplified)
            misspelled_companies = ['paypa1', 'amaz0n', 'g00gle', 'micr0s0ft', 'appleid', 'faceb00k']
            features[i, 11] = any(company in text.lower() for company in misspelled_companies)
            
            # Check for suspicious email senders
            features[i, 12] = 1 if re.search(r'@(mail\.com|outlook\.com|gmail\.com|yahoo\.com)', text, re.IGNORECASE) and \
                            re.search(r'bank|paypal|amazon|ebay|microsoft|apple|google|facebook|secure|update|verify', text, re.IGNORECASE) else 0
        
        return features

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase text
    text = text.lower()
    
    # Remove URLs (save them for other features)
    text = re.sub(r'https?://\S+', ' URL ', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove special characters but keep certain meaningful punctuation
    text = re.sub(r'[^\w\s.!?@]', ' ', text)
    
    # Replace numbers with a token
    text = re.sub(r'\d+', ' NUM ', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    
    return ' '.join(tokens)

# Extract email headers and body for header analysis
def extract_email_parts(raw_email):
    try:
        # Try to parse as a full email with headers
        parser = BytesParser(policy=policy.default)
        if isinstance(raw_email, str):
            # Convert string to bytes if necessary
            raw_email_bytes = raw_email.encode()
        else:
            raw_email_bytes = raw_email
            
        parsed_email = parser.parsebytes(raw_email_bytes)
        
        headers = {}
        for key in parsed_email.keys():
            headers[key.lower()] = parsed_email[key]
        
        # Get the body
        if parsed_email.is_multipart():
            body = ""
            for part in parsed_email.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            body = parsed_email.get_payload(decode=True).decode('utf-8', errors='ignore')
        
        return headers, body
    except Exception as e:
        # If parsing fails, assume it's just the body without headers
        return {}, raw_email

# Email header analysis
def analyze_headers(headers):
    features = {}
    
    # Check for mismatched from/reply-to domains
    if 'from' in headers and 'reply-to' in headers:
        from_domain = re.search(r'@([^>]+)', headers['from'])
        reply_domain = re.search(r'@([^>]+)', headers['reply-to'])
        
        if from_domain and reply_domain and from_domain.group(1) != reply_domain.group(1):
            features['mismatched_reply'] = 1
        else:
            features['mismatched_reply'] = 0
    else:
        features['mismatched_reply'] = 0
    
    # Check for suspicious received headers
    if 'received' in headers:
        received = headers['received']
        suspicious_servers = ['mail.ru', 'yandex.ru', '126.com', 'qq.com']
        features['suspicious_server'] = any(server in received for server in suspicious_servers)
    else:
        features['suspicious_server'] = 0
    
    # Check for authentication results
    if 'authentication-results' in headers:
        auth = headers['authentication-results'].lower()
        features['spf_fail'] = 1 if 'spf=fail' in auth else 0
        features['dkim_fail'] = 1 if 'dkim=fail' in auth else 0
        features['dmarc_fail'] = 1 if 'dmarc=fail' in auth else 0
    else:
        features['spf_fail'] = 0
        features['dkim_fail'] = 0
        features['dmarc_fail'] = 0
    
    return features

# Load real phishing dataset
def load_phishing_dataset():
    try:
        print("Downloading phishing dataset...")
        # Try to download a more phishing-focused dataset
        try:
            path = kagglehub.dataset_download("shantanudhakadd/email-spam-detection-dataset-classification")
            dataset_path = f"{path}/spam.csv"
            
            # Load base dataset
            data = pd.read_csv(dataset_path, encoding='ISO-8859-1')
            data = data.rename(columns={data.columns[0]: 'label', data.columns[1]: 'text'})
            print(f"Loaded {len(data)} emails from dataset")
        except Exception as e:
            print(f"Could not load main dataset: {e}")
            # Create a fallback minimal dataset if main dataset fails
            fallback_data = {
                'text': [
                    "Hello, how are you?",
                    "URGENT: Your PayPal account has been suspended. Click here to verify: http://bit.ly/paypal-verify",
                    "Meeting at 2pm tomorrow",
                    "Dear customer, your Amazon account needs verification. Please login at http://amaz0n-secure.com",
                    "Your order has been shipped and will arrive tomorrow",
                    "Congratulations! You've won a free iPhone. Click here to claim: http://free-iphone-winner.com",
                    "The meeting has been moved to 3pm instead of 2pm",
                    "ALERT: Your Microsoft account has been compromised. Reset your password at http://microsoft-secure.net"
                ],
                'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam']
            }
            data = pd.DataFrame(fallback_data)
            print("Using fallback dataset with 8 examples")
        
        # Try to augment with phishing examples if available
        try:
            phishing_path = kagglehub.dataset_download("debanjanpaul/email-phishing-data")
            phishing_data = pd.read_csv(f"{phishing_path}/phishing.csv")
            if 'email' in phishing_data.columns and 'label' in phishing_data.columns:
                phishing_data = phishing_data.rename(columns={'email': 'text'})
                # Standardize labels
                phishing_data['label'] = phishing_data['label'].map({1: 'spam', 0: 'ham'})
                # Combine datasets
                data = pd.concat([data, phishing_data[['text', 'label']]])
                print(f"Augmented with {len(phishing_data)} additional phishing examples")
        except Exception as e:
            print(f"Could not load additional phishing data: {e}")
            print("Continuing with base dataset only")
            
        return data
    except Exception as e:
        print(f"Error in dataset loading process: {e}")
        # Create a minimal fallback dataset as last resort
        fallback_data = {
            'text': [
                "Hello, how are you?",
                "URGENT: Your PayPal account has been suspended. Click here to verify: http://bit.ly/paypal-verify",
                "Meeting at 2pm tomorrow",
                "Dear customer, your Amazon account needs verification. Please login at http://amaz0n-secure.com"
            ],
            'label': ['ham', 'spam', 'ham', 'spam']
        }
        return pd.DataFrame(fallback_data)

# Main function for model training
def train_phishing_model(data=None, save_model=True):
    print("Starting phishing model training...")
    
    # Load data if not provided
    if data is None:
        data = load_phishing_dataset()
    
    # Data preprocessing
    data['processed_text'] = data['text'].apply(preprocess_text)
    data['label_num'] = data['label'].apply(lambda x: 1 if x.lower() == 'spam' else 0)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data['processed_text'], data['label_num'], test_size=0.2, random_state=42
    )
    
    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
    
    # Create feature pipeline
    text_features = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)))
    ])
    
    phishing_features = Pipeline([
        ('extractor', PhishingFeatureExtractor())
    ])
    
    # Combine all features
    features_union = FeatureUnion([
        ('text_features', text_features),
        ('phishing_features', phishing_features)
    ])
    
    # Create model pipeline
    pipeline = Pipeline([
        ('features', features_union),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # Save detailed classification report
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'], output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save model if requested
    if save_model:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = os.path.join(MODEL_PATH, f'phishing_model_{timestamp}.joblib')
        report_filename = os.path.join(MODEL_PATH, f'model_report_{timestamp}.json')
        
        # Save model
        joblib.dump(pipeline, model_filename)
        print(f"Model saved to {model_filename}")
        
        # Save report
        with open(report_filename, 'w') as f:
            json.dump({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'report': report,
                'confusion_matrix': cm.tolist(),
                'timestamp': timestamp,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }, f, indent=4)
        
        print(f"Model report saved to {report_filename}")
        
        # Save latest model reference
        with open(os.path.join(MODEL_PATH, 'latest_model.txt'), 'w') as f:
            f.write(model_filename)
    
    return pipeline, (accuracy, precision, recall, f1)

# Function to load the latest model
def load_latest_model():
    try:
        with open(os.path.join(MODEL_PATH, 'latest_model.txt'), 'r') as f:
            model_path = f.read().strip()
        
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to analyze a single email
def analyze_email(email_text, model=None):
    if model is None:
        model = load_latest_model()
        if model is None:
            # Train a new model if none exists
            model, _ = train_phishing_model()
    
    # Extract headers and body if possible
    headers, body = extract_email_parts(email_text)
    
    # Preprocess text
    processed_text = preprocess_text(body if body else email_text)
    
    # Make prediction
    prediction = model.predict([processed_text])[0]
    probability = model.predict_proba([processed_text])[0][1]  # Probability of being phishing
    
    # Extract features for explanation
    feature_extractor = PhishingFeatureExtractor()
    phishing_features = feature_extractor.transform([email_text])
    
    feature_names = [
        'URLs count', 'URL shorteners', 'Misleading domains', 'Urgent language', 
        'Sensitive requests', 'Action verbs', 'HTML content', 'Exclamation marks',
        'ALL CAPS text', 'From/Reply-To mismatch', 'Attachment mention', 'Misspelled companies',
        'Suspicious sender'
    ]
    
    # Create explanation
    explanation = {
        'phishing_probability': probability,
        'is_phishing': bool(prediction),
        'features': {feature_names[i]: phishing_features[0][i] for i in range(len(feature_names))}
    }
    
    # Add header analysis if available
    if headers:
        header_features = analyze_headers(headers)
        explanation['header_analysis'] = header_features
    
    return explanation

# Run the model training if this file is executed directly
if __name__ == '__main__':
    data = load_phishing_dataset()
    model, metrics = train_phishing_model(data)
