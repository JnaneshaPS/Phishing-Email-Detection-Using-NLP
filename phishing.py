# Import necessary libraries
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk

# Download and access dataset from online kaggle api
import kagglehub

path = kagglehub.dataset_download("shantanudhakadd/email-spam-detection-dataset-classification")
print("Path to dataset files:", path)

# Set up dataset path
dataset_path = f"{path}/spam.csv" 

# Load dataset
data = pd.read_csv(dataset_path, encoding='ISO-8859-1')


# Inspect dataset
print("Columns in new dataset:", data.columns)
print(data.head())


# Data Preprocessing
nltk.download('stopwords')
nltk.download('punkt')

# Assigns all stop words to a set for easy lookup
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to email body column to create new column 'processed_text'
data['processed_text'] = data['v2'].apply(preprocess_text)  

# Feature Engineering with TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X = tfidf_vectorizer.fit_transform(data['processed_text']).toarray()
y = data['v1'].apply(lambda x: 1 if x == 'spam' else 0)  # Convert 'spam' to 1 and 'ham' to 0

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training with multinomial naive bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display results
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
