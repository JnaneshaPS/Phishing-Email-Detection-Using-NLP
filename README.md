# Phishing Email Detection using NLP and Machine Learning

This project is designed to detect phishing emails by analyzing the text content using Natural Language Processing (NLP) and Machine Learning (ML). This project will help me understand how to preprocess email text, extract meaningful features, and build a simple classifier to differentiate phishing emails from legitimate ones.

## Challenges

Coming into this project, I faced a few challenges that I needed to address. First, I had limited experience with NLP and text preprocessing, so understanding how to clean and prepare raw email data was initially overwhelming. Additionally, working with a dataset from Kaggle presented its own challenges, including learning how to download and load the dataset properly within my environment. I also needed to familiarize myself with evaluation metrics like precision, recall, and F1-score, as I had mainly worked with accuracy in previous projects. Lastly, understanding how to use TF-IDF for feature extraction was a new concept that took some time to grasp.

## Project Structure

1. **Set Up the Environment**: Install Python and essential libraries.
2. **Gather the Dataset**: Use a dataset containing phishing and non-phishing emails (such as the Enron Email Dataset).
3. **Data Preprocessing**: Clean and preprocess the email text data.
4. **Feature Engineering**: Transform text data into numerical features that the model can process.
5. **Model Training**: Train a machine learning model to classify emails.
6. **Evaluation**: Evaluate model performance using accuracy, precision, recall, and F1-score.
7. **Testing and Improvement**: Test the model on new samples and improve if necessary.

## Detailed Steps

### 1. Set Up the Environment

To set up the environment, ensure you have Python installed. Then, install the required libraries by running the following commands:

```bash
pip install pandas numpy scikit-learn nltk kagglehub
```

**Libraries Used**
1. Pandas: For data manipulation and analysis.
2. NumPy: For handling numerical operations.
3. Scikit-learn: For machine learning algorithms, data splitting, and model evaluation.
4. NLTK (Natural Language Toolkit): For text processing, tokenization, and removing stopwords.
4. KaggleHub: For downloading datasets from Kaggle directly into the project.
5. Jupyter Notebook: For interactive coding and visualizing steps (optional).

### 2. Downloading and Accessing the Dataset
The dataset is downloaded using the kagglehub library. We specify the Enron Email Dataset and print out the path to where the dataset is saved on the system.

```python
import kagglehub
path = kagglehub.dataset_download("wcukierski/enron-email-dataset")
print("Path to dataset files:", path)
```

Next, the dataset is loaded into a pandas DataFrame for easier manipulation and analysis. We inspect the first few rows to understand its structure.

```python
data = pd.read_csv(f"{path}/emails.csv") 
print(data.head())
```
### 5. Data Preprocessing
We preprocess the email text data to prepare it for model training focsuing on:
    - Convert text to lowercase.
    - Remove special characters and numbers.
    - Tokenize the text and remove stopwords using NLTK.
    - Store the cleaned text in a new column, processed_text.

``` python
import nltk
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['processed_text'] = data['message'].apply(preprocess_text)
```

### 6. Feature Engineering
We convert the processed text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency), limiting it to the top 3000 features for simplicity. This transforms the text into a format that the model can work with.

```
python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X = tfidf_vectorizer.fit_transform(data['processed_text']).toarray()
y = data['label'].apply(lambda x: 1 if x == 'phishing' else 0)  # Convert labels to binary
```

### 7. Splitting the Dataset
We split the dataset into training and testing sets, with 80% for training and 20% for testing. This helps evaluate the model's performance on unseen data.

```
python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 8. Model Training
We use a Naive Bayes classifier, which is effective for text classification tasks. The model is trained on the training data.

```
python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
```

### 9. Making Predictions and Evaluation
After training the model, we use it to make predictions on the test set. We then calculate evaluation metrics, such as accuracy, precision, recall, and F1-score, to assess model performance.

```
python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
```


## What I Learned
Through this project, I learned how to preprocess text data for NLP tasks, including steps like tokenization, stopword removal, and lowercasing, which are crucial for cleaning raw text data. Working with TF-IDF taught me how to transform text into numerical features that machine learning models can understand, which was eye-opening. I also gained a deeper understanding of model evaluation metrics beyond accuracy, such as precision, recall, and F1-score, and how each metric is valuable in understanding the performance of a classification model. Finally, I learned how to use libraries like NLTK and scikit-learn together to build an end-to-end machine learning pipeline for text classification, which has given me more confidence in my ability to work on similar NLP projects in the future.

# Conclusion
This project guides you through the end-to-end process of building a phishing email detection model, from data preprocessing to model training and evaluation. By following these steps, you can understand the basics of text classification, feature engineering with TF-IDF, and model evaluation using essential metrics.