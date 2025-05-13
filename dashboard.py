import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from lime.lime_text import LimeTextExplainer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import kagglehub

# Configure page
st.set_page_config(
    page_title="Phishing Email Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    h1, h2 {color: #1E3A8A;}
    .metric-container {background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

# Helper function for text preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d', ' ', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    return ""

# Extract phishing indicators
def extract_indicators(email_text):
    if not isinstance(email_text, str):
        return {"has_url": 0, "urgent_words": 0, "sensitive_requests": 0}
    
    indicators = {}
    # URL detection
    indicators["has_url"] = 1 if re.search(r'https?://\S+', email_text) else 0
    
    # Count urgent words
    urgent_words = ['urgent', 'immediately', 'alert', 'verify', 'suspended', 'account']
    indicators["urgent_words"] = sum(word in email_text.lower() for word in urgent_words)
    
    # Detect sensitive info requests
    indicators["sensitive_requests"] = 1 if re.search(r'password|credit.?card|ssn|bank|account', email_text.lower()) else 0
    
    return indicators

# Load dataset
def load_data():
    path = kagglehub.dataset_download("shantanudhakadd/email-spam-detection-dataset-classification")
    dataset_path = f"{path}/spam.csv"
    data = pd.read_csv(dataset_path, encoding='ISO-8859-1')
    data = data.rename(columns={data.columns[0]: 'label', data.columns[1]: 'text'})
    return data

@st.cache_resource
def load_and_process_data():
    data = load_data()
    data['processed'] = data['text'].apply(preprocess_text)
    data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})
    return data

@st.cache_resource
def train_models(X_train, _X_train_vec, y_train, _X_test_vec, y_test):
    models = {
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=50), 
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(_X_train_vec, y_train)
        y_pred = model.predict(_X_test_vec)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'cm': confusion_matrix(y_test, y_pred)
        }
    return models, results

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
               xticklabels=['Legitimate', 'Phishing'],
               yticklabels=['Legitimate', 'Phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    return fig

def retrain_with_feedback(X_train, y_train, vectorizer, models):
    if os.path.exists('feedback_log.csv'):
        feedback_data = pd.read_csv('feedback_log.csv')
        # Process feedback data
        feedback_data['processed'] = feedback_data['text'].apply(preprocess_text)
        feedback_data['label_num'] = feedback_data['label'].apply(
            lambda x: 1 if 'Phishing' in x else 0
        )
        # Combine with original training data
        combined_X = pd.concat([X_train, feedback_data['processed']])
        combined_y = pd.concat([y_train, feedback_data['label_num']])
        # Retrain all models
        X_vec = vectorizer.transform(combined_X)
        for name, model in models.items():
            model.fit(X_vec, combined_y)
        return "All models retrained with feedback data!"
    return "No feedback data available for retraining."

def main():
    st.title('🛡️ Phishing Email Detection Using NLP')
    st.markdown('### Advanced NLP and Machine Learning for Email Security')

    # Download NLTK data
    nltk.download('stopwords')
    nltk.download('punkt')

    # Load and process data
    with st.spinner("Loading dataset and training models..."):
        data = load_and_process_data()
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            data['processed'], data['label_num'], test_size=0.2, random_state=42
        )
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train models
        models, results = train_models(X_train, X_train_vec, y_train, X_test_vec, y_test)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Live Detection", "📈 Model Insights", "📝 Feedback"])
    
    # Tab 1: Dashboard Overview
    with tab1:
        st.header("Email Security Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Emails", len(data))
        with col2:
            st.metric("Phishing Emails", sum(data['label_num'] == 1), f"{100*sum(data['label_num'] == 1)/len(data):.1f}%")
        with col3:
            st.metric("Legitimate Emails", sum(data['label_num'] == 0))
        with col4:
            best_model = max(results.items(), key=lambda x: x[1]['f1'])[0]
            st.metric("Best Model", best_model, f"{results[best_model]['f1']:.2f} F1")
            
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Email Distribution")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.pie([sum(data["label_num"] == 1), sum(data["label_num"] == 0)], 
                   labels=['Phishing', 'Legitimate'], 
                   autopct='%1.1f%%', 
                   startangle=90, 
                   colors=['#FF5252', '#4CAF50'],
                   explode=(0.1, 0))
            ax1.axis('equal')
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Model Performance")
            # Create data for model comparison
            metrics_data = []
            for model_name, metrics in results.items():
                metrics_data.append([model_name, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']])
            
            metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
            st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score']), 
                         use_container_width=True)
            
        st.subheader("Email Length Analysis")
        data['text_length'] = data['text'].str.len()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=data, x='text_length', hue='label', bins=30, kde=True)
        plt.title('Distribution of Email Lengths')
        plt.xlabel('Email Length (characters)')
        st.pyplot(fig)
    
    # Tab 2: Live Detection
    with tab2:
        st.header("Live Email Analysis")
        
        st.markdown("""
        ### Analyze an email for phishing indicators
        Paste an email below to analyze its content and determine if it's phishing or legitimate.
        """)
        
        user_input = st.text_area('Email Content:', height=200)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            model_choice = st.selectbox("Select Model:", list(models.keys()))
        
        detect_button = st.button('🔍 Detect Phishing', type="primary", use_container_width=True)
        
        if detect_button and user_input:
            try:
                # Process text
                processed = preprocess_text(user_input)
                vec = vectorizer.transform([processed])
                pred = models[model_choice].predict(vec)[0]
                prob = models[model_choice].predict_proba(vec)[0]
                
                # Get indicators
                indicators = extract_indicators(user_input)
                
                # Display result in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    if pred == 1:
                        st.markdown(f"<div style='background-color:#FFE0E0; padding:20px; border-radius:5px;'>"
                                    f"<h2 style='color:#D32F2F; margin:0;'>⚠️ Phishing Detected</h2>"
                                    f"<p>Confidence: {prob[1]*100:.1f}%</p></div>", 
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='background-color:#E0F2E9; padding:20px; border-radius:5px;'>"
                                    f"<h2 style='color:#2E7D32; margin:0;'>✅ Legitimate Email</h2>"
                                    f"<p>Confidence: {prob[0]*100:.1f}%</p></div>", 
                                    unsafe_allow_html=True)
                    
                    st.markdown("### Risk Indicators")
                    if indicators['has_url']:
                        st.warning("⚠️ Contains URLs - potential phishing links")
                    if indicators['urgent_words'] > 2:
                        st.warning(f"⚠️ High urgency language ({indicators['urgent_words']} urgent terms)")
                    if indicators['sensitive_requests']:
                        st.warning("⚠️ Requests sensitive information")
                
                with col2:
                    st.markdown("### Why this prediction?")
                    try:
                        explainer = LimeTextExplainer(class_names=['Legitimate', 'Phishing'])
                        exp = explainer.explain_instance(
                            user_input, 
                            lambda x: models[model_choice].predict_proba(vectorizer.transform([preprocess_text(i) for i in x])), 
                            num_features=6
                        )
                        st.components.v1.html(exp.as_html(), height=350)
                    except Exception as e:
                        st.error(f"Could not generate explanation: {e}")
            
            except Exception as e:
                st.error(f"Error analyzing email: {str(e)}")
    
    # Tab 3: Model Insights
    with tab3:
        st.header("Advanced Model Analysis")
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        cols = st.columns(3)
        for i, (model_name, metrics) in enumerate(results.items()):
            with cols[i % 3]:
                st.markdown(f"**{model_name}**")
                cm_fig = plot_confusion_matrix(metrics['cm'])
                st.pyplot(cm_fig)
        
        # Common phishing words
        st.subheader("Common Words in Phishing Emails")
        # Get most common words in phishing emails
        phishing_words = ' '.join(data[data['label'] == 'spam']['processed'])
        phishing_word_counts = {}
        for word in phishing_words.split():
            if len(word) > 3:  # Only count words of reasonable length
                phishing_word_counts[word] = phishing_word_counts.get(word, 0) + 1
        
        # Convert to DataFrame and sort
        if phishing_word_counts:
            word_df = pd.DataFrame({
                'Word': list(phishing_word_counts.keys()),
                'Count': list(phishing_word_counts.values())
            }).sort_values('Count', ascending=False).head(20)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Count', y='Word', data=word_df, palette='Reds_r')
            plt.title('Top 20 Words in Phishing Emails')
            st.pyplot(fig)
        
        # Educational section
        st.subheader("NLP in Phishing Detection")
        st.markdown("""
        ### How Our System Uses NLP
        
        1. **Text Preprocessing**
           - Converting to lowercase
           - Removing special characters and numbers
           - Tokenizing text into individual words
           - Removing stopwords (common words like "the", "and")
        
        2. **TF-IDF Vectorization**
           - Converting text into numerical features
           - Giving higher weight to important/distinctive words
           - Reducing weight of common words across all emails
        
        3. **Machine Learning Classification**
           - Training models on labeled examples
           - Identifying patterns that distinguish phishing from legitimate emails
           - Using multiple algorithms for robustness
        
        4. **Explainable AI**
           - Using LIME to explain why an email is classified as phishing
           - Highlighting suspicious words and phrases
           - Providing transparency in detection decisions
        """)
    
    # Tab 4: Feedback
    with tab4:
        st.header("User Feedback System")
        
        st.markdown("""
        ### Help Improve Our Phishing Detection
        
        If you find an email that was incorrectly classified, submit it below to improve the model.
        """)
        
        feedback_email = st.text_area('Email content:', height=150)
        feedback_label = st.selectbox('Correct classification:', ['Legitimate (Ham)', 'Phishing (Spam)'])
        
        if st.button('Submit Feedback'):
            if not feedback_email:
                st.warning("Please enter some email content.")
            else:
                # Log feedback to a CSV file
                feedback_df = pd.DataFrame({'text': [feedback_email], 'label': [feedback_label]})
                if os.path.exists('feedback_log.csv'):
                    feedback_df.to_csv('feedback_log.csv', mode='a', header=False, index=False)
                else:
                    feedback_df.to_csv('feedback_log.csv', index=False)
                
                st.success('Feedback submitted! Thank you for helping improve our model.')
                retrain_message = retrain_with_feedback(X_train, y_train, vectorizer, models)
                st.info(retrain_message)
        
        # Show feedback history if available
        if os.path.exists('feedback_log.csv'):
            st.subheader("Recent Feedback Submissions")
            feedback_history = pd.read_csv('feedback_log.csv')
            st.dataframe(feedback_history.tail(5), use_container_width=True)
            
    # Sidebar with project info
    with st.sidebar:
        st.title("Project Information")
        st.markdown("""
        ### NLP-Based Phishing Detection
        
        This project uses Natural Language Processing (NLP) to detect phishing emails by analyzing text content.
        
        **Key Features:**
        - Text preprocessing with NLTK
        - TF-IDF vectorization
        - Multiple ML models comparison
        - Explainable AI with LIME
        - Real-time email analysis
        - User feedback integration
        
        **Accuracy**: ~98%  
        **Precision**: ~99%
        """)
        
        st.markdown("---")
        st.markdown("Created for Advanced NLP Course")
        
if __name__ == '__main__':
    main()
