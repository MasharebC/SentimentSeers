import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from textblob import TextBlob

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
def load_data(file_path):
    """
    Load the dataset from a CSV file and select relevant columns.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        DataFrame: Processed DataFrame with 'text' and 'label' columns.
    """
    df = pd.read_csv(file_path)
    df = df[['text', 'label']]
    return df

# Preprocess text
def preprocess_text(text):
    """
    Clean and preprocess text data by removing URLs, special characters, and stopwords.
    Args:
        text (str): Input text to be cleaned.
    Returns:
        str: Processed text.
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Sentiment analysis using TextBlob
def get_sentiment(text):
    """
    Compute the sentiment polarity of the text.
    Args:
        text (str): Input text.
    Returns:
        float: Sentiment polarity score (-1 to 1).
    """
    return TextBlob(text).sentiment.polarity

# Detect fake tweets
def detect_fake_tweets(df):
    """
    Detect fake tweets based on sentiment analysis (simple rule-based approach).
    Args:
        df (DataFrame): Input DataFrame with tweets.
    Returns:
        DataFrame: DataFrame with added 'sentiment' and 'fake' columns.
    """
    df['sentiment'] = df['text'].apply(get_sentiment)
    df['fake'] = df['sentiment'].apply(lambda x: 1 if x == 0 else 0)  # Consider neutral sentiment as fake
    return df

# Train ML model
def train_model(df):
    """
    Train a Naive Bayes classifier to classify tweets as real or fake.
    Args:
        df (DataFrame): Input DataFrame with processed text and labels.
    Returns:
        Pipeline: Trained machine learning model.
    """
    df['processed_text'] = df['text'].apply(preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42)
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    return model

# Visualize results
def visualize_results(df):
    """
    Generate and display sentiment distribution and word cloud visualization.
    Args:
        df (DataFrame): Input DataFrame with sentiment scores.
    """
    plt.figure(figsize=(10,5))
    sns.histplot(df['sentiment'], bins=30, kde=True)
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment Polarity")
    plt.ylabel("Frequency")
    plt.show()
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Most Common Words in Tweets")
    plt.show()

# Main function
def main():
    """
    Main execution function to load data, process, train, and visualize results.
    """
    file_path = 'tweets.csv'  # Update this with the actual path
    df = load_data(file_path)
    df = detect_fake_tweets(df)
    model = train_model(df)
    visualize_results(df)
    
if __name__ == "__main__":
    main()
