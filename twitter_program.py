import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    stop_words = set(stopwords.words("english"))
    return " ".join(word for word in text.split() if word not in stop_words)

def add_features(df):
    if 'target' in df.columns:
        df['label'] = df['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
    df['processed_text'] = df['text'].apply(preprocess_text)
    df['sentiment'] = df['processed_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['processed_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['text_length'] = df['text'].apply(len)
    df['punctuation_count'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c in '!?.,'))

    def get_sentiment_label(score):
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    df['sentiment_label'] = df['sentiment'].apply(get_sentiment_label)
    return df

def detect_bots(df, threshold=0.1):
    def rule(row):
        if abs(row['sentiment']) < threshold and row['subjectivity'] < 0.3:
            return "bot"
        return "human"
    df['bot_prediction'] = df.apply(rule, axis=1)
    return df

def train_and_evaluate(df):
    if 'label' not in df.columns:
        print("No label column found – skipping supervised ML training.")
        return None

    X = df[['processed_text', 'sentiment', 'subjectivity', 'text_length', 'punctuation_count']]
    y = df['label']

    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000))
    ])
    numeric_features = ['sentiment', 'subjectivity', 'text_length', 'punctuation_count']
    numeric_pipeline = Pipeline([
        ('scale', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('text', text_pipeline, 'processed_text'),
        ('num', numeric_pipeline, numeric_features)
    ])

    model = Pipeline([
        ('preprocess', preprocessor),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n🔍 Supervised ML Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model

def cluster_tweets(df, n_clusters=3):
    features = df[['sentiment', 'subjectivity', 'text_length', 'punctuation_count']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled)

    print(f"\n📊 Cluster distribution:\n{df['cluster'].value_counts()}")
    return df

def report_clusters(df):
    print("\n🧮 Cluster Sentiment Breakdown:")
    cluster_summary = df.groupby(['cluster', 'sentiment_label']).size().unstack(fill_value=0)
    print(cluster_summary)

    print("\n📋 Bot Prediction Breakdown per Cluster:")
    bot_summary = df.groupby(['cluster', 'bot_prediction']).size().unstack(fill_value=0)
    print(bot_summary)

def visualize(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(df['sentiment'], bins=30, kde=True)
    plt.title("Sentiment Polarity Distribution")
    plt.xlabel("Sentiment Polarity")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.histplot(df['subjectivity'], bins=30, kde=True)
    plt.title("Subjectivity Distribution")
    plt.xlabel("Subjectivity")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.countplot(x='sentiment_label', data=df, palette="Set2", order=['positive', 'neutral', 'negative'])
    plt.title("Sentiment Label Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='sentiment', y='subjectivity', hue='cluster', palette='tab10')
    plt.title("Tweet Clusters (KMeans)")
    plt.xlabel("Sentiment")
    plt.ylabel("Subjectivity")
    plt.legend(title='Cluster')
    plt.show()

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['processed_text']))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Most Common Words")
    plt.show()

def process_twitter(file_path):
    print("\n📦 Processing Twitter Dataset...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print("❌ Failed to load file:", e)
        return

    if 'text' not in df.columns:
        print("❌ Missing required column 'text'")
        return

    df = add_features(df)
    df = detect_bots(df)
    train_and_evaluate(df)
    df = cluster_tweets(df)
    report_clusters(df)
    visualize(df)

    print("\n🧾 Sample Output:")
    print(df[['text', 'sentiment_label', 'bot_prediction', 'cluster']].head(10))

    df.to_csv("twitter_processed_reviews.csv", index=False)
    print("\n📁 Results saved to 'twitter_processed_reviews.csv'")
