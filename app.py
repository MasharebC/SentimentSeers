import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
import os

def process_amazon(file_path):
    print("\n📦 Processing Amazon Review Dataset...")

    try:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
    except Exception as e:
        print("❌ Failed to load file:", e)
        return

    print("📋 Columns in dataset:", list(data.columns))

    # Rename columns to match expected names
    data.rename(columns={
        'text': 'review_bold',
        'rating': 'ratings',
        'verified_purchase': 'verified',
        'helpful_vote': 'helpful'
    }, inplace=True)

    # Convert 'verified' from TRUE/FALSE to 1/0
    if 'verified' in data.columns:
        data['verified'] = data['verified'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)

    if 'most_rev' not in data.columns:
        print("➕ Adding missing column 'most_rev' with default value 0")
        data['most_rev'] = 0

    # Generate sentiment using VADER
    print("🧠 Calculating sentiment scores with VADER...")
    sid = SentimentIntensityAnalyzer()
    data['review_sentiment'] = data['review_bold'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

    def label_sentiment(score):
        if score > 0.05:
            return 'Positive'
        elif score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    data['sentiment_label'] = data['review_sentiment'].apply(label_sentiment)

    # Drop missing
    required_columns = {'review_bold', 'ratings', 'verified', 'review_sentiment'}
    if not required_columns.issubset(data.columns):
        print(f"❌ Missing required columns: {required_columns - set(data.columns)}")
        return
    data = data.dropna(subset=required_columns)

    # Normalize
    scaler = MinMaxScaler()
    features_to_scale = ['ratings', 'review_sentiment', 'helpful', 'most_rev']
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    # Heuristic fake review detection
    data['is_fake'] = (
        (data['verified'] == 0) &
        ((data['helpful'] < 0.2) | (data['most_rev'] > 0.8) | (data['review_sentiment'] > 0.8))
    ).astype(int)

    # Summary counts
    fake_counts = data['is_fake'].value_counts().rename(index={0: 'Real', 1: 'Fake'})
    print("\n🧮 Review Authenticity Distribution:")
    print(fake_counts.to_string())

    print("\n📈 Sentiment Summary (All Reviews):")
    print(data['review_sentiment'].describe())

    print("\n🧠 Sentiment Distribution (All Reviews):")
    print(data['sentiment_label'].value_counts())

    # Real review subset
    real_reviews = data[data['is_fake'] == 0].copy()
    print("\n📈 Sentiment Summary (Real Reviews Only):")
    print(real_reviews['review_sentiment'].describe())

    print("\n🧠 Sentiment Distribution (Real Reviews Only):")
    print(real_reviews['sentiment_label'].value_counts())

    # Top reviews
    print("\n🌟 Most Positive Review:")
    print(data.loc[data['review_sentiment'].idxmax(), 'review_bold'])

    print("\n💢 Most Negative Review:")
    print(data.loc[data['review_sentiment'].idxmin(), 'review_bold'])

    print("\n🔍 Sample of Final Processed Data:")
    print(data[['review_bold', 'ratings', 'verified', 'review_sentiment', 'sentiment_label', 'is_fake']].head(5))

    # --- Visualizations ---

    # Fake vs Real
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_fake', data=data)
    plt.xticks([0, 1], ['Real', 'Fake'])
    plt.title("Fake vs Real Reviews")
    plt.savefig("amazon_fake_vs_real.png")
    plt.show()

    # Sentiment bar plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sentiment_label', data=data, order=['Positive', 'Neutral', 'Negative'])
    plt.title("Sentiment Label Distribution")
    plt.savefig("amazon_sentiment_bar.png")
    plt.show()

    # --- Random Forest Classifier ---
    X = data[['ratings', 'review_sentiment', 'helpful', 'most_rev']]
    y = data['is_fake']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("amazon_confusion_matrix.png")
    plt.show()

    # Feature Importance
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_df.sort_values(by='Importance').plot.barh(x='Feature', title="Feature Importance")
    plt.tight_layout()
    plt.savefig("amazon_feature_importance.png")
    plt.show()

    print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

    # GPT Summary
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("\n⚠️ Skipping GPT Summary. Set your OPENAI_API_KEY as an environment variable.")
    else:
        openai.api_key = openai_api_key
        all_real_text = ' '.join(real_reviews['review_bold'].dropna().astype(str).tolist())[:12000]
        prompt = (
            "Read the following Amazon product reviews and summarize the main pros and cons. "
            "Give two concise paragraphs: one for pros, one for cons.\n\n"
            f"Reviews:\n{all_real_text}"
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            summary = response.choices[0].message.content
            with open("amazon_gpt_summary.txt", "w") as f:
                f.write(summary)
            print("\n🧠 GPT Summary:\n", summary)
        except Exception as e:
            print("❌ Error calling GPT:", e)

    # Sample review summaries
    print("\n📝 Representative Real Reviews:")
    sample_reviews = real_reviews['review_bold'].dropna().sample(3, random_state=42)
    for i, review in enumerate(sample_reviews):
        sentences = sent_tokenize(review)
        summary = " ".join(sentences[:2]) if len(sentences) >= 2 else review
        print(f"\nReview {i+1}:\n{summary}")

    # Save processed file
    data.to_csv("amazon_processed_reviews.csv", index=False)
    print("\n📁 Output saved as 'amazon_processed_reviews.csv'")

