import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

def process_yelp(file_path):
    print("\n📦 Processing Yelp Review Dataset...")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print("❌ Failed to load file:", e)
        return

    print(f"\n📋 Columns in dataset: {list(df.columns)}")
    print(f"Number of reviews: {len(df)}")
    print("\nFirst 5 rows:")
    print(df.head())

    # Clean and enrich
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()

    # Sentiment analysis
    def analyze_sentiment(text):
        polarity = TextBlob(str(text)).sentiment.polarity
        if polarity > 0:
            return 'Positive'
        elif polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'

    df['sentiment'] = df['text'].apply(analyze_sentiment)
    df['review_length'] = df['text'].apply(lambda x: len(str(x).split()))

    # Star rating distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='stars', data=df, palette='viridis')
    plt.title('Distribution of Star Ratings')
    plt.savefig("yelp_star_distribution.png")
    plt.show()

    # Sentiment distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment', data=df, palette='Set2', order=['Positive', 'Neutral', 'Negative'])
    plt.title('Sentiment Analysis of Reviews')
    plt.savefig("yelp_sentiment_distribution.png")
    plt.show()

    # Reviews over time
    reviews_over_time = df.groupby(['year', 'month']).size().unstack().fillna(0)
    reviews_over_time.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Number of Reviews Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Reviews')
    plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("yelp_reviews_over_time.png")
    plt.show()

    # Engagement metrics
    if all(col in df.columns for col in ['useful', 'funny', 'cool']):
        engagement_metrics = df[['useful', 'funny', 'cool']].sum()
        plt.figure(figsize=(8, 6))
        engagement_metrics.plot(kind='bar', color=['blue', 'orange', 'green'])
        plt.title('Engagement Metrics (Total Counts)')
        plt.savefig("yelp_engagement_metrics.png")
        plt.show()

        # Correlation
        correlation = df[['stars', 'useful', 'funny', 'cool']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Between Stars and Engagement Metrics')
        plt.savefig("yelp_correlation.png")
        plt.show()

    # Top businesses
    if 'business_id' in df.columns:
        top_businesses = df['business_id'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        top_businesses.plot(kind='bar')
        plt.title('Top 10 Businesses by Number of Reviews')
        plt.xlabel('Business ID')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45)
        plt.savefig("yelp_top_businesses.png")
        plt.show()

    # Review length by star rating
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='stars', y='review_length', data=df)
    plt.title('Review Length by Star Rating')
    plt.savefig("yelp_review_length.png")
    plt.show()

    # Sentiment by star rating
    plt.figure(figsize=(10, 6))
    sns.countplot(x='stars', hue='sentiment', data=df, palette='Set2')
    plt.title('Sentiment Distribution by Star Rating')
    plt.savefig("yelp_sentiment_by_star.png")
    plt.tight_layout()
    plt.show()

    # Text summary
    print("\n📊 Key Statistics:")
    print(f"Average star rating: {df['stars'].mean():.2f}")
    print(f"Most common star rating: {df['stars'].mode()[0]}")
    print(f"Percentage of positive sentiment reviews: {(df['sentiment'] == 'Positive').mean()*100:.2f}%")
    print(f"Average review length: {df['review_length'].mean():.2f} words")

    df.to_csv("yelp_processed_reviews.csv", index=False)
    print("\n📁 Output saved as 'yelp_processed_reviews.csv'")
