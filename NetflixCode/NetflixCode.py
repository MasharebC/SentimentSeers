import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
try:
    print("Loading dataset...")
    df = pd.read_csv('/Users/jayant/Desktop/Updated_Netflix_Reviews.csv')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Inspect the data
print("\nFirst few rows of the dataset:")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nDataset Description:")
print(df.describe())

# ---------------------------------
# 1. Dataframe Operations
# ---------------------------------
try:
    # Histogram of star ratings
    print("Starting histogram plot...")
    plt.figure(figsize=(10, 6))
    df['stars'].hist(bins=10, color='blue')
    plt.title('Histogram of Star Ratings')
    plt.xlabel('Star Ratings')
    plt.ylabel('Frequency')
    plt.show()
    print("Histogram plot completed.")

    # Correlation matrix
    print("Starting correlation matrix...")
    numeric_columns = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_columns.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    plt.show()
    print("Correlation matrix completed.")

    # Pivot table for average stars by sentiment
    print("Calculating average stars by sentiment...")
    pivot_table_example = df.pivot_table(values='stars', index='sentiment', aggfunc='mean')
    print(pivot_table_example)

    # Bar plot for average stars by sentiment
    print("Starting bar plot for average stars by sentiment...")
    plt.figure(figsize=(8, 6))
    sns.barplot(x=pivot_table_example.index, y=pivot_table_example['stars'], palette='coolwarm')
    plt.title('Average Stars by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Average Rating')
    plt.show()
    print("Bar plot for average stars completed.")

    # Grouping data
    grouped = df.groupby('sentiment')['stars'].mean()
    print("\nGrouped Average Stars by Sentiment:")
    print(grouped)

    # Merging data (example)
    print("Performing data merge...")
    merged_df = df.drop_duplicates(subset=['sentiment']).merge(df, on='sentiment', suffixes=('_left', '_right'))
    print("\nMerged DataFrame Shape:", merged_df.shape)
    print("\nMerged DataFrame Sample:")
    print(merged_df.head())
except Exception as e:
    print(f"Error during dataframe operations: {e}")

# ---------------------------------
# 2. Data Preprocessing
# ---------------------------------
try:
    print("\nStarting data preprocessing...")
    
    # Handle 'review_date'
    if 'review_date' in df.columns:
        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    else:
        print("'review_date' column not found, skipping date processing.")

    # Text preprocessing
    if 'review' in df.columns:
        df['cleaned_review'] = df['review'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        print("\nSample cleaned reviews:")
        print(df['cleaned_review'].head())
    else:
        print("'review' column not found, skipping text preprocessing.")

    # Handling missing values
    df['stars'] = df['stars'].fillna(df['stars'].mean())

    # Encoding categorical features
    label_encoder = LabelEncoder()
    df['type_encoded'] = label_encoder.fit_transform(df['type'])

    # Scaling numerical features
    scaler = StandardScaler()
    df['stars_scaled'] = scaler.fit_transform(df[['stars']])
    print("Data preprocessing completed successfully.")
except Exception as e:
    print(f"Error during data preprocessing: {e}")

# ---------------------------------
# 3. Building and Evaluating Models
# ---------------------------------
try:
    print("\nStarting feature extraction...")
    if 'cleaned_review' in df.columns:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(df['cleaned_review'].fillna(''))
        print("Feature extraction completed.")

        # Ensure target variable exists
        if 'sentiment' in df.columns:
            y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print("Training and testing data created successfully.")

            # Logistic Regression
            classifier = LogisticRegression()
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
            print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, y_pred))
        else:
            print("'sentiment' column not found, skipping model training.")
    else:
        print("'cleaned_review' column missing, feature extraction skipped.")
except Exception as e:
    print(f"Error during model training/evaluation: {e}")

# ---------------------------------
# 4. Clustering and Feature Analysis
# ---------------------------------
try:
    print("\nStarting KMeans clustering...")
    if 'cleaned_review' in df.columns:
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)
        print("\nCluster Distribution:")
        print(df['cluster'].value_counts())

        # Visualizing clusters
        plt.figure(figsize=(8, 6))
        sns.countplot(x=df['cluster'])
        plt.title('Cluster Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Reviews')
        plt.show()
        print("Clustering and visualization completed.")
    else:
        print("Clustering skipped due to missing feature matrix.")
except Exception as e:
    print(f"Error during clustering: {e}")

# ---------------------------------
# 5. Feature Analysis
# ---------------------------------
try:
    print("\nAnalyzing feature importance...")
    if 'cleaned_review' in df.columns and 'sentiment' in df.columns:
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)
        rf_importances = pd.Series(rf_classifier.feature_importances_, index=vectorizer.get_feature_names_out())
        print("\nTop Features by Importance (Random Forest):")
        print(rf_importances.nlargest(10))

        # Visualize feature importance
        rf_importances.nlargest(10).plot(kind='bar', color='teal')
        plt.title('Top Features by Importance (Random Forest)')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.show()
        print("Feature analysis completed.")
    else:
        print("Feature analysis skipped due to missing data.")
except Exception as e:
    print(f"Error during feature analysis: {e}")
