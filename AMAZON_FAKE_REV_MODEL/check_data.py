import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('final_dataset.csv', encoding='ISO-8859-1')

# Drop rows with missing values in critical columns
data = data.dropna(subset=['review_bold', 'ratings', 'verified', 'review_sentiment'])

# Normalize numeric features
scaler = MinMaxScaler()
data[['ratings', 'review_sentiment', 'helpful', 'most_rev']] = scaler.fit_transform(
    data[['ratings', 'review_sentiment', 'helpful', 'most_rev']]
)

# Adjusted heuristic for labeling fake reviews
data['is_fake'] = (
    (data['verified'] == 0) & 
    ((data['helpful'] < 2) | (data['most_rev'] > 10) | (data['review_sentiment'] > 0.8))
).astype(int)

# Define features and target
X = data[['ratings', 'review_sentiment', 'helpful', 'most_rev']]
y = data['is_fake']

# Print feature statistics by class
# Filter numeric columns before calculating the mean
numeric_cols = data.select_dtypes(include=['number']).columns
print("Feature means by class (is_fake):\n", data.groupby('is_fake')[numeric_cols].mean())


# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("\nCross-Validation Scores:", scores)
print("Mean Accuracy:", np.mean(scores))

# Feature importance visualization
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.show()
