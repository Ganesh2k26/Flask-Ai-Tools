import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle

# Load and clean dataset
df = pd.read_csv('D:\GANESH\MINI PROJECTS\Flask\spam_mail\model\spam.csv', encoding='ISO-8859-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("Label distribution:\n", df['label'].value_counts())

# Split with stratification to preserve label proportions
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Resample training data
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train.to_frame(), y_train)

# Build model pipeline with resampled data
model = ImbPipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train and evaluate
model.fit(X_train_resampled['message'], y_train_resampled)
pred = model.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, pred))

# Save model
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Balanced model trained and saved as spam_model.pkl")
