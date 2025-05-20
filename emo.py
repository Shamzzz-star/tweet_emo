# svm_emotion_classifier.py

import pandas as pd
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('emotions.csv')  # Replace with the correct path if needed

# Balance dataset (sample 7000 entries per label)
df_sample = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=7000, random_state=42),
    include_groups=True
)
df = df_sample.reset_index(drop=True)

# Text cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+|[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train and evaluate SVM models with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = []

for kernel in kernels:
    print(f"\nTraining SVM with {kernel} kernel...")
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results.append((kernel, acc))

    print(f"\n--- {kernel.upper()} Kernel ---")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f'Confusion Matrix - {kernel.capitalize()} Kernel')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # Save the model
    model_filename = f'svm_model_{kernel}.joblib'
    joblib.dump(model, model_filename)
    print(f"✅ Saved model to {model_filename}")

# Print best kernel
best_kernel = max(results, key=lambda x: x[1])
print(f"\n✅ Best performing kernel: {best_kernel[0]} with accuracy: {best_kernel[1]:.4f}")
