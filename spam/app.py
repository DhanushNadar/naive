import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    # Lowercase
    text = str(text).lower()
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def main():
    print("--- Spam Email Detection ---")
    
    # 1. Load Dataset
    print("\nLoading dataset...")
    try:
        # Assuming the standard Kaggle spam.csv formatting with 'v1' and 'v2' columns
        df = pd.read_csv('dataset/spam.csv', encoding='latin-1')
        # Keep only necessary columns and rename them
        df = df[['v1', 'v2']]
        df.columns = ['Label', 'EmailText']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Analyze class distribution (Part A)
    print("\n--- Part A: Preprocessing & EDA ---")
    print("Class Distribution:")
    print(df['Label'].value_counts())
    
    # Preprocess text
    print("\nPreprocessing text (lowercase, remove stopwords)...")
    df['CleanText'] = df['EmailText'].apply(preprocess_text)
    
    # Convert text into TF-IDF features
    print("Converting text to TF-IDF features...")
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['CleanText'])
    
    # Target variable
    y = df['Label']
    
    # Part B: Model Building
    print("\n--- Part B: Model Building ---")
    print("Splitting dataset into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Multinomial Naive Bayes model...")
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    
    # Part C: Evaluation
    print("\n--- Part C: Evaluation ---")
    y_pred = mnb.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='spam')
    rec = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    print("\nPlotting Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    # No image saved locally as requested
    plt.show()
    
    print("\n--- Interpretation ---")
    print("The Multinomial Naive Bayes classifier generally performs well on text classification tasks like spam detection.")
    print("A high Precision means few legitimate emails were caught in the spam filter (low false positives).")
    print("A high Recall means most of the actual spam emails were correctly identified (low false negatives).")

if __name__ == "__main__":
    main()
