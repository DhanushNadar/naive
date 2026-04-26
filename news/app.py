import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

def main():
    print("--- News Classification ---")
    
    try:
        df = pd.read_csv('dataset/news.csv')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Part A
    print("\n--- Part A: Preprocessing & EDA ---")
    print("Category distribution:")
    if 'Category' in df.columns:
        print(df['Category'].value_counts())

    print("\nPreprocessing text...")
    df['CleanText'] = df['NewsText'].apply(preprocess_text)
    
    print("Converting text into TF-IDF features...")
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['CleanText'])
    y = df['Category']

    # Part B
    print("\n--- Part B: Model Building ---")
    print("Splitting dataset into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Applying Multinomial Naive Bayes...")
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)

    # Part C
    print("\n--- Part C: Evaluation ---")
    y_pred = mnb.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (Macro): {prec:.4f}")
    print(f"Recall (Macro):    {rec:.4f}")
    print(f"F1-score (Macro):  {f1:.4f}")
    
    print("\nPlotting Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    print("\n--- Interpretation ---")
    print("Multinomial Naive Bayes is highly effective for text classification. Term frequency is crucial here.")

if __name__ == "__main__":
    main()