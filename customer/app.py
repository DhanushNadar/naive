import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    print("--- Customer Purchase Prediction ---")
    
    try:
        df = pd.read_csv('dataset/Social_Network_Ads.csv')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Part A
    print("\n--- Part A: Preprocessing & EDA ---")
    df = df.dropna()
    
    if 'User ID' in df.columns:
        df = df.drop('User ID', axis=1)

    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'male': 1, 'female': 0})
        df['Gender'] = df['Gender'].fillna(0) # fallback

    print("\nVisualizing feature distributions...")
    df.hist(figsize=(10, 8), bins=20)
    plt.tight_layout()
    plt.show()

    print("\nClass distribution (Purchased):")
    if 'Purchased' in df.columns:
        print(df['Purchased'].value_counts())

    # Part B
    print("\n--- Part B: Model Building ---")
    X = df.drop('Purchased', axis=1) if 'Purchased' in df.columns else df.iloc[:, :-1]
    y = df['Purchased'] if 'Purchased' in df.columns else df.iloc[:, -1]

    print("Splitting dataset into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Applying Gaussian Naive Bayes...")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # Part C
    print("\n--- Part C: Evaluation ---")
    y_pred = gnb.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
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
    print("A Gaussian Naive Bayes model works reasonably well for predicting purchase intent based on Age and EstimatedSalary.")

if __name__ == "__main__":
    main()