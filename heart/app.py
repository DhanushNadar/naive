import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    print("--- Heart Disease Prediction ---")
    
    try:
        df = pd.read_csv('dataset/heart.csv')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Part A
    print("\n--- Part A: Preprocessing & EDA ---")
    df = df.dropna()
    
    print("\nVisualizing feature relationships...")
    if all(col in df.columns for col in ['age', 'chol', 'target']):
        sns.pairplot(df[['age', 'chol', 'trestbps', 'target']], hue='target', diag_kind='kde')
        plt.tight_layout()
        plt.show()
    else:
        print("Missing expected columns for pairplot visualization.")

    # Part B
    print("\n--- Part B: Model Building ---")
    X = df.drop('target', axis=1) if 'target' in df.columns else df.iloc[:, :-1]
    y = df['target'] if 'target' in df.columns else df.iloc[:, -1]

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
    print("Gaussian Naive Bayes works well considering medical measurements often roughly follow normal distributions.")

if __name__ == "__main__":
    main()