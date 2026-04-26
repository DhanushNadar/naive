import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import glob

def main():
    print("--- Employee Attrition Prediction ---")
    
    try:
        files = glob.glob('dataset/*.csv')
        if not files:
            print("Dataset not found!")
            return
        df = pd.read_csv(files[0])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Part A
    print("\n--- Part A: Preprocessing & EDA ---")
    df = df.dropna()

    print("Encoding categorical features...")
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    print("\nExploratory Analysis - Attrition Distribution:")
    if 'Attrition' in df.columns:
        print(df['Attrition'].value_counts())

    # Part B
    print("\n--- Part B: Model Building ---")
    X = df.drop('Attrition', axis=1) if 'Attrition' in df.columns else df.iloc[:, :-1]
    y = df['Attrition'] if 'Attrition' in df.columns else df.iloc[:, -1]

    print("Splitting dataset into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Applying Naive Bayes classifier...")
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
    print("Naive Bayes helps predict if an employee will leave the company based on their demographic and job role attributes.")

if __name__ == "__main__":
    main()