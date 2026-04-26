import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    print("--- Diabetes Prediction ---")
    
    try:
        df = pd.read_csv('dataset/diabetes.csv')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Part A
    print("\n--- Part A: Preprocessing & EDA ---")
    cols_with_zeros = ['Glucose', 'BloodPressure', 'BMI']
    for col in cols_with_zeros:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    
    df = df.fillna(df.median())

    print("\nVisualizing distributions...")
    df.hist(figsize=(10, 8), bins=20)
    plt.tight_layout()
    plt.show()

    print("\nClass imbalance analysis (Outcome):")
    if 'Outcome' in df.columns:
        print(df['Outcome'].value_counts())

    # Part B
    print("\n--- Part B: Model Building ---")
    X = df.drop('Outcome', axis=1) if 'Outcome' in df.columns else df.iloc[:, :-1]
    y = df['Outcome'] if 'Outcome' in df.columns else df.iloc[:, -1]

    print("Splitting dataset into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Applying Gaussian Naive Bayes classifier...")
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
    print("The model evaluates the probability of diabetes based on medical predictor variables using Gaussian distributions.")

if __name__ == "__main__":
    main()