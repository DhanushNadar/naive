import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    print("--- Loan Approval Prediction ---")
    
    try:
        df = pd.read_csv('dataset/train_u6lujuX_CVtuZ9i.csv')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Part A
    print("\n--- Part A: Preprocessing & EDA ---")
    if 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)
    
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    print("\nEncoding categorical variables...")
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    print("\nAnalyzing dataset distribution...")
    print(df.describe())

    # Part B
    print("\n--- Part B: Model Building ---")
    X = df.drop('Loan_Status', axis=1) if 'Loan_Status' in df.columns else df.iloc[:, :-1]
    y = df['Loan_Status'] if 'Loan_Status' in df.columns else df.iloc[:, -1]

    print("Splitting dataset into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Applying Naive Bayes classifier...")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # Part C
    print("\n--- Part C: Evaluation ---")
    y_pred = gnb.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
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
    print("The model evaluates factors like Income and LoanAmount to predict loan approval. Naive Bayes assumes feature independence.")

if __name__ == "__main__":
    main()