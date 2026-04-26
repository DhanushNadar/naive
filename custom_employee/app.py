import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    print("--- Employee Attrition Prediction ---")
    
    try:
        df = pd.read_csv('dataset/data.csv')
    except Exception as e:
        print(f"Error loading dataset: {{e}}")
        return

    print("\n--- Part A: Preprocessing & EDA ---")
    print("Missing values before cleaning:")
    print(df.isnull().sum())
    
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())
            
    print("\nStatistical summary:")
    print(df.describe())
    
    print("\nVisualizing feature distributions...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols].hist(figsize=(10, 8), bins=15)
        plt.tight_layout()
        plt.show()

    print("\nClass distribution (Attrition):")
    print(df['Attrition'].value_counts())

    print("\n--- Part B: Model Building ---")
    
    le = LabelEncoder()
    df['Attrition'] = le.fit_transform(df['Attrition'].astype(str))
    
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and col != 'Attrition':
            df[col] = le.fit_transform(df[col].astype(str))
            
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    print("Splitting dataset into training and testing sets (70/30)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Applying Gaussian Naive Bayes classifier...")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    print("\n--- Part C: Evaluation ---")
    y_pred = gnb.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"Accuracy:  {{acc:.4f}}")
    print(f"Precision (Macro): {{prec:.4f}}")
    print(f"Recall (Macro):    {{rec:.4f}}")
    print(f"F1-score (Macro):  {{f1:.4f}}")
    
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
    print("Gaussian Naive Bayes classifier results on the synthetic dataset.")

if __name__ == "__main__":
    main()
