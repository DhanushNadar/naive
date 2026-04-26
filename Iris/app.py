import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("./dataset/iris.data", header=None)

df.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

print("\n===== DATA PREVIEW =====")
print(df.head())

# =========================
# CHECK MISSING VALUES
# =========================
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# =========================
# STATISTICAL SUMMARY
# =========================
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# =========================
# VISUALIZATION
# =========================
print("\nShowing Histograms...")
df.hist(figsize=(8,6))
plt.tight_layout()
plt.show()

# Class Distribution
print("\n===== CLASS DISTRIBUTION =====")
print(df['Species'].value_counts())

sns.countplot(x='Species', data=df)
plt.title("Class Distribution")
plt.show()

# =========================
# MODEL BUILDING
# =========================
X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================
print("\n===== MODEL EVALUATION =====")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# INTERPRETATION
# =========================
print("\n===== INTERPRETATION =====")
print("Model successfully classifies Iris species using Naive Bayes.")
print("High accuracy indicates good performance.")