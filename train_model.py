import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
data = pd.read_csv("dataset.csv")

X = data["text"]
y = data["label"]

# Convert Text into Numerical Features
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.25, random_state=42, stratify=y
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed!")
print("Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["Phishing", "Safe"])

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Phishing", "Safe"],
    yticklabels=["Phishing", "Safe"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Phishing Email Detection - Confusion Matrix")
plt.show()

# Save Model
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
# Save Model Accuracy
with open("model_results.txt", "w") as file:
    file.write("PHISHING EMAIL DETECTION MODEL RESULTS\n")
    file.write("--------------------------------------\n")
    file.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")
    file.write("\nClassification Report:\n")
    file.write(classification_report(y_test, y_pred))

print("Model results saved successfully!")

print("\nModel saved successfully!")