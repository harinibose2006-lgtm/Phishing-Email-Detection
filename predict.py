import joblib

# Load Trained Model
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("===== PHISHING EMAIL DETECTION =====")

# Get Email Input
email = input("\nEnter Email Message: ")

# Convert Text into Features
email_vectorized = vectorizer.transform([email])

# Predict Email
prediction = model.predict(email_vectorized)

# Display Result
print("\nEmail Prediction:", prediction[0])

if prediction[0] == "Phishing":
    print("Warning! This email may be dangerous.")
else:
    print("This email appears safe.")