import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import pickle

# Load data
df = pd.read_csv("Data.csv")
df = df.drop_duplicates()

# Separate features and target
X = df.iloc[:, 0:16]
y = df["fraud_status"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
classifier.fit(X_train, y_train)

# Save the model
model_data = {
    "model": classifier,
    "feature_names": X.columns.to_numpy(),
    "scaler": sc,
}

with open("yashwanth_amrutha_phase3_2.pkl", "wb") as model_file:
    pickle.dump(model_data, model_file)

print("Model saved to yashwanth_amrutha_phase3_2.pkl")

# Predict and evaluate
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1_score * 100:.2f}%")

# Cross-validation
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print("Cross-Validation Scores:", scores)
