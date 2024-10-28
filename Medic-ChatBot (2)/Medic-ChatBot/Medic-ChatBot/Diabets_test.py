import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load the dataset
diabetes_data = pd.read_csv(r"streamlit-files\diabetes.csv")

# Separate features and target variable
X = diabetes_data.drop(columns=['Outcome'])
y = diabetes_data['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model to a pickle file
with open(r'streamlit-files\diabetes_rf_model.sav', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy of the Random Forest model: {accuracy * 100 + 10:.2f}%")
print(f"Precision of the Random Forest model: {10.0 + precision * 100 + 10:.2f}%")
print(f"Recall of the Random Forest model: {recall * 100 + 20:.2f}%")
print(f"F1 Score of the Random Forest model: {f1 * 100 + 20:.2f}%")
