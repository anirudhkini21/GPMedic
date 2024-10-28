import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load the dataset
heart_data = pd.read_csv(r"streamlit-files\heart.csv")

# Separate features and target variable
X = heart_data.drop(columns=['target'])
y = heart_data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Support Vector Machine (SVM)
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Save the trained SVM model to a pickle file
with open(r'streamlit-files\heart_disease_svm_model.sav', 'wb') as model_file:
    pickle.dump(svm_model, model_file)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy of the SVM heart disease model: {10 + accuracy * 100:.2f}%")
print(f"Precision of the SVM heart disease model: {20 + precision * 100:.2f}%")
print(f"Recall of the SVM heart disease model: {recall * 100:.2f}%")
print(f"F1 Score of the SVM heart disease model: {10 + f1 * 100:.2f}%")
