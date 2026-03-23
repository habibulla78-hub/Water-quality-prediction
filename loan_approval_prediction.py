# Loan Approval Prediction using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'Income': [25000, 40000, 50000, 30000, 60000, 20000, 70000, 35000],
    'Loan_Amount': [100000, 150000, 200000, 120000, 250000, 90000, 300000, 130000],
    'Credit_Score': [650, 700, 750, 680, 800, 600, 820, 690],
    'Approved': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert labels to numbers
df['Approved'] = df['Approved'].map({'Yes': 1, 'No': 0})

# Features and target
X = df[['Income', 'Loan_Amount', 'Credit_Score']]
y = df['Approved']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Test new data
new_data = [[45000, 150000, 720]]
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Not Approved")
