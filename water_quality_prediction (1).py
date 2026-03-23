# Water Quality Prediction using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'pH': [7, 6.5, 8, 5.5, 7.5, 6, 8.5, 5],
    'Turbidity': [3, 5, 2, 7, 3, 6, 2, 8],
    'Dissolved_Oxygen': [8, 6, 9, 5, 7, 6, 9, 4],
    'Quality': ['Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert output labels to numbers
df['Quality'] = df['Quality'].map({'Good': 1, 'Bad': 0})

# Features and target
X = df[['pH', 'Turbidity', 'Dissolved_Oxygen']]
y = df['Quality']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Test new data
new_data = [[7.2, 3, 8]]
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Water Quality: Good")
else:
    print("Water Quality: Bad")
