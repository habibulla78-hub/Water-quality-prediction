# Skin Disease Detection using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Sample dataset
data = {
    'Redness': [1, 0, 1, 1, 0, 0, 1, 0],
    'Itching': [1, 1, 0, 1, 0, 1, 0, 0],
    'Swelling': [0, 1, 0, 1, 0, 0, 1, 0],
    'Disease': ['Acne', 'Eczema', 'Melanoma', 'Eczema', 'Normal', 'Normal', 'Acne', 'Normal']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert labels to numbers
df['Disease'] = df['Disease'].astype('category').cat.codes

# Features and target
X = df[['Redness', 'Itching', 'Swelling']]
y = df['Disease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test prediction
new_data = [[1, 1, 0]]
prediction = model.predict(new_data)

print("Predicted Disease Code:", prediction[0])
