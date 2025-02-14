import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
from flask import Flask, request, jsonify, render_template
# Load Titanic dataset
data = pd.read_csv('train.csv')
# Preprocessing
def preprocess_data(data):
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    return data
data = preprocess_data(data)
# Splitting data
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train decision tree classifier
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
# Save the model
with open('titanic_model.pkl', 'wb') as file:
    pickle.dump(model, file)
# Flask app
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data
    data = request.form
    features = [
        int(data['Pclass']),
        int(data['Sex']),
        float(data['Age']),
        int(data['SibSp']),
        int(data['Parch']),
        float(data['Fare']),
        int(data['Embarked'])
    ]
    # Load model and predict
    with open('titanic_model.pkl', 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict([features])[0]
    # Respond with the result
    result = 'Survived' if prediction == 1 else 'Did Not Survive'
    return jsonify({'Prediction': result})
if __name__ == '__main__':
    app.run(debug=True)