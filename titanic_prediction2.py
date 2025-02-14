import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
from flask import Flask, jsonify, request, render_template
#loading titanic dataset
data = pd.read_csv('train.csv')
#preprocessing the dataset
def preprocess_data(ourdata):
    ourdata = ourdata.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
    ourdata['Sex'] = ourdata['Sex'].map({'male': 0, 'female': 1})
    ourdata['Embarked'] = ourdata['Embarked'].map({'C':0, 'Q':1, 'S':2})
    ourdata['Embarked'] = ourdata['Embarked'].fillna(ourdata['Embarked'].mode()[0])
    ourdata['Age'] = ourdata['Age'].fillna(ourdata['Age'].median())
    ourdata['Fare'] = ourdata['Fare'].fillna(ourdata['Fare'].median())
    return ourdata
data = preprocess_data(data)
# splitting data
X = data.drop('Survived', axis =1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train the model
model = DecisionTreeClassifier(max_depth = 5, random_state=42)
model.fit(X_train, y_train)
#Evaluate the model
y_pred = model.predict(X_test)
print(f'model accuracy: {accuracy_score(y_test, y_pred):.2f}')
# save the model
with open('titanic_model.pkl', 'wb') as file:
    pickle.dump(model, file)
# Flask app
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods = ['POST'])
def predict():
    #extract input data
    data = request.form
    features = [
        int(data['Pclass']),
        int(data['Sex']),
        float(data['Age']),
        int(data['sbSp']),
        int(data['Parch']),
        float(data['Fare']),
        int(data['Embarked']),
    ]
    # Load Model and Predict
    with open('titanic_model.pkl', 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict([features])[0]
    #respond with result
    result = 'survived' if prediction ==1 else 'Did Not Survive'
    return jsonify({'Prediction': result})
if __name__ == '__main__':
    app.run(debug= True)    