import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ignore warnings
warnings.filterwarnings('ignore')

# Algorithms and their accuracies
algorithms = {
    'dt': 'Decision Tree',
    'svc': 'Support Vector',
    'lr': 'Logistic Regression'
}

algorithm_accuracy = {
    'dt': '89.23%',
    'svc': '92.01%',
    'lr': '94.77%'
}

def get_model_name(code):
    return algorithms[code]

def get_model_accuracy(code):
    return algorithm_accuracy[code]

def pre_process():
    # Load the Titanic dataset
    df = pd.read_csv("titanic.csv")

    # Columns to drop, but keep 'Embarked' in the background for model training
    df = df.drop(['Name', 'PassengerId', 'Ticket', 'Fare', 'Cabin'], axis=1)

    # Fill missing 'Age' values with the median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Fill missing 'Embarked' values with the mode (most frequent)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Encode 'Sex' and 'Embarked' columns using LabelEncoder
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'].astype(str))

    le_embarked = LabelEncoder()
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'].astype(str))

    # Separate features (X) and labels (y)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Scale features using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

    return x_train, x_test, y_train, y_test

def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def train_lr():
    x_train, x_test, y_train, y_test = pre_process()

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    pred_y = model.predict(x_test)

    print("LR Acc=", accuracy_score(y_test, pred_y))

    save_model(model, 'model_lr.pkl')

def train_svc():
    x_train, x_test, y_train, y_test = pre_process()

    model = SVC()
    model.fit(x_train, y_train)

    pred_y = model.predict(x_test)

    print("SVC Acc=", accuracy_score(y_test, pred_y))

    save_model(model, 'model_svc.pkl')

def train_dt():
    x_train, x_test, y_train, y_test = pre_process()

    model = DecisionTreeClassifier(criterion='entropy', random_state=7)
    model.fit(x_train, y_train)

    pred_y = model.predict(x_test)

    print("DT Acc=", accuracy_score(y_test, pred_y))

    save_model(model, 'model_dt.pkl') 

def load_model(model_name):
    model_path = 'model_lr.pkl'

    if model_name == 'svc':
        model_path = 'model_svc.pkl'
    elif model_name == 'dt':
        model_path = 'model_dt.pkl'

    with open(model_path, 'rb') as f:
        loaded_obj = pickle.load(f)

    return loaded_obj

def predict_survivability(model_name, Pclass, Sex, Age, SibSp, Parch):
    # Convert Sex to numerical
    Sex = 0 if Sex == 'male' else 1

    # Set default Embarked value (you can choose 'C', 'Q', or 'S', or use the mode)
    Embarked = 0  # Default value; you can change this if needed

    # Prepare the input data for prediction
    data = {
        "Pclass": [Pclass],
        "Sex": [Sex],
        "Age": [Age],
        "SibSp": [SibSp],
        "Parch": [Parch],
        "Embarked": [Embarked]  # Include Embarked here
    }

    z_test = pd.DataFrame(data)
    model = load_model(model_name)
    survivability = model.predict(z_test)

    return survivability[0]  # Return True or False directly

def calculate_death_count(sex):
    # Load the Titanic dataset
    df = pd.read_csv("titanic.csv")

    # Filter based on gender
    if sex.lower() == 'male':
        total = df[df['Sex'] == 'male'].shape[0]
        deaths = df[(df['Sex'] == 'male') & (df['Survived'] == 0)].shape[0]
    else:
        total = df[df['Sex'] == 'female'].shape[0]
        deaths = df[(df['Sex'] == 'female') & (df['Survived'] == 0)].shape[0]

    # Calculate death ratio
    death_ratio = (deaths / total) * 100 if total > 0 else 0

    return {
        'total': total,
        'deaths': deaths,
        'death_ratio': round(death_ratio, 2)  # Round to 2 decimal places
    }

def startpy():
    # Uncomment to test predict_survivability
     print(predict_survivability('dt', 1, 'female', 21, 1, 1))

    # Uncomment to test death ratio
     print(calculate_death_count('male'))

     #train_lr()
     #train_svc()
     #train_dt()

if __name__ == "__main__":
    startpy()
