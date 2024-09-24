import numpy as np
import flask
from flask import Flask, request, render_template
import titanic_business as tib

app = Flask(__name__)

CURRENT_MODEL_CODE = 'svc'

def render_template(page_name, **kwargs):
    return flask.render_template(
        page_name, 
        APP_TITLE="Titanic Prediction",
        **kwargs
    )

@app.route('/')
def home():
    current_model_code  = CURRENT_MODEL_CODE
    current_model_full  = tib.get_model_name(current_model_code)

    return render_template(
        'index.html',
        current_model=current_model_full,
        result=None,  # Ensure result is defined
        death_count=None  # Initialize death_count as None
    )

@app.route('/', methods=['POST'])
def predict():
    rev = request.values
    sex = rev.get('sex')

    # Encode 'Sex'
    sex_encoded = 1 if sex.lower() == 'male' else 0

    current_model_code  = CURRENT_MODEL_CODE
    current_model_full  = tib.get_model_name(current_model_code)
    current_model_acc   = tib.get_model_accuracy(current_model_code)

    # Predict survivability
    survived_result = tib.predict_survivability(
        current_model_code,
        1,               # Example Pclass
        sex_encoded,     # Encoded sex
        30,              # Dummy age
        1,               # Example SibSp
        1                # Example Parch
    )

    survived_result = "Survived" if survived_result else "Not Survived"

    # Calculate death count and death ratio for the given gender
    death_count = tib.calculate_death_count(sex)

    result = {
        'sex': sex,
        'survived': survived_result,
        'model': current_model_full,
        'accuracy': current_model_acc
    }

    return render_template(
        'index.html',
        result=result,
        death_count=death_count,
        current_model=current_model_full
    )


if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        debug=True,
        port=4400
    )
