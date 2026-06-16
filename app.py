from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model (Random Forest, no scaler needed)
model = joblib.load('rand_model.joblib')

# Hard‑coded feature columns in EXACT order the model was trained on
FEATURE_COLUMNS = [
    'credit_score', 'gender', 'age', 'tenure', 'balance',
    'products_number', 'credit_card', 'active_member', 'estimated_salary',
    'country_France', 'country_Germany', 'country_Spain'
]

def preprocess_input(input_dict):
    """
    Convert raw form inputs into a DataFrame row matching the training columns.
    No scaling, no pd.get_dummies – we manually build the row.
    """
    # Build a single-row DataFrame
    row = {
        'credit_score': float(input_dict['credit_score']),
        'gender': 1 if input_dict['gender'] == 'Male' else 0,   # 0 = Female, 1 = Male (adjust if needed)
        'age': float(input_dict['age']),
        'tenure': float(input_dict['tenure']),
        'balance': float(input_dict['balance']),
        'products_number': int(input_dict['products_number']),
        'credit_card': int(input_dict['credit_card']),
        'active_member': int(input_dict['active_member']),
        'estimated_salary': float(input_dict['estimated_salary']),
        # Geography dummies – exactly one will be 1, others 0
        'country_France': 1 if input_dict['country'] == 'France' else 0,
        'country_Germany': 1 if input_dict['country'] == 'Germany' else 0,
        'country_Spain': 1 if input_dict['country'] == 'Spain' else 0
    }

    df = pd.DataFrame([row])

    # Enforce the exact column order (just to be safe)
    df = df[FEATURE_COLUMNS]

    return df

# ---------- Home page: show the form ----------
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction=None)

# ---------- Handle form submission ----------
@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'credit_score': request.form['creditscore'],
        'gender': request.form['gender'],
        'age': request.form['age'],
        'tenure': request.form['tenure'],
        'balance': request.form['balance'],
        'products_number': request.form['products_number'],
        'credit_card': request.form['credit_card'],
        'active_member': request.form['active_member'],
        'estimated_salary': request.form['estimatedsalary'],
        'country': request.form['country']
    }

    processed = preprocess_input(input_data)

    # Predict churn probability (second column = churn class)
    proba = model.predict_proba(processed)[0, 1]
    risk = round(proba * 100, 1)

    # Risk level and colour
    if proba < 0.3:
        risk_level = 'Low'
        color = 'green'
    elif proba < 0.7:
        risk_level = 'Medium'
        color = 'orange'
    else:
        risk_level = 'High'
        color = 'red'

    # Global feature importance as a rough explanation
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    top_factors = importances.sort_values(ascending=False).head(3).index.tolist()

    return render_template('index.html',
                           prediction=True,
                           risk=risk,
                           risk_level=risk_level,
                           color=color,
                           top_factors=top_factors)

if __name__ == '__main__':
    app.run(debug=True)