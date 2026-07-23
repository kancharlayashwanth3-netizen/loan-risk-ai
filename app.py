import os
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# App & extensions setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')

# Use PostgreSQL when a DATABASE_URL is provided (production / Render),
# otherwise fall back to a local SQLite file for easy local development.
database_url = os.environ.get('DATABASE_URL')
if database_url:
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql+psycopg://', 1)
    elif database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+psycopg://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'database.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True}

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to continue.'
login_manager.login_message_category = 'warning'

# ---------------------------------------------------------------------------
# Load trained ML artifacts
# ---------------------------------------------------------------------------
model = joblib.load(os.path.join(BASE_DIR, 'ml', 'model.pkl'))
feature_columns = joblib.load(os.path.join(BASE_DIR, 'ml', 'columns.pkl'))


# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    applicant_name = db.Column(db.String(120))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    married = db.Column(db.String(10))
    dependents = db.Column(db.Integer)
    education = db.Column(db.String(20))
    self_employed = db.Column(db.String(10))
    applicant_income = db.Column(db.Float)
    coapplicant_income = db.Column(db.Float)
    loan_amount = db.Column(db.Float)
    loan_term = db.Column(db.Integer)
    existing_loans = db.Column(db.Integer)
    credit_history_label = db.Column(db.String(10))
    property_area = db.Column(db.String(20))

    cibil_score = db.Column(db.Integer)
    risk_score = db.Column(db.Float)        # 0-100, higher = riskier
    risk_level = db.Column(db.String(20))   # Low / Medium / High
    prediction = db.Column(db.String(20))   # Approved / Rejected
    eligibility = db.Column(db.String(30))  # Eligibility band
    suggestions = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
HISTORY_MAP = {'Good': 300, 'Average': 150, 'Poor': 50}


def compute_cibil_score(credit_history_label, existing_loans, applicant_income,
                         coapplicant_income, dependents):
    total_income = applicant_income + coapplicant_income
    score = 300
    score += HISTORY_MAP.get(credit_history_label, 100)
    score += min(total_income / 1000 * 2, 200)
    score -= existing_loans * 25
    score -= dependents * 5
    return int(max(300, min(900, score)))


def cibil_band(score):
    if score >= 750:
        return 'Excellent'
    elif score >= 700:
        return 'Good'
    elif score >= 650:
        return 'Fair'
    elif score >= 600:
        return 'Poor'
    else:
        return 'Very Poor'


def eligibility_from_score(cibil_score, income_to_loan_ratio):
    if cibil_score >= 750 and income_to_loan_ratio >= 1.5:
        return 'Highly Eligible'
    elif cibil_score >= 650:
        return 'Eligible'
    elif cibil_score >= 550:
        return 'Conditionally Eligible'
    else:
        return 'Not Eligible'


def build_model_input(data):
    row = {
        'Dependents': data['dependents'],
        'ApplicantIncome': data['applicant_income'],
        'CoapplicantIncome': data['coapplicant_income'],
        'LoanAmount': data['loan_amount'],
        'Loan_Amount_Term': data['loan_term'],
        'Credit_History': 1 if data['credit_history_label'] in ('Good', 'Average') else 0,
        'ExistingLoans': data['existing_loans'],
        'CibilScore': data['cibil_score'],
        f"Gender_{data['gender']}": 1,
        f"Married_{data['married']}": 1,
        f"Education_{data['education']}": 1,
        f"Self_Employed_{data['self_employed']}": 1,
        f"Property_Area_{data['property_area']}": 1,
    }
    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def generate_suggestions(data, cibil_score):
    tips = []
    if data['credit_history_label'] == 'Poor':
        tips.append('Build a consistent on-time repayment record for at least 6-12 months to improve your credit history.')
    elif data['credit_history_label'] == 'Average':
        tips.append('Avoid missed or late payments going forward; a longer clean repayment streak will boost your score.')

    if data['existing_loans'] >= 3:
        tips.append('Reduce the number of active loans/credit lines before applying — too many open accounts increases perceived risk.')

    total_income = data['applicant_income'] + data['coapplicant_income']
    if data['loan_amount'] > 0 and total_income * (data['loan_term'] / 12) / data['loan_amount'] < 1.0:
        tips.append('Consider requesting a smaller loan amount or adding a co-applicant to improve your income-to-loan ratio.')

    if data['dependents'] >= 3:
        tips.append('A high number of dependents raises perceived financial burden — highlight additional income sources if available.')

    if data['self_employed'] == 'Yes':
        tips.append('Maintain clear, consistent income documentation (ITRs/bank statements) for at least the last 2 years.')

    if cibil_score < 650:
        tips.append('Keep credit utilization below 30% of your available limit and avoid applying for multiple loans at once.')

    if not tips:
        tips.append('Your profile looks strong — maintain your current repayment discipline and credit utilization habits.')

    return tips


def run_prediction(data):
    cibil_score = compute_cibil_score(
        data['credit_history_label'], data['existing_loans'],
        data['applicant_income'], data['coapplicant_income'], data['dependents']
    )
    data['cibil_score'] = cibil_score

    X = build_model_input(data)
    proba_approved = float(model.predict_proba(X)[0][1])
    risk_score = round((1 - proba_approved) * 100, 1)

    if risk_score < 30:
        risk_level = 'Low'
    elif risk_score < 60:
        risk_level = 'Medium'
    else:
        risk_level = 'High'

    prediction = 'Approved' if proba_approved >= 0.5 else 'Rejected'

    total_income = data['applicant_income'] + data['coapplicant_income']
    income_to_loan_ratio = (total_income * (data['loan_term'] / 12)) / (data['loan_amount'] + 1)
    eligibility = eligibility_from_score(cibil_score, income_to_loan_ratio)

    suggestions = generate_suggestions(data, cibil_score)

    return {
        'cibil_score': cibil_score,
        'cibil_band': cibil_band(cibil_score),
        'risk_score': risk_score,
        'risk_level': risk_level,
        'prediction': prediction,
        'approval_probability': round(proba_approved * 100, 1),
        'eligibility': eligibility,
        'suggestions': suggestions,
    }


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        if not username or not email or not password:
            flash('All fields are required.', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
        elif password != confirm:
            flash('Passwords do not match.', 'danger')
        elif User.query.filter_by(username=username).first():
            flash('Username already taken.', 'danger')
        elif User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
        else:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        identifier = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter(
            (User.username == identifier) | (User.email == identifier.lower())
        ).first()

        if user and user.check_password(password):
            login_user(user)
            flash(f'Welcome back, {user.username}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username/email or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ---------------------------------------------------------------------------
# Core app routes
# ---------------------------------------------------------------------------
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            data = {
                'applicant_name': request.form.get('applicant_name', '').strip() or 'Applicant',
                'age': int(request.form['age']),
                'gender': request.form['gender'],
                'married': request.form['married'],
                'dependents': int(request.form['dependents']),
                'education': request.form['education'],
                'self_employed': request.form['self_employed'],
                'applicant_income': float(request.form['applicant_income']),
                'coapplicant_income': float(request.form.get('coapplicant_income') or 0),
                'loan_amount': float(request.form['loan_amount']),
                'loan_term': int(request.form['loan_term']),
                'existing_loans': int(request.form['existing_loans']),
                'credit_history_label': request.form['credit_history_label'],
                'property_area': request.form['property_area'],
            }
        except (KeyError, ValueError):
            flash('Please fill in all fields with valid values.', 'danger')
            return redirect(url_for('predict'))

        result = run_prediction(data)

        record = Prediction(
            user_id=current_user.id,
            applicant_name=data['applicant_name'],
            age=data['age'],
            gender=data['gender'],
            married=data['married'],
            dependents=data['dependents'],
            education=data['education'],
            self_employed=data['self_employed'],
            applicant_income=data['applicant_income'],
            coapplicant_income=data['coapplicant_income'],
            loan_amount=data['loan_amount'],
            loan_term=data['loan_term'],
            existing_loans=data['existing_loans'],
            credit_history_label=data['credit_history_label'],
            property_area=data['property_area'],
            cibil_score=result['cibil_score'],
            risk_score=result['risk_score'],
            risk_level=result['risk_level'],
            prediction=result['prediction'],
            eligibility=result['eligibility'],
            suggestions='\n'.join(result['suggestions']),
        )
        db.session.add(record)
        db.session.commit()

        return redirect(url_for('result', prediction_id=record.id))

    return render_template('predict.html')


@app.route('/result/<int:prediction_id>')
@login_required
def result(prediction_id):
    record = Prediction.query.get_or_404(prediction_id)
    if record.user_id != current_user.id:
        flash('You do not have access to that record.', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('result.html', r=record, suggestions=record.suggestions.split('\n'))


@app.route('/dashboard')
@login_required
def dashboard():
    records = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()

    total = len(records)
    approved = sum(1 for r in records if r.prediction == 'Approved')
    rejected = total - approved
    avg_cibil = round(sum(r.cibil_score for r in records) / total, 0) if total else 0
    avg_risk = round(sum(r.risk_score for r in records) / total, 1) if total else 0

    risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
    for r in records:
        risk_counts[r.risk_level] = risk_counts.get(r.risk_level, 0) + 1

    stats = {
        'total': total,
        'approved': approved,
        'rejected': rejected,
        'avg_cibil': avg_cibil,
        'avg_risk': avg_risk,
        'risk_low': risk_counts['Low'],
        'risk_medium': risk_counts['Medium'],
        'risk_high': risk_counts['High'],
    }

    return render_template('dashboard.html', records=records[:10], stats=stats)


@app.route('/history')
@login_required
def history():
    records = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('history.html', records=records)


@app.route('/delete/<int:prediction_id>', methods=['POST'])
@login_required
def delete_record(prediction_id):
    record = Prediction.query.get_or_404(prediction_id)
    if record.user_id == current_user.id:
        db.session.delete(record)
        db.session.commit()
        flash('Record deleted.', 'info')
    return redirect(url_for('history'))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def init_db():
    with app.app_context():
        db.create_all()


# Create tables on import so this also works when run under gunicorn
# (gunicorn imports this module and calls `app` directly — it never hits
# the __main__ block below).
init_db()


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
