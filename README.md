# LoanRisk AI — AI-Based Loan Application Risk Analysis

A complete, ready-to-run web application that uses data analytics and machine
learning to assess loan applicant risk, estimate a CIBIL-style credit score,
predict loan approval/rejection, and give personalized suggestions to improve
creditworthiness — with an interactive analytics dashboard.

## Features

- ✅ **User Registration & Secure Login** — Flask-Login sessions, passwords
  hashed with Werkzeug (never stored in plain text).
- ✅ **Risk Prediction Based on Applicant Data** — a scikit-learn
  RandomForest model trained on income, employment, loan, and credit-history
  features returns a 0–100% risk score.
- ✅ **CIBIL Score & Loan Eligibility Analysis** — a transparent, rule-based
  300–900 score computed from repayment history, income, and existing debt,
  mapped to Excellent / Good / Fair / Poor / Very Poor bands and an
  eligibility verdict.
- ✅ **Loan Approval/Rejection Prediction** — the ML model outputs an
  Approved/Rejected decision with an approval probability.
- ✅ **Personalized Suggestions to Improve Creditworthiness** — rule-based
  tips generated from the applicant's weakest factors (repayment history,
  active loans, income-to-loan ratio, dependents, employment type).
- ✅ **Interactive Dashboard with Data Visualization** — Chart.js doughnut
  and bar charts (approvals vs rejections, risk-level distribution) plus a
  recent-applications table, all scoped to the logged-in user.

## Tech Stack

| Layer     | Technology                                   |
|-----------|-----------------------------------------------|
| Backend   | Python, Flask, Flask-SQLAlchemy, Flask-Login  |
| ML/Data   | scikit-learn (RandomForestClassifier), pandas, numpy |
| Database  | SQLite (file-based, zero setup)               |
| Frontend  | Jinja2 templates, Bootstrap 5, Chart.js, Font Awesome |

## Project Structure

```
loan_risk_app/
├── app.py                 # Main Flask application (routes, models, scoring logic)
├── requirements.txt
├── database.db             # created automatically on first run
├── ml/
│   ├── train_model.py      # generates synthetic data & trains the model
│   ├── model.pkl            # pre-trained RandomForest model (already included)
│   ├── columns.pkl          # feature column order the model expects
│   └── metrics.json         # train/test accuracy report
├── templates/
│   ├── base.html, login.html, register.html
│   ├── predict.html, result.html
│   ├── dashboard.html, history.html
└── static/css/style.css
```

## Setup & Run (local machine)

1. **Extract the zip** and open a terminal in the `loan_risk_app` folder.

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Retrain the ML model** — a trained model is already included,
   so this step is optional. Run it only if you want to regenerate the model
   or tweak the synthetic dataset:
   ```bash
   python ml/train_model.py
   ```

5. **Run the app**
   ```bash
   python app.py
   ```
   This also auto-creates `database.db` with the required tables on first run.

6. Open **http://127.0.0.1:5000** in your browser, register an account, log
   in, and submit a loan application from "New Application".

## How the Risk Model Works

Since public loan-approval datasets don't include real CIBIL scores, this
project trains on a **synthetically generated dataset** (6,000 records) with
realistic relationships between income, credit history, existing debt, and
loan outcome (see `ml/train_model.py`). This keeps the project self-contained
and runnable offline, with no external dataset download required.

- **CIBIL-style score**: `compute_cibil_score()` in `app.py` — a transparent
  weighted formula (repayment history + income − existing loans − dependents),
  clipped to the standard 300–900 range.
- **Risk / approval prediction**: a `RandomForestClassifier` trained on the
  synthetic dataset (~79% test accuracy) predicts approval probability;
  `risk_score = (1 − approval_probability) × 100`.
- **Eligibility band**: derived from the CIBIL score and income-to-loan ratio.
- **Suggestions**: rule-based tips triggered by the applicant's weakest
  factors.

To use a **real dataset** (e.g. Kaggle's Loan Prediction dataset) instead,
replace the synthetic-data block in `ml/train_model.py` with `pd.read_csv(...)`
on your CSV, keeping the same column names, then rerun the script.

## Deploying for Long-Term Use (Render + PostgreSQL)

The app already supports this out of the box — it uses SQLite locally, but
automatically switches to PostgreSQL if a `DATABASE_URL` environment
variable is present (see `app.py`). This means your data won't disappear
between deploys or restarts, unlike plain SQLite on most free hosts.

### Option A — One-click with `render.yaml` (recommended)

1. Push this project to a **GitHub repository**.
2. Go to [render.com](https://render.com) → **New → Blueprint** → connect
   your repo. Render will read `render.yaml` and automatically provision:
   - A free web service running `gunicorn app:app`
   - A free PostgreSQL database, wired up via `DATABASE_URL`
   - A random `SECRET_KEY`
3. Click **Apply** / **Deploy**. After the build finishes you'll get a live
   `https://<your-app>.onrender.com` URL.

### Option B — Manual setup

1. Push to GitHub.
2. On Render: **New → PostgreSQL** → create a free database, copy its
   **Internal Connection String**.
3. **New → Web Service** → connect your repo:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
   - Add environment variables: `DATABASE_URL` (paste the connection
     string) and `SECRET_KEY` (any random string).
4. Deploy.

### Notes

- Render's **free web service** tier spins down after 15 minutes of
  inactivity; the next request takes ~30–60 seconds to wake up. Your
  **data is unaffected** by this since it now lives in PostgreSQL, not on
  the web service's local disk. Upgrade to a paid instance ($7/mo at time
  of writing) only if you need the app to stay always-on.
- Render's **free PostgreSQL** databases are provisioned for a limited
  period before requiring an upgrade — check Render's current pricing page
  before relying on it long-term.
- Locally, nothing changes: run `python app.py` and it still uses SQLite
  automatically (no `DATABASE_URL` set on your machine).
- Alternative hosts that work the same way (Flask + `DATABASE_URL` +
  gunicorn): Railway, Fly.io, PythonAnywhere (paid tier for Postgres).

## Security Notes for Production

- Change `SECRET_KEY` in `app.py` (or set it via the `SECRET_KEY` environment
  variable) before deploying.
- Switch `SQLALCHEMY_DATABASE_URI` to a production database (PostgreSQL/MySQL).
- Turn off `debug=True` in `app.run()`.
- Add HTTPS/TLS termination (e.g. via a reverse proxy such as Nginx).

## Customizing

- Add/remove input fields in `templates/predict.html`, `app.py` (`predict()`
  route and `Prediction` model), and retrain `ml/train_model.py` with matching
  columns.
- Adjust CIBIL scoring weights in `compute_cibil_score()`.
- Adjust suggestion rules in `generate_suggestions()`.
