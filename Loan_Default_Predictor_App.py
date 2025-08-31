import pandas as pd
import streamlit as st
import joblib

# Load saved ML model
model = joblib.load("loan_default_predictor.pkl")

# --- Page Config ---
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #f0f4ff, #dff6f0);
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            color: #1a1a1a;
            font-size: 38px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #444;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 40px;
            font-size: 14px;
        }
        .footer a {
            color: #1a73e8;
            text-decoration: none;
            margin: 0 8px;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 15px;
            background: #ffffff;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header with Logo ---
st.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=100)  # Add your logo URL or local path
st.markdown('<p class="title">Loan Default Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict loan repayment risk with AI 🔮</p>', unsafe_allow_html=True)

# --- Sidebar for Inputs ---
st.sidebar.header("📋 Loan & Client Details")
loanamount = st.sidebar.number_input("💵 Loan Amount", 1000, 10000000, 100000)
termdays = st.sidebar.selectbox("📅 Loan Length (days)", [15, 30, 60, 90])
interest_rate = st.sidebar.number_input("📈 Interest Rate (%)", 0.00, 100.00, 10.15)
bank_account_type = st.sidebar.selectbox("🏦 Account Type", ["Current", "Savings", "Other", "Unknown"])
bank_name_clients = st.sidebar.selectbox("🏛 Bank Name", [
    'GT Bank', 'Sterling Bank', 'Fidelity Bank','Access Bank', 'Eco Bank', 'FCMB', 'Skye Bank', 
    'UBA', 'Zenith Bank','Diamond Bank', 'First Bank', 'Union Bank', 'Stanbic IBTC',
    'Standard Chartered', 'Heritage Bank', 'Keystone Bank','Unity Bank', 'Wema Bank', 'Unknown'
])
employment_status_clients = st.sidebar.selectbox("👔 Employment Status", [
    'Retired', 'Permanent', 'Contract', 'Self-Employed','Unemployed','Student', 'Unknown'
])
num_prev_loans = st.sidebar.number_input("🔄 Previous Loans", 0, 100, 5)
avg_termdays = st.sidebar.slider("📊 Avg Previous Loan Length (days)", 0, 100, 30)
avg_prev_delay_days = st.sidebar.slider("⏳ Avg Repay Delay (days)", -50, 150, 10)
age = st.sidebar.slider("🎂 Age", 18, 100, 30)

# Compute Loan Term
loan_term = "Short Term" if termdays <= 30 else "Long Term"

# --- Predict Button ---
if st.button("🔍 Predict Loan Default Risk"):
    data = {
        "loanamount": [loanamount],
        "termdays": [termdays],
        "interest_rate(%)": [interest_rate],
        "bank_account_type": [bank_account_type],
        "bank_name_clients": [bank_name_clients],
        "employment_status_clients": [employment_status_clients],
        "num_prev_loans": [num_prev_loans],
        "avg_termdays": [avg_termdays],
        "avg_prev_delay_days": [avg_prev_delay_days],
        "age": [age],
        "loan_term": [loan_term]
    }

    df = pd.DataFrame(data)

    # Prediction
    loan_default = model.predict(df)
    risk = "Good" if loan_default[0] == 1 else "Bad"

    # Display result with style
    if risk == "Good":
        st.markdown(
            """
            <div class="prediction-box">
                <h2 style="color:green;">✅ Safe Loan</h2>
                <p style="font-size:18px; color:#2e7d32; margin-top:10px;">
                    The borrower is likely to repay.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="prediction-box">
                <h2 style="color:red;">⚠️ Risky Loan</h2>
                <p style="font-size:18px; color:#c62828; margin-top:10px;">
                    The borrower is likely to default!
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
 
        

# --- Footer ---
st.markdown(
    """
    <div class="footer">
        Made with ❤️ using Streamlit <br>
        📧 <a href="mailto:ezeakakingsley@gmail.com">ezeakakingsley@gmail.com</a> | 
        💼 <a href="https://www.linkedin.com/in/yourprofile" target="_blank">LinkedIn</a> | 
        🌐 <a href="https://github.com/yourgithub" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
