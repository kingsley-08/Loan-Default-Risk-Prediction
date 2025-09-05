import pandas as pd
import streamlit as st
import joblib
import time

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
            background: linear-gradient(135deg, #e0f7fa, #f1f8e9);
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            color: #0d47a1;
            font-size: 40px;
            font-weight: bold;
            margin-top: 10px;
        }
        .subtitle {
            text-align: center;
            color: #444;
            font-size: 18px;
            margin-bottom: 30px;
        }
        .hero-img {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
        }
        .prediction-box {
            padding: 25px;
            border-radius: 18px;
            background: #ffffff;
            box-shadow: 0px 6px 18px rgba(0,0,0,0.1);
            text-align: center;
            margin-top: 25px;
            transition: all 0.3s ease-in-out;
        }
        .prediction-box:hover {
            transform: scale(1.02);
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 50px;
            font-size: 14px;
        }
        .footer a {
            color: #1a73e8;
            text-decoration: none;
            margin: 0 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
                                        
# --- Header with Logo ---
st.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=100)
st.markdown('<p class="title">Loan Default Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict loan repayment risk with AI 🔮</p>', unsafe_allow_html=True)


# --- Sidebar for Inputs ---
st.sidebar.header("📋 Loan & Client Details")

with st.sidebar.expander("💵 Loan Details", expanded=True):
    loanamount = st.number_input("Loan Amount", 1000, 10000000, 100000)
    termdays = st.selectbox("Loan Length (days)", [15, 30, 60, 90])
    interest_rate = st.number_input("Interest Rate (%)", 0.00, 100.00, 10.15)

with st.sidebar.expander("🏦 Banking Info", expanded=True):
    bank_account_type = st.selectbox("Account Type", ["Current", "Savings", "Other", "Unknown"])
    bank_name_clients = st.selectbox("Bank Name", [
        'GT Bank', 'Sterling Bank', 'Fidelity Bank','Access Bank', 'Eco Bank', 'FCMB', 'Skye Bank', 
        'UBA', 'Zenith Bank','Diamond Bank', 'First Bank', 'Union Bank', 'Stanbic IBTC',
        'Standard Chartered', 'Heritage Bank', 'Keystone Bank','Unity Bank', 'Wema Bank', 'Unknown'
    ])

with st.sidebar.expander("👤 Client Info", expanded=True):
    employment_status_clients = st.selectbox("Employment Status", [
        'Retired', 'Permanent', 'Contract', 'Self-Employed','Unemployed','Student', 'Unknown'
    ])
    age = st.slider("Age", 18, 100, 30)

with st.sidebar.expander("📊 Previous Loans", expanded=True):
    num_prev_loans = st.number_input("Previous Loans", 0, 100, 5)
    avg_termdays = st.slider("Avg Previous Loan Length (days)", 0, 100, 30)
    avg_prev_delay_days = st.slider("Avg Repay Delay (days)", -50, 150, 10)

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

    # Progress animation
    progress_text = "⏳ Analyzing loan risk..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.02)
        my_bar.progress(percent_complete + 1, text=progress_text)
    my_bar.empty()

    # Prediction
    loan_default = model.predict(df)
    prob = model.predict_proba(df)[0][loan_default[0]] * 100  # confidence %
    risk = "Good" if loan_default[0] == 1 else "Bad"

    # Display result
    if risk == "Good":
        st.markdown(
            f"""
            <div class="prediction-box">
                <h2 style="color:green;">✅ Safe Loan</h2>
                <p style="font-size:18px; color:#2e7d32; margin-top:10px;">
                    The borrower is likely to repay.<br>
                    <b>Confidence: {prob:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="prediction-box">
                <h2 style="color:red;">⚠️ Risky Loan</h2>
                <p style="font-size:18px; color:#c62828; margin-top:10px;">
                    The borrower is likely to default!<br>
                    <b>Confidence: {prob:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Footer ---
st.markdown(
    """
    <div class="footer">
        Made with ❤️ using <b>Streamlit</b> <br>
        📧 <a href="mailto:ezeakakingsley@gmail.com">ezeakakingsley@gmail.com</a> | 
        💼 <a href="https://www.linkedin.com/in/kingsley-ezeaka-a7024a153" target="_blank">LinkedIn/kingsley-ezeaka</a> | 
        🌐 <a href="https://github.com/kingsley-08" target="_blank">GitHub/kingsley-08</a>
    </div>
    """,
    unsafe_allow_html=True
)
