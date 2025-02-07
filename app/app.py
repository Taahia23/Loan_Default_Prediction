import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='Loan Default Predictor', layout='wide')


st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Loan Default Prediction App</h1>
    <h4 style='text-align: center; color: #777;'>Predict the likelihood of a loan default using machine learning</h4>
""", unsafe_allow_html=True)

st.sidebar.header('Enter Loan Details')

st.sidebar.subheader("Personal Information")
Age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
Income = st.sidebar.number_input('Income ($)', min_value=0.0, value=50000.0, step=1000.0)
CreditScore = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=650)
MonthsEmployed = st.sidebar.number_input('Months Employed', min_value=0, value=12)
NumCreditLines = st.sidebar.number_input('Number of Credit Lines', min_value=0, value=3)

st.sidebar.subheader("Loan Details")
LoanAmount = st.sidebar.number_input('Loan Amount ($)', min_value=1000.0, value=10000.0, step=500.0)
InterestRate = st.sidebar.number_input('Interest Rate (%)', min_value=0.1, value=5.0, step=0.1)
LoanTerm = st.sidebar.number_input('Loan Term (months)', min_value=6, value=36)
DTIRatio = st.sidebar.number_input('Debt-to-Income Ratio', min_value=0.0, max_value=1.0, value=0.2, step=0.01)

st.sidebar.subheader("Other Factors")
Education = st.sidebar.selectbox('Education Level', ['Bachelors', 'Masters', 'High School', 'PhD'])
EmploymentType = st.sidebar.selectbox('Employment Type', ['Full-time', 'Unemployed', 'Self-employed', 'Part-time'])
MaritalStatus = st.sidebar.selectbox('Marital Status', ['Married', 'Divorced', 'Single'])
HasMortgage = st.sidebar.selectbox('Has Mortgage?', ['Yes', 'No'])
HasDependents = st.sidebar.selectbox('Has Dependents?', ['Yes', 'No'])
LoanPurpose = st.sidebar.selectbox('Loan Purpose', ['Other', 'Auto', 'Business', 'Home', 'Education'])
HasCoSigner = st.sidebar.selectbox('Has Co-Signer?', ['Yes', 'No'])

input_data = {
    'Age': Age, 'Income': Income, 'LoanAmount': LoanAmount, 'CreditScore': CreditScore,
    'MonthsEmployed': MonthsEmployed, 'NumCreditLines': NumCreditLines, 'InterestRate': InterestRate,
    'LoanTerm': LoanTerm, 'DTIRatio': DTIRatio, 'Education': Education, 'EmploymentType': EmploymentType,
    'MaritalStatus': MaritalStatus, 'HasMortgage': HasMortgage, 'HasDependents': HasDependents,
    'LoanPurpose': LoanPurpose, 'HasCoSigner': HasCoSigner
}

input_df = pd.DataFrame([input_data])

model = joblib.load('models/final_ensemble_model.pkl')

st.markdown(
    """
    <style>
        div.stButton > button {
            background-color: #4CAF50;
            color: white !important;
            font-size: 16px;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #45a049 !important;
            color: white;
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("Predict Loan Default", use_container_width=True):
    result = model.predict(input_df)
    st.dataframe(input_df.style.format({"Income": "${:,.2f}", 
                                        "LoanAmount": "${:,.2f}"}), 
                                        height=50)

    st.metric('Predicted Loan Default:', f'{result[0]:,.2f}')

    