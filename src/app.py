import joblib
import pandas as pd
sample_data = {
    'Age' : 28,
    'Income' : 3713,
    'LoanAmount' : 230883,
    'CreditScore' : 531,
    'MonthsEmployed' : 67,
    'NumCreditLines' : 4,
    'InterestRate' : 8.2,
    'LoanTerm' : 60,
    'DTIRatio' : 0.43,
    'Education' : 'PhD',
    'EmploymentType' : 'Full-time',
    'MaritalStatus' : 'Married',
    'HasMortgage' : 'No', 
    'HasDependents' : 'No',
    'LoanPurpose' : 'Home',
    'HasCoSigner' : 'Yes',
    'Default' : 0
}
sample_data_df = pd.DataFrame([sample_data])
model = joblib.load('models/final_ensemble_model.pkl')
result = model.predict(sample_data_df)
print(f'result : {result[0]}')