import joblib
import pandas as pd

#  Load the Saved Model
mlp_model = joblib.load('models/final_ensemble_model.pkl')

#  Load the Test Data
test_data = pd.read_csv('data/test_set.csv')

# Separate Features (X_test) and Target (y_test)
'''
 X_test contains all input features (independent variables).
 y_test contains the actual target values (dependent variable).
 
'''

X_test = test_data.drop('Default', axis=1)
y_test = test_data['Default'].copy()


#  Make Predictions Using the Model
y_prediction = mlp_model.predict(X_test)


#  Evaluate the Model’s Performance
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_test, y_prediction)
print(f'Root Mean Squared Error: {rmse}')

'''
RMSE measures the error in predictions.
Lower RMSE means better model performance.
'''

#  Display a Random Sample from the Test Data
test_data.sample(1).to_dict()









