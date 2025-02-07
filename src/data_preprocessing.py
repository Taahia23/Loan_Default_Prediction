# steps 
# 1. Load the dataset
# 2. EDA
# 3. Feature Engineering
# 4. Splitting data into train and test sets
# 5. Preprocessing numerical and categorical columns
# 6. Training and evaluating models
# 7. saving models and pre-processing objects



import pandas as pd
import numpy as np

# 1.  Load the data
data = pd.read_csv('data/Loan_default.csv')

# Drop LoanID column
data.drop(columns=['LoanID'], inplace=True)

print(data.head())

# Exploratory Data Analysis (EDA)
# Develop a dashboard for data visualization
import dtale

d = dtale.show(data)
d.open_browser()

d.kill()



# Categorize the Income variable into different bins (groups). 
# Helps in stratified sampling when splitting the data, ensuring a balanced distribution

data['temp_income_cat'] = pd.cut(data['Income'],
                                 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                 labels=[1,2,3,4,5]
                                 )
# split data
# Stratified sampling ensures the income categories are evenly represented in both sets.
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data,
                                       test_size = 0.2,
                                       random_state=42,
                                       stratify=data['temp_income_cat']
                                       )
# drop income column
# It was only used for stratification and is no longer needed.

train_set.drop('temp_income_cat', axis = 1, inplace = True)
test_set.drop('temp_income_cat', axis = 1, inplace = True)

print(f'Train set shape : {train_set.shape}',
      f'Test set shape : {test_set.shape}')


# save train and test set
# To use them later without needing to re-split the data.
import os
os.makedirs('data', exist_ok=True)

train_set.to_csv('data/train_set.csv', index=False)
test_set.to_csv('data/test_set.csv', index=False)

# Reload Training Data
train_set = pd.read_csv('data/train_set.csv')

print(train_set.dtypes)

# Split Features (X_train) & Target Variable (y_train)
# Machine learning models require separate inputs (features) and outputs (targets).

X_train = train_set.drop('Default', axis=1)
y_train = train_set['Default'].copy()

# Create a Validation Set
# Further split training data into training (80%) and validation (20%) sets. To evaluate model performance before testing.

X_train, X_val , y_train, y_val = train_test_split(X_train, y_train,
                                                   test_size=0.2,
                                                   random_state=42)

print(f'Train set shape:{X_train.shape}',
    f'Validation target shape:{y_val.shape}')


# Identify Numerical & Categorical Columns
# Helps in applying different preprocessing techniques to each type.

numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = X_train.select_dtypes('object').columns.tolist()

print(f'Numeric columns : {numeric_columns}',
      f'categorical columns : {categorical_columns}')



# Preprocessing: Handling Missing Values
# note: StandardScaler --- for numerical columns
#       OrdinalEncoder --- for categorical columns
        
print(f'Before preprocessing , no of missing values: {data.isnull().sum()}')

# Create imputers to handle missing values in numerical (mean) and categorical (most frequent) columns.
from sklearn.impute import SimpleImputer
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Standardize Numerical Features
# Standardization improves model performance by normalizing feature distributions.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[numeric_columns] = numeric_imputer.fit_transform(X_train[numeric_columns])
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])

X_val[numeric_columns] = numeric_imputer.transform(X_val[numeric_columns])
X_val[numeric_columns] = scaler.transform(X_val[numeric_columns])



# Encode Categorical Features
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# Any unknown categories in X_val are replaced with -1 instead of causing an error.

X_train[categorical_columns] = categorical_imputer.fit_transform(X_train[categorical_columns])
X_train[categorical_columns] = encoder.fit_transform(X_train[categorical_columns])

X_val[categorical_columns] = categorical_imputer.transform(X_val[categorical_columns])
X_val[categorical_columns] = encoder.transform(X_val[categorical_columns])


print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)


'''
Note: 
We apply StandardScaler and OrdinalEncoder only to X_train and X_val because they contain features (input data) that need scaling and encoding for the model to understand.
We don’t apply them to y_train and y_val because y is the target (output), which the model predicts. The target should remain in its original form to correctly evaluate predictions.

'''

# Train a Linear Regression Model
from sklearn.linear_model import LinearRegression

lin_regression = LinearRegression()

# fit the model
lin_regression.fit(X_train, y_train)

# predict on validation set
y_predict = lin_regression.predict(X_val)

# evaluate the model
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_val, y_predict)

print(f'RMSE of Linear regression: {rmse}')

'''
summary : 

1. Fitting the model: We use only X_train and y_train because the model learns patterns from the training data. Other data is not used to avoid overfitting.

2. Predicting on validation set: We use X_val to check how well the trained model performs on unseen data before testing.

3. Evaluating the model: We compare y_val (actual values) with y_predict (predicted values) to measure the model's accuracy. Using other data wouldn't make sense for validation.

'''


# Train a random forest model
from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(n_estimators=120, random_state=42)

# fit the model
random_forest.fit(X_train, y_train)

# predict on validation set
y_predict = random_forest.predict(X_val)

# evaluate the model
rmse = root_mean_squared_error(y_val, y_predict)

print(f'RMSE of random forest : {rmse}')


# train a neural network
# when the dataset has complex patterns
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(100,50), activation='relu', solver='adam', random_state=42)
mlp.fit(X_train, y_train)
y_predict = mlp.predict(X_val)
rmse = root_mean_squared_error(y_val, y_predict)
print(f'RMSE of MLP Regressor: {rmse}')


'''
compare: 
highest accuracy : Neural network,
2nd highest accuracy : Random forest,
lowest accuracy : Linear regression,

'''

# Save the Model & Preprocessors

import joblib

os.makedirs('models', exist_ok=True)

joblib.dump(mlp, 'models/mlp_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoder, 'models/encoder.pkl')
joblib.dump(numeric_imputer, 'models/num_imputer.pkl')
joblib.dump(categorical_imputer, 'models/cat_imputer.pkl')

print('As neural network does the best perform ,thats why we only save mlp model for future work')

