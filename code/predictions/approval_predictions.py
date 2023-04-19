import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

def predict_new_data(production=False, save=True):
    loaded_model = tf.keras.models.load_model('./models/loan_prediction_model.h5') # load trained model

    with open('./models/loan_prediction_scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f) # load scalers

    if production:
        data = pd.read_csv('./data/predict_data.csv') # Load and preprocess the new data
    else:
        data = pd.read_csv('./preprocessing/raw_data/training_data.csv') # Load and preprocess the old training data
        data = data.drop(['Loan_Status'], axis=1, errors='ignore') # when testing we already have the csv, but dont need Loan_Status as we are predicing that using the trained model

    label_encoders = {}
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'] # encoding categorical columns

    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        label_encoders[col].fit(data[col].astype(str))

    for col in categorical_columns:
        data[col] = label_encoders[col].transform(data[col].astype(str))

    numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in numerical_columns:
        data[col].fillna(data[col].mean(), inplace=True)

    for col in categorical_columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    X_new = data.drop(['Loan_ID'], axis=1) # Loan id is not needed
    X_new_scaled = loaded_scaler.transform(X_new)
    predictions = loaded_model.predict(X_new_scaled) # get our predictions based on scaled model and {dataset}
    binary_predictions = (predictions > 0.5).astype(int) # interger rounding to get either 1 or 0 i.e. denied or approved

    if save: # default save results to new csv file to see the data and the predicted results || used for input to sg2 model
        data['Predictions'] = binary_predictions
        data['Raw_Predictions'] = predictions # will be used for 
        data.to_csv('./data/predicted_data.csv', index=False)