import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

def predict_new_data():
    loaded_model = tf.keras.models.load_model('loan_prediction_model.h5')

    # Load the saved StandardScaler
    with open('scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)

    # Load and preprocess the new data
    data = pd.read_csv('./preprocessing/raw_data/training_data.csv') # Load and preprocess the data

    # Create LabelEncoders for categorical columns
    label_encoders = {}

    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'] # Encoding categorical columns

    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        label_encoders[col].fit(data[col].astype(str))

    # Encoding categorical columns in the new data
    for col in categorical_columns:
        data[col] = label_encoders[col].transform(data[col].astype(str))

    # Fill missing values in numerical columns with their mean
    numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in numerical_columns:
        data[col].fillna(data[col].mean(), inplace=True)

    # Fill missing values in categorical columns with their mode
    for col in categorical_columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Extract features from the new data
    X_new = data.drop(['Loan_ID'], axis=1)

    # Scale the numerical features
    X_new_scaled = loaded_scaler.transform(X_new)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(X_new_scaled)

    # Convert predictions to binary values (1 for approved, 0 for rejected)
    binary_predictions = (predictions > 0.5).astype(int)

    # Print the predictions
    print(binary_predictions)