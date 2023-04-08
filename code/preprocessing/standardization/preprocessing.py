import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_raw_data():
    data = pd.read_csv('./preprocessing/raw_data/training_data.csv') # Load and preprocess the data
    data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

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

    # Split data into features (X) and target (y)
    X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = data['Loan_Status']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
