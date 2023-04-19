import pandas as pd

def predict_new_data():
    data = pd.read_csv('./data/predicted_data.csv') # Load new data
    raw_predictions = data['Raw_Predictions']
    interest_rates = rbes_range(raw_predictions)
    new_data = data.drop(['Raw_Predictions'], axis=1)
    new_data['Interest_Rates'] = interest_rates
    new_data.to_csv('./data/predicted_data.csv', index=False)

def rbes_range(raw_predictions):
    interest_rates = []
    for prediction in raw_predictions:
        rate = (1 - prediction) * 100

        rate_range = "Very Low"
        if rate <= 10 and rate > 5:
            rate_range = "Low"
        elif rate <= 15 and rate > 10:
            rate_range = "Medium"
        elif rate <= 20 and rate > 15:
            rate_range = "Medium-High"
        elif rate <= 25 and rate > 20:
            rate_range = "High"
        elif rate <= 50:
            rate_range = "Extremly High"
        else: 
            rate_range = "Denied"

        interest_rates.append(rate_range)

    return interest_rates