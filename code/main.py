import models.train_model as model
import preprocessing.standardization.preprocessing as pp
import predictions.predictions as p

production = True

def main_train():
    X_train, X_test, y_train, y_test, scaler = pp.preprocess_raw_data() # preprocess and train algorithm to generate model
    model.train_model(X_train, X_test, y_train, y_test, scaler) # load model and weights and train on new data

def main_production():
    predictions = p.predict_new_data()
    #print(predictions)

def main():

    if production:
        main_production()
    else:
        main_train()

main()