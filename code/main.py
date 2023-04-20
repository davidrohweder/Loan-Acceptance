import models.train_approval_model as approval_model # train the accept or reject model ... sg1
import preprocessing.standardization.preprocessing as pp # preprocess the training data
import predictions.approval_predictions as a_p # predict loan approvals 
import predictions.interest_predictions as i_p # predict interest rate brackets
import analysis.results_analysis as ans # do analytics on our results

production = False # if false then we train the models, if true then we predict new data 
analytics_mode = False # if true no predicting or training, only an analysis of the strcutured data

def main_train():
    X_train, X_test, y_train, y_test, scaler = pp.preprocess_raw_data() # preprocess and train algorithm to generate model
    approval_model.train_model(X_train, X_test, y_train, y_test, scaler) # load model and weights and train on new data

def main_production():
    a_p.predict_new_data()
    i_p.predict_new_data()

def main():

    if analytics_mode:
        ans.data_analysis(1)
        ans.data_analysis(2)
    else:
        if production:
            main_production()
        else:
            main_train()

main()