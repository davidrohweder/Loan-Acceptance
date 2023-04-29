# Loan-Acceptance

## Project Description 
The Loan Acceptance project aims to use AI in order to select which candidates should be approved for a loan & what the loan interest rate should be based on various attributes. This project aims to reduce the bias with a person-to-person meeting that a bank may have towards a particular person and go purely off the numbers in order to make a decision on whether or not the loan request should be approved. 

## Goals, Envitonment, Adaptation
### Goals
The goals are approval or denial based off given data about an applicant as well as an estimate of interest rates.

### Environment
User demographics, credit history, application data, geographical location, current FED rates and standard industry values. 

### Adaptation
Augmenting the shape of the algorithm by adding new layers or nodes, changing set interest rates based on data sources, and using human reviews and appeal systems in case an applicant believes the system's decision was incorrect. 

## Design & Implementation
In this section, we will delve into the intricacies of the machine learning model utilized for loan approval prediction. To achieve the most accurate model, two primary training approaches were considered: a brute-force method and a randomly optimal method, known as the lazy_train function. Notably, the lazy_train function differs from the brute-force method as it only has one hidden layer.  

The model's architecture follows an input layer with nodes equivalent to the feature size, connected to a hidden layer with 16 neurons and a tanh activation function. The output layer has a single neuron with a sigmoid activation function for binary classification, mapping the result to either 1 (approved) or 0 (denied).  

The training process consists of 1000 epochs, a batch size of 128, and the choice of Adagrad or simply “adam” as learning functions. The model's accuracy is evaluated against the testing dataset, and the best model is determined based on the highest accuracy achieved during training. The primary function, train_model(), implements a loop to optimize each layers node quantity and after the most optimal batch size and then trains the model using the most optimized configuration. The lazy_train() function, on the other hand, relies on a more simplified architecture and training process. Both approaches resulted in similar results the highest the full brute-force approach was able to achieve was 80.12% accuracy in the data but took over full day to train. 

 The lazy_train() method however achieved 78.86% accuracy only taking several seconds to preprocess and train the model.  

The main.py script is the router and brain behind the application and determines whether to train the model, make predictions, or analyze the data. In the preprocessing.py script, the raw data is loaded, preprocessed, and split into training and testing datasets (80/20 split). The approval_predictions.py script loads the trained model, preprocesses new data, and generates predictions based on the model's output. By evaluating and comparing the two training approaches, we can determine the best model for predicting loan approvals. The chosen model should exhibit high accuracy and low loss in its evaluation against the testing dataset, proving its reliability and correctness compared to alternative models. This comprehensive analysis allows us to confidently deploy the most suitable model for loan approval prediction. 

In addition to predicting loan approvals with binary outputs of 1 or 0, the code provided also calculates the interest rate for each customer using a Rule-Based Expert System. This calculation leverages the raw output values from the prediction model to determine the appropriate interest rate range for each user.  

The algorithm begins by loading the new data with the predicted loan approvals and corresponding raw predictions. The interest rates for each customer are calculated using the rbes_range() function. This function iterates over the raw predictions and computes the interest rate by subtracting the prediction value from 1 and multiplying the result by 100. Depending on the calculated interest rate, the customer is assigned to a specific rate range, such as "Very Low," "Low," "Medium," "Medium-High," "High," "Extremely High," or "Denied."  

Once the interest rate ranges have been calculated for each customer, the new_data dataframe is updated to include these values. The 'Raw_Predictions' column is dropped, as it is no longer necessary, and the 'Interest_Rates' column is added to store the assigned rate ranges. Finally, the updated dataframe is saved as a new CSV file, 'predicted_data.csv,' preserving the loan approval predictions and corresponding interest rate ranges for further analysis or decision-making.  

## Project Dependenceis 
This project has the following dependencies: 
1. Pandas
2. Tensorflow
3. Sklearn

## Installation Steps

### Training the Model & Predicting Data ###
To install the dependencies, run the following commands (NOTE: SKlearn has dependencies #3-6 and those need to be installed before intalling SKlearn): 
1. ``pip install pandas``
2. ``pip install tensorflow`` 
3. ``pip install numpy``
4. ``pip install scipy``
5. ``pip install joblib``
6. ``pip install threadpoolctl``
7. ``pip install -U scikit-learn``

If you have Python version 3.9 or above, the Pickle module does not need to be installed. To check your current python version, you can do ``python --version``. If you have Python version 3.9 or below, you need to install the Pickle module with the following command: 

8. ``pip install pickle``

### Analyzing Data ### 
1. ``pip install matplotlib``
2. ``pip install seaborn`` 
3. ``pip install scipy``

## Steps to Run 

### Initalization ###

*How we reccomend*
1. In the ``\code\`` subdirectory run: ``python -m venv modelEnv``
2. In the that same directory run: ``modelEnv\Scripts\Activate.ps1``

### Running the application ###
To run this project, open the terminal from the code subfolder. There, run the following command: 
Ubuntu: ``$ python main.py``
Windows: ``python main.py``

### Augmenting Arguments ###

1. Training the Model
  * In the ``main.py`` on line 7 make sure ``production = False``
  * In the ``main.py`` on line 8 make sure ``analytics_mode = False``
  * In the ``main.py`` on line 12 either add or do not add the argument ``lazy_mode=False`` depending on if a full model should be trained
2. Predicting Data
  * In the ``main.py`` on line 7 make sure ``production = True``
  * In the ``approval_predictions.py`` on line 6 default arguments are presented and can be supplied in the main file when calling to change the default configuration. Arguments include: production for files to use and save to save predicted results
3. Analyzing Data
  * In the ``main.py`` on line 8 make sure ``analytics_mode = True``

