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

