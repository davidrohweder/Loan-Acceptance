import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def data_analysis(statistic=1):
    data = pd.read_csv('./data/predicted_data.csv') # Load new data

    # case 1: hist of interest cats totals
    if statistic == 1:
        interest_rate_counts = data['Interest_Rates'].value_counts()
        plt.figure(figsize=(8, 6))
        sns.barplot(x=interest_rate_counts.index, y=interest_rate_counts.values)
        plt.title('Histogram of Total in Each Interest Rates Category')
        plt.xlabel('Interest Rates Categories')
        plt.ylabel('Total')
        plt.show()

    # case 2: 95% confidence intervals for the average loan amount in the low cat
    if statistic == 2:
        low_category = data[data['Interest_Rates'] == 'Low']
        low_category_mean = low_category['LoanAmount'].mean()
        low_category_std = low_category['LoanAmount'].std()
        n = len(low_category)
        confidence = 0.95
        margin_of_error = stats.norm.ppf(1 - (1 - confidence) / 2) * (low_category_std / np.sqrt(n))
        lower_bound = low_category_mean - margin_of_error
        upper_bound = low_category_mean + margin_of_error
        print(f"Confidence Intervals (95%): ({lower_bound}, {upper_bound})")

    # case 3: Box plot of LoanAmount by Property_Area
    elif statistic == 3:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Property_Area', y='LoanAmount', data=data)
        plt.title('Box Plot of LoanAmount by Property Area')
        plt.xlabel('Property Area')
        plt.ylabel('Loan Amount')
        plt.show()

    #case 4: # Scatter plot of ApplicantIncome vs. CoapplicantIncome
    elif statistic == 4:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='ApplicantIncome', y='CoapplicantIncome', data=data)
        plt.title('Scatter Plot of ApplicantIncome vs. CoapplicantIncome')
        plt.xlabel('Applicant Income')
        plt.ylabel('Coapplicant Income')
        plt.show()

    #case 5 : Count plot of Loan_Amount_Term
    elif statistic == 5:
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Loan_Amount_Term', data=data)
        plt.title('Count Plot of Loan Amount Term')
        plt.xlabel('Loan Amount Term')
        plt.ylabel('Count')
        plt.show()
    