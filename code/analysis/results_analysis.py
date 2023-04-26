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

    # todo more stats >:)