import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('C:\\Users\\greg\\Desktop\\workspace\\dip\\oasis_cross-sectional.csv')


df = df.drop('Delay',1)  # delete a column
df = df.drop('Hand',1)

df = df[np.isfinite(df['SES'])]  # delete row in "SES" with NaN
df = df[np.isfinite(df['Educ'])]
df = df[np.isfinite(df['MMSE'])]
df = df[np.isfinite(df['CDR'])]


def calculate_intervals(col,n):
    mean_values = []
    intervals = []
    max_val = col.max()
    min_val = col.min()
    interval = (max_val - min_val)/ n
    for i in range(0,n):
        interval_min = min_val + i * interval # min of every interval
        interval_max = min_val+(i+1) * interval # max --//--
        mean_val = (interval_min + interval_max) / 2
        mean_values.append(mean_val)
        intervals.append((interval_min, interval_max))
    print(mean_values)
    return mean_values, intervals


def norma(col,intervals,mean_values):
    for i in col:
        for j,k in enumerate(intervals): # j is a counter, k is "intervals[j]
            if i >= k[0] and i < k[1]:
                col.replace(i, mean_values[j], inplace = True) # True -> replaces and saves new values
                break
    print(col.head())


mean_values, intervals = calculate_intervals(df["ASF"], 10)
norma(df["ASF"], intervals, mean_values )
mean_values, intervals = calculate_intervals(df["nWBV"],10)
norma(df["nWBV"],intervals, mean_values)
#print(df)

