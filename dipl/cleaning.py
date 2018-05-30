import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('C:\\Users\\greg\\Desktop\\workspace\\dip\\oasis_cross-sectional.csv')


def count_misvalues (df, col):
    c = 0
    for i, r in df.iterrows():
        mv = pd.isnull(df.iloc[i][col])
        if mv == True:
            c = c + 1

    print(c)


# count_misvalues(df,"SES")
df = df.drop('Delay',1)  # delete a column
df = df.drop('Hand',1)

df = df[np.isfinite(df['SES'])]  # delete row in "SES" with NaN
df = df[np.isfinite(df['Educ'])]
df = df[np.isfinite(df['MMSE'])]
df = df[np.isfinite(df['CDR'])]


df['M/F'].replace('M', 0, inplace = True)
df['M/F'].replace('F', 1, inplace = True)
print(df)

