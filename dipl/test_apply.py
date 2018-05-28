import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('C:\\Users\\greg\\Desktop\\workspace\\dip\\oasis_cross-sectional.csv')


#print(df.iloc[9]["SES"])
#print(df.iloc[2]["SES":"CDR"])
#print(df.head(10))
#t=pd.isnull(df.iloc[1]["SES"])#returns true if row 1 in "ses" is NaN
#print(t)
def misval_count(df,col):
  
    c=0
    for i,r in df.iterrows():
        mv=pd.isnull(df.iloc[i][col])
        if mv == True:
            c=c+1
        
    print(c)
misval_count(df,"SES")

df=df.drop('MR Delay',1) #delete a column

df=df[np.isfinite(df['SES'])] #delete row in "SES" with NaN

print(df)

#print(df.dropna(axis=0, how='all'))

# def misval_handle(df,col):

# def outliers_z_score(ys):
#     threashold=3
#     mean_y=np.mean(ys)
#     stdev_y=np.std(ys)
#     z_scores=[(y-mean_y)/stdev_y for y in ys]
#     return np.where(np.abs(z_scores)>threashold)
# outliers_z_score(df.iloc[:]["SES"].values)