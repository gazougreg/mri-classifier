import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('C:\\Users\\greg\\Desktop\\workspace\\dip\\oasis_cross-sectional.csv')

var=df.head(3)#print(df.head(3)) - prints first 3 rows
print(var)

print(df.describe()) # returns statistics
print(df['MMSE'].value_counts()) # counts and returns unique values
h=df.loc[(df["Hand"]=="R") & (df["SES"]==2.0) & (df["CDR"]==0.5),["Hand","SES","CDR","ID"]]#boolean = "what to look for"
# & ["","",...]="what to return"
print(h)

#create new function
def num_missing(x):
    return sum(x.isnull())
#Applying per column:
print("Missing values per column: ")
print (df.apply(num_missing,axis=0))#axis=0 defines that
#function is to be applied on each column

#Applying per row
print("\nMissing per row: ")
print(df.apply(num_missing, axis=1).head())

from scipy.stats import  mode # returns an array of the most common value of the given array
print(mode(df['SES']).mode[0])
df['SES'].fillna(mode(df['SES']).mode[0], inplace=True) # inplace= TR -> saves the changes in df object
df['MMSE'].fillna(mode(df['MMSE']).mode[0], inplace=True)
print(df.apply(num_missing, axis=0))

impute_grps = df.pivot_table(values = ["eTIV"], index = ["SES","MMSE"],aggfunc = np.mean)
print(impute_grps)

#dfname['column_name']basic indexing
#technique to acess a particular column

h=df['CDR'].hist(bins=20) #hist calls plot
plt.show()
df.boxplot(column='CDR')
plt.show()
df.boxplot(column='CDR', by = 'Age')
plt.show()

h= df.hist()
plt.show()