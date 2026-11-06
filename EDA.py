import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Heart Project/heart.csv")

# Visualizations

"""df.shape                # gives rows X columns
df.head()               # gives top 5 rows
df.info()               # gives info like no. of columns with their name, datatype, non-null count
df.describe()           # gives count, mean, std, min, max, 25%, 50%, 75% just for the columns that have numeric values
df.isnull().sum()       # gives the total of null values for each columns 
df.columns              # gives list of columns
df.duplicated().sum()   # gives the total of duplicated values
df['HeartDisease'].value_counts()       # we want equally distributed data so that our model is accurate

df['HeartDisease'].value_counts().plot(kind="bar")

def plotting(var,num):
    plt.figure(figsize=(6,4))       # if you ignore this all the graph on single page
    plt.subplot(2,2,num)    # 2,2,1 ---> 1st graph till # 2,2,4 ---> 4th and last Graph
    sns.histplot(df[var],kde=True)

plotting('Age',1)
plotting('RestingBP',2)
plotting('Cholesterol',3)
plotting('MaxHR',4)

plt.tight_layout()      # so that there is no overlapping
plt.show()
print(df['Cholesterol'].value_counts()) # from research cholesterol can never be zero but we have zero values
print(df['RestingBP'].value_counts())"""

# EDA and Data Cleaning

ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
df['Cholesterol'] = df['Cholesterol'].replace(0, ch_mean)
df['Cholesterol'] = df['Cholesterol'].round(2)
RBP_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
df['RestingBP'] = df['RestingBP'].replace(0, RBP_mean)
df['RestingBP'] = df['RestingBP'].round(2)

# Now for categorical data

df['HeartDisease'] = df['HeartDisease'].astype(str)         # Convert it to str(0/1) from 0/1 to use it with hue

"""sns.countplot(x = df['Sex'], hue= df['HeartDisease'])       # since HeartDisease is Categorical data target we can compare it together
plt.show()
sns.countplot(x = df['ChestPainType'], hue= df['HeartDisease'])
plt.show()
sns.countplot(x = df['FastingBS'], hue= df['HeartDisease'])
plt.show()"""

# For Numeric and Categorical data

"""sns.boxplot(x='HeartDisease', y='Cholesterol', data=df)
plt.show()
sns.violinplot(x='HeartDisease', y='Age', data=df)
plt.show()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()"""

# Data preprocessing

df_encode= pd.get_dummies(df, drop_first=True)
df_encode = df_encode.astype(int)

# Standard Scaling

from sklearn.preprocessing import StandardScaler
cols = ['Age','RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler= StandardScaler()
df_encode[cols] = scaler.fit_transform(df_encode[cols])
#print(df_encode.head())

#print("Done")

