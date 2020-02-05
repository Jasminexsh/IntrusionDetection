import pandas as pd

df = pd.read_csv(r'data\1-kddcup.data_10_percent_corrected.csv')
print(df.iloc[:, -1].unique())
df1 = pd.read_csv(r'data\3-corrected.csv')
print(df1.iloc[:, -1].unique())
