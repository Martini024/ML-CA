import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/student/student-mat-categorical-processed.csv')

numerical_feature_mask = df.dtypes == 'int64'
numerical_cols = df.columns[numerical_feature_mask].tolist()
numerical_cols = numerical_cols[: -3]

# perform standardization(zero-mean, unit variance) to numerical columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

df.to_csv('data/student/student-mat-fully-processed.csv', index=False)
