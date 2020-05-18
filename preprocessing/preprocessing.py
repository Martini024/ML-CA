import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

df = pd.read_csv('data/student/student-mat.csv')

# split columns as categorical and numerical
categorical_feature_mask = df.dtypes == object
categorical_cols = df.columns[categorical_feature_mask].tolist()
numerical_feature_mask = df.dtypes == 'int64'
numerical_cols = df.columns[numerical_feature_mask].tolist()
numerical_cols = numerical_cols[: -3]

# perform label encoding to categorical columns
le = LabelEncoder()
df[categorical_cols] = df[categorical_cols].apply(
    lambda col: le.fit_transform(col))

# perform one hot key encoding to categorical columns and drop original columns
df = df.join(pd.get_dummies(
    df[categorical_cols], columns=categorical_cols, prefix=categorical_cols))
df = df.drop(categorical_cols, 1)

# perform standardization(zero-mean, unit variance) to numerical columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# export as processed csv file
df.to_csv('data/student/student-mat-processed.csv', index=False)
