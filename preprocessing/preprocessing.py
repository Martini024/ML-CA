import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def preprocessing(df, y, perform_ohe=False, perform_scale=False, scaler=StandardScaler(), output_Path=None, index=False):
    # split columns as categorical and numerical, don't perform scale to numerical y
    is_numerical = df[y].dtypes == 'int64'

    categorical_feature_mask = df.dtypes == object
    categorical_cols = df.columns[categorical_feature_mask].tolist()
    numerical_feature_mask = df.dtypes == 'int64'
    if is_numerical:
        numerical_cols = df.columns[numerical_feature_mask].drop(y).tolist()
    else:
        numerical_cols = df.columns[numerical_feature_mask].tolist()

    # perform label encoding to categorical columns
    le = LabelEncoder()
    df[categorical_cols] = df[categorical_cols].apply(
        lambda col: le.fit_transform(col))

    # perform one hot key encoding to categorical columns and drop original columns
    if perform_ohe:
        if not is_numerical:
            categorical_cols = df.columns[categorical_feature_mask].drop(
                y).tolist()
        df = df.join(pd.get_dummies(
            df[categorical_cols], columns=categorical_cols, prefix=categorical_cols))
        df = df.drop(categorical_cols, 1)

    # perform standardization(zero-mean, unit variance) to numerical columns
    if perform_scale:
        scaler = scaler
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # export as processed csv file
    if output_Path != None:
        df.to_csv(output_Path, index=index)
    return df


df = pd.read_csv('data/mushroom/mushrooms.csv')
preprocessing(df, 'class', output_Path='data/mushroom/mushrooms-processed.csv')
