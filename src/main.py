import pandas as pd
from processing.processing import preprocessing, feature_selection, pca
from models.supervised_learning import *


def comprison(df, y, model=None, log_matrix=True, out_path=None):
    res = []
    df = preprocessing(data=df, y=df[y])
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))

    df = preprocessing(data=df, y=df[y], perform_ohe=True)
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))

    df = preprocessing(data=df, y=df[y],
                       perform_ohe=True, perform_scale=True)
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))

    df = preprocessing(data=df, y=df[y], perform_scale=True)
    df = feature_selection(df=df, target=df[y])
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))

    df = preprocessing(data=df, y=df[y], perform_scale=True)
    df = pca(df.loc[:, df.columns.difference([y])], df[y], 0.9)
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))
    if log_matrix:
        df = pd.DataFrame(res, columns=['Accuracy Score', 'Confusion Matrix', 'Training Time', 'Predict Time'], index=[
            'Label Encoded', 'One Hot Encoded', 'Standard Scaled', 'Feature Selection', 'PCA'])
        print(model.__name__)
        print(df)
    if out_path != None:
        df.to_csv(out_path)


# student = pd.read_csv('data/student/student-mat.csv')
# # student['G3'] = student['G3'].apply(lambda x: 1 if x >= 10 else 0)
# # y = student['G3']
# student['G3'] = student[['G1', 'G2', 'G3']].mean(axis=1)
# student = student.drop(columns=['G1', 'G2'])
# student['G3'] = student['G3'].apply(lambda x: 1 if x >= 10 else 0)
# y = student['G3']

# comprison(student, y, model=logistic_regression,
#           out_path='reports/supervised_learning/logistic_regression_avg.csv')
# comprison(student, y, model=knn,
#           out_path='reports/supervised_learning/knn_avg.csv')
# comprison(student, y, model=decision_tree,
#           out_path='reports/supervised_learning/decision_tree_avg.csv')
# comprison(student, y, model=neural_network,
#           out_path='reports/supervised_learning/neural_network_avg.csv')

mushroom = pd.read_csv('data/mushroom/mushrooms.csv')
y = 'class'

comprison(mushroom, y, model=logistic_regression,
          out_path='reports/supervised_learning/mushroom/logistic_regression_mushroom.csv')
comprison(mushroom, y, model=knn,
          out_path='reports/supervised_learning/mushroom/knn_mushroom.csv')
comprison(mushroom, y, model=decision_tree,
          out_path='reports/supervised_learning/mushroom/decision_tree_mushroom.csv')
comprison(mushroom, y, model=neural_network,
          out_path='reports/supervised_learning/mushroom/neural_network_mushroom.csv')
