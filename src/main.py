from models.supervised_learning import *
from models.unsupervised_learning import *
from processing.processing import preprocessing, feature_selection, pca
import pandas as pd
import seaborn as sns


def input_comprison(data, y, model=None, log_matrix=True, out_path=None):
    res = []
    df = preprocessing(data=data, y=data[y])
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))
    res[0].append('Label Encoded')

    df = preprocessing(data=data, y=data[y], perform_ohe=True)
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))
    res[1].append('One Hot Encoded')

    df = preprocessing(data=data, y=data[y], perform_scale=True)
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))
    res[2].append('Standard Scaled')

    df = preprocessing(data=data, y=data[y], perform_scale=True)
    df = feature_selection(df=df, target=df[y], show_process=False)
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))
    res[3].append('Feature Selection')

    df = preprocessing(data=data, y=data[y], perform_scale=True)
    df = pca(df.loc[:, df.columns.difference([y])],
             df[y], 0.9, show_result=False)
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))
    res[4].append('PCA')

    df = pd.DataFrame(res, columns=[
                      'Accuracy Score', 'Confusion Matrix', 'Training Time', 'Predict Time', 'Processing'])
    if log_matrix:
        print(model.__name__)
        print(df)
    if out_path != None:
        df.to_csv(out_path)
    return df


def classification_model_comparison(data, y, name, log_matrix=True):
    res = input_comprison(data, y, model=logistic_regression, log_matrix=log_matrix,
                          out_path='reports/supervised_learning/' + name + '/logistic_regression_' + name + '.csv')

    res = res.append(input_comprison(data, y, model=knn, log_matrix=log_matrix,
                                     out_path='reports/supervised_learning/' + name + '/logistic_regression_' + name + '.csv'))
    res = res.append(input_comprison(data, y, model=decision_tree, log_matrix=log_matrix,
                                     out_path='reports/supervised_learning/' + name + '/logistic_regression_' + name + '.csv'))
    res = res.append(input_comprison(data, y, model=neural_network, log_matrix=log_matrix,
                                     out_path='reports/supervised_learning/' + name + '/logistic_regression_' + name + '.csv'))
    model = ['logistic regression'] * 5 + ['knn'] * 5 + \
        ['decision_tree'] * 5 + ['neural_network'] * 5
    res['model'] = model

    fig, axs = plt.subplots(3)
    fig.suptitle(name)
    plt.subplots_adjust(hspace=0.3)

    sns.barplot(
        y=res['Processing'], x=res['Accuracy Score'], hue=res.model, data=res, orient='h', ax=axs[0])

    sns.barplot(
        y=res['Processing'], x=res['Training Time'], hue=res.model, data=res, orient='h', ax=axs[1])

    sns.barplot(
        y=res['Processing'], x=res['Predict Time'], hue=res.model, data=res, orient='h', ax=axs[2])
    plt.legend(fontsize='small')
    plt.show()


# df = pd.read_csv('data/student/student-mat.csv')
# df['G3'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
# y = 'G3'
# classification_model_comparison(df, y, 'student', log_matrix=False)
# # student['G3'] = student[['G1', 'G2', 'G3']].mean(axis=1)
# # student = student.drop(columns=['G1', 'G2'])
# # student['G3'] = student['G3'].apply(lambda x: 1 if x >= 10 else 0)


# df = pd.read_csv('data/wine/winequality-red.csv')
# df['quality'] = df['quality'].apply(lambda x: 0 if x >= 5 else 1)
# y = 'quality'
# classification_model_comparison(df, y, 'wine', log_matrix=False)
# # print(linear_regression(
# #     df.loc[:, df.columns.difference(['quality'])], df['quality']))

# df = pd.read_csv('data/mushroom/mushrooms.csv')
# y = 'class'
# classification_model_comparison(df, y, 'mushroom', log_matrix=False)

df = pd.read_csv('data/student/student-mat.csv')
df['G3'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
y = 'G3'
kmeans(df, y)
# student['G3'] = student[['G1', 'G2', 'G3']].mean(axis=1)
# student = student.drop(columns=['G1', 'G2'])
# student['G3'] = student['G3'].apply(lambda x: 1 if x >= 10 else 0)


df = pd.read_csv('data/wine/winequality-red.csv')
df['quality'] = df['quality'].apply(lambda x: 0 if x >= 5 else 1)
y = 'quality'
kmeans(df, y)
# classification_model_comparison(df, y, 'wine', log_matrix=False)
# # print(linear_regression(
# #     df.loc[:, df.columns.difference(['quality'])], df['quality']))

df = pd.read_csv('data/mushroom/mushrooms.csv')
y = 'class'
kmeans(df, y)
# classification_model_comparison(df, y, 'mushroom', log_matrix=False)
