import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def feature_selection(df, target, show_heat_map=False):
    corr_mat = df.corr()

    if show_heat_map:
        plt.figure(figsize=(13, 5))
        sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
        plt.show()

    target_name = target.name
    candidates = corr_mat.index[
        (corr_mat[target_name] > 0.5) | (corr_mat[target_name] < -0.5)
    ].values
    candidates = candidates[candidates != target_name]
    print('Correlated to', target, ': ', candidates)

    removed = []
    for c1 in candidates:
        for c2 in candidates:
            if (c1 not in removed) and (c2 not in removed):
                if c1 != c2:
                    coef = corr_mat.loc[c1, c2]
                    if coef > 0.6 or coef < -0.6:
                        removed.append(c1)
    print('Removed: ', removed)

    selected_features = [x for x in candidates if x not in removed]
    print('Selected features: ', selected_features)
    return pd.concat([df[selected_features], target], axis=1)


# test example
# feature_selection('data/student/student-mat-fully-processed.csv', 'G3')
# df = pd.read_csv('data/mushroom/mushrooms-l.csv')
# y = df['class']
# print(feature_selection(df, y))
