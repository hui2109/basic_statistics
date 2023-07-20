import os

import pandas as pd
from sklearn.metrics import roc_curve, auc


def get_plot_data(path, y_true_label, y_score_label):
    df = pd.read_csv(path)
    y_true = list(df[y_true_label])
    yscore = list(df[y_score_label])

    fpr, tpr, thresholds = roc_curve(y_true, yscore)
    ac = auc(fpr, tpr)
    return ac


if __name__ == '__main__':
    for i in os.listdir('../../aresources/fans_data/data'):
        ac = get_plot_data(os.path.join('../../aresources/fans_data/data', i), 'Label', 'Pred')
        print(i, ac)


