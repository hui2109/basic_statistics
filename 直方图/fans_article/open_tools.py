import os

import pandas
import pandas as pd


def open_specific_path(pre_or_post, norm, fdr, fs_nums_feature, classifier):
    root_path = r'D:\exe_code\FAE\fans_data'
    path = os.path.join(root_path, pre_or_post, 'results', norm, fdr, fs_nums_feature, classifier, 'metrics.csv')
    with open(path, 'r', 1, 'utf-8') as f:
        for line in f.readlines():
            if line.startswith('cv_train_AUC'):
                cv_train_AUC = line.strip().split(',')[-1]
            elif line.startswith('test_AUC'):
                test_AUC = line.strip().split(',')[-1]
            elif line.startswith('cv_val_AUC'):
                cv_val_AUC = line.strip().split(',')[-1]
            elif line.startswith('cv_val_95%'):
                cv_val_AUC_low, cv_val_AUC_high = line.strip().split('[')[-1][:-1].split('-')
            elif line.startswith('cv_val_Std'):
                cv_val_std = line.strip().split(',')[-1]
    return {
        'cross-validation training': float(cv_train_AUC),
        'cross-validation validation': float(cv_val_AUC),
        'testing': float(test_AUC),
        'cross-validation lower_limit': float(cv_val_AUC_low),
        'cross-validation upper_limit': float(cv_val_AUC_high),
        'cross-validation std': float(cv_val_std)
    }


def open_specific_prediction_file(pre_or_post, norm, fdr, fs_nums_feature, classifier, prediction_file):
    root_path = r'D:\exe_code\FAE\fans_data'
    path = os.path.join(root_path, pre_or_post, 'results', norm, fdr, fs_nums_feature, classifier, prediction_file)
    with open(path, 'r', 1, 'utf-8') as f:
        df = pandas.read_csv(path)
        y_true = df['Label']
        y_score = df['Pred']
        return y_true, y_score


if __name__ == '__main__':
    # print(open_specific_path('ceus_pre', 'Mean', 'PCA', 'ANOVA_14', 'LR'))
    print(open_specific_prediction_file('ceus_pre', 'Mean', 'PCA', 'ANOVA_14', 'LR', 'cv_val_prediction.csv'))
