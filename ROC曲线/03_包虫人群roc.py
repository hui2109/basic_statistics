import math

import pandas
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report


# def get_answers(answer):
#     with open('./answers.txt', 'a+', 1, 'utf-8') as f:
#         for i in range(15):
#             f.write(answer)
#         f.write('\n')


# A: CE; B: AE
# answer_list = ['A', 'A', 'B', 'B', 'A',
#                'A', 'B', 'A', 'A', 'A',
#                'A', 'B', 'B', 'B', 'A',
#                'B', 'B', 'B', 'B', 'A',
#                'A', 'B', 'A', 'A', 'B',
#                'B', 'B', 'A', 'B', 'A',
#                ]
# answer_list = ['A', 'A', 'A', 'A', 'A',
#                'A', 'A', 'A', 'A', 'A',
#                'A', 'A', 'A', 'A', 'A',
#                'B', 'B', 'B', 'B', 'B',
#                'B', 'B', 'B', 'B', 'B',
#                'B', 'B', 'B', 'B', 'B',
#                ]
# for i in answer_list:
#     get_answers(i)


def preprocess():
    stat = {}
    df = pandas.read_excel('./detail.xlsx')

    for column in df.columns:
        total = 0
        total_A = 0
        total_B = 0
        for item in df[column]:
            if not item.startswith('('):
                total += 1
            if item.startswith('A'):
                total_A += 1
            elif item.startswith('B'):
                total_B += 1
        seq = column.split('、')[0]
        stat[f'第{seq}题'] = {
            'total_Num': total,
            'choose_A': total_A,
            'choose_B': total_B
        }
    return stat


if __name__ == '__main__':
    with open('./answers.txt', 'r', 1, 'utf-8') as f:
        answers = f.read().replace('\n', '')

    stat = preprocess()
    y_score = []
    accuracy_li = []

    for i in range(len(answers)):
        total_Num = stat[f'第{i + 1}题']['total_Num']
        choose_A = stat[f'第{i + 1}题']['choose_A']
        choose_B = stat[f'第{i + 1}题']['choose_B']

        if answers[i] == 'A':
            accuracy = choose_A/total_Num
        else:
            accuracy = choose_B/total_Num
        y_score.append(choose_B / total_Num)
        accuracy_li.append(accuracy)

    y_true = list(answers.replace('A', '0').replace('B', '1'))
    y_true = [int(i) for i in y_true]
    # print(y_score)

    y_pred = [1 if i >= 0.5 else 0 for i in y_score]
    # print(y_pred)

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    print('auc:', auc(fpr, tpr))
    print(classification_report(y_true, y_pred))
    print('accuracy:', sum(accuracy_li)/len(accuracy_li))

    plt.figure(figsize=(10, 10), dpi=300)
    plt.plot(fpr, tpr, marker='o')
    plt.plot([0, 1], [0, 1], marker='.', color='pink')
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title('ROC Curve', fontsize=22)
    plt.show()
