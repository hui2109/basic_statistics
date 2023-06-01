import numpy as np
import pandas
from sklearn.metrics import roc_curve, auc


def accuracy(scores, label, thresholds):
    acc_list = []
    for threshold in thresholds:
        sco = scores.copy()
        mask = (sco >= threshold)
        anti_mask = (sco < threshold)
        sco[mask] = 1
        sco[anti_mask] = 0

        pre_mask = (sco == label)
        acc = len(label[pre_mask]) / len(label)
        # print(len(label[pre_mask]), len(label))
        acc_list.append(acc)
        # break
    return acc_list


df = pandas.read_excel('./data.xlsx')

label = np.array(df['label'])
scores = np.array(df['score'])

# roc_curve的输入为
# y: 样本标签
# scores: 模型对样本属于正例的概率输出

fpr, tpr, thresholds = roc_curve(label, scores, pos_label=1)
AUC = auc(fpr, tpr)
print('假阳性率为：', fpr)
print('真阳性率为：', tpr)
print('阈值为：', thresholds)
print('AUC为：', AUC)

# 计算准确率
acc = accuracy(scores, label, thresholds)
# print(x)
print('准确率为：', max(acc))

# print('--------------------')
#
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {AUC})')
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC曲线')
# plt.legend(loc='lower right')
# plt.show()
