from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import pandas

df = pandas.read_csv('./数据.csv')
# print(len(np.array(df['label'])))

my_params = {
    'pre_PSV': '前PSV',
    'post_chang': '后长径',
    'post_hou': '后厚径',
    'post_RI': '后RI',
    'chang_bianhua': '长径变化率',
    'hou_bianhua': '厚径变化率',
    'post_CEUS': '后造影分级',
    'de_': '降级修改',
    'PRE_2': 'PRE_2',
    'RI_bianhua': 'RI变化率'
}

label = np.array(df['label'])
pre_PSV = np.array(df['前PSV'])
post_chang = np.array(df['后长径'])
post_hou = np.array(df['后厚径'])
post_RI = np.array(df['后RI'])
chang_bianhua = np.array(df['长径变化率'])  # s
hou_bianhua = np.array(df['厚径变化率'])  # s
post_CEUS = np.array(df['后造影分级'])
de_ = np.array(df['降级修改'])  # s
PRE_2 = np.array(df['PRE_2'])
RI_bianhua = np.array(df['RI变化率'])

# roc_curve的输入为
# y: 样本标签
# scores: 模型对样本属于正例的概率输出
fig = plt.figure(dpi=300, figsize=(10, 10))


def my_plot(label, scores, color, name, pos_label=1):
    fpr, tpr, thresholds = roc_curve(label, scores, pos_label=pos_label)
    AUC = round(auc(fpr, tpr), 2)

    lw = 2
    plt.plot(fpr, tpr, color=color, lw=lw, label=f'{name} (area = {AUC})')


if __name__ == '__main__':
    my_plot(label, pre_PSV, 'red', my_params['pre_PSV'])
    my_plot(label, post_chang, 'blue', my_params['post_chang'])
    my_plot(label, post_hou, 'green', my_params['post_hou'])
    my_plot(label, post_RI, 'yellow', my_params['post_RI'])
    my_plot(label, chang_bianhua, 'pink', my_params['chang_bianhua'], pos_label=0)
    my_plot(label, hou_bianhua, 'purple', my_params['hou_bianhua'], pos_label=0)
    my_plot(label, post_CEUS, 'grey', my_params['post_CEUS'])
    my_plot(label, de_, 'orange', my_params['de_'], pos_label=0)
    my_plot(label, PRE_2, 'cyan', my_params['PRE_2'])
    my_plot(label, RI_bianhua, 'darkred', my_params['RI_bianhua'], pos_label=0)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC曲线')
    plt.legend(loc='lower right')
    plt.show()
    fig.savefig('./m_ROC.jpg')
