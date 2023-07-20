import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

with open('../../aresources/my_data/answers.txt', 'r', 1, 'utf-8') as f:
    con = f.read().replace('\n', '')
answers_list = list(con)

# 构造计算字典
# key: city
# value: dict
# key: 题号
# value: 浮点数, 代表本题选择AE的概率
cal_dict = {
    '四川_雅安市': dict(),
    '四川_阿坝藏族羌族自治州': dict(),
    '四川_成都市': dict(),
    '四川_甘孜藏族自治州': dict(),
    '四川_遂宁市': dict(),
    '四川_凉山彝族自治州': dict(),
    '四川_绵阳市': dict(),
    '西藏_拉萨市': dict(),
    '阿坝_甘孜_凉山': dict()
}


def get_plot_data(city):
    df = pd.read_excel('./classify.xlsx', sheet_name=city)
    for i in range(len(df.columns)):
        col = df.iloc[:, i]
        mask = pd.isna(col)
        new_col = col[~mask]

        ques_index = new_col.name.split('、')[0]
        total_num = len(new_col)
        choose_B = [j for j in new_col if j.startswith('B')]
        pro = len(choose_B) / total_num  # pro代表选择B的概率

        cal_dict[city][ques_index] = pro


for key in cal_dict.keys():
    get_plot_data(key)

# key: city
# value: dict
# key: 'y_true'
# value: y_true
# key: 'y_score'
# value: y_score
plot_data_dict = {
    '四川_雅安市': dict(),
    '四川_阿坝藏族羌族自治州': dict(),
    '四川_成都市': dict(),
    '四川_甘孜藏族自治州': dict(),
    '四川_遂宁市': dict(),
    '四川_凉山彝族自治州': dict(),
    '四川_绵阳市': dict(),
    '西藏_拉萨市': dict(),
    '阿坝_甘孜_凉山': dict()
}

y_true = []
y_score = []
for key in plot_data_dict:
    y_true = []
    y_score = []
    for ques_index in cal_dict[key].keys():
        label = answers_list[int(ques_index) - 1]
        if label == 'A':
            y_true.append(0)
        else:
            y_true.append(1)
        y_score.append(cal_dict[key][ques_index])
    plot_data_dict[key]['y_true'] = y_true
    plot_data_dict[key]['y_score'] = y_score


# print(plot_data_dict['四川_雅安市']['y_score'])


def my_plot(label, scores, name, pos_label=1):
    fpr, tpr, thresholds = roc_curve(label, scores, pos_label=pos_label)
    AUC = round(auc(fpr, tpr), 2)

    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{name} (area = {AUC})')


for city, data in plot_data_dict.items():  # data是一个字典
    label = np.array(data['y_true'])
    scores = np.array(data['y_score'])
    # 写入到文件
    df = pd.DataFrame({'label': label, 'scores': scores})
    df.to_csv(f'./plot_data/{city}.csv', index=False)


    # my_plot(label, scores, name=city)
#
# plt.legend(loc='lower right')
# plt.savefig('./roc_new.png', dpi=300)
# plt.show()
