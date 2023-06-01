import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# 数据获取
df = pd.read_excel('./data.xlsx')
label = np.array(df['label'])
scores = np.array(df['score'])

# 数据处理
fpr, tpr, thresholds = roc_curve(label, scores, pos_label=1)
auc_score = auc(fpr, tpr)

# 开始绘图
# 参数设置
figure, ax = plt.subplots(figsize=(6, 6))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(which='both', linestyle='--', color='darkgrey', alpha=0.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
ax.set_xlabel('False Positive Rate', fontsize=12, fontdict={
    'family': 'Times New Roman',
    'size': 12,
    'weight': 'bold'})
ax.set_ylabel('True Positive Rate', fontsize=12, fontdict={
    'family': 'Times New Roman',
    'size': 12,
    'weight': 'bold'})
ax.set_title('Model SVM', fontsize=14, fontdict={
    'family': 'Times New Roman',
    'size': 14,
    'weight': 'bold'})

# 绘制ROC曲线
param_dict = {
    'color': '#203378',
    'lw': 2,
    'marker': '.',
    'label': 'ROC curve',
}
roc, = ax.plot(fpr, tpr, **param_dict)

# 绘制曲线下面积
area = ax.fill_between(fpr, tpr, 0, facecolor='#6989b9', alpha=0.5, label='Area under curve (AUC)')

# 找到最靠近左上角的点并绘制
distance = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
closest_point_idx = np.argmin(distance)
closest_point = (fpr[closest_point_idx], tpr[closest_point_idx])
bp = ax.scatter(closest_point[0], closest_point[1], c='red', label='Current classifier')
# 定义文本框的位置和内容
textbox_props = dict(boxstyle='round', alpha=0)
text_content = f'({closest_point[0]:.2f}, {closest_point[1]:.2f})'
# 在指定区域插入文本框
ax.text(closest_point[0]+0.1, closest_point[1]-0.025, text_content, bbox=textbox_props, fontdict={
    'family': 'Times New Roman',
    'size': 12,
    'weight': 'bold',
    'horizontalalignment': 'center',
    'verticalalignment': 'center'})

# 绘制辅助线
param_dict = {
    'color': '#da927c',
    'lw': 2,
    'linestyle': ':',
    'alpha': 0.5,
    'label': 'Baseline',
}
base, = ax.plot([0, 1], [0, 1], **param_dict)

# 设置图例
font_props = {'family': 'Times New Roman', 'size': 10, 'weight': 'bold'}
ax.legend(prop=font_props, loc='lower right', facecolor='white', edgecolor='black', framealpha=0.5)

# 添加AUC参数
# 定义文本框的位置和内容
textbox_props = dict(boxstyle='round', facecolor='#a9c1a3', edgecolor='#a9c1a3', alpha=0.5)
text_content = f'Positive class: 1\nAUC: {auc_score:.2f}'
# 在指定区域插入文本框
ax.text(0.6, 0.35, text_content, bbox=textbox_props, fontdict={
    'family': 'Times New Roman',
    'size': 12,
    'weight': 'bold',
    'horizontalalignment': 'center',
    'verticalalignment': 'center'})

plt.savefig('./roc.png', dpi=300)
plt.show()
