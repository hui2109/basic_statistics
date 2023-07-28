import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from open_tools import open_specific_path


def draw_grouped_histogram(ax, species: list, data: dict, fig_title):
    keys = list(data.keys())
    vals = list(data.values())
    index = len(keys)
    x = np.arange(len(species))  # the label locations
    colors = ['#203378', '#6989b9', '#a9c1a3']
    width = 0.25  # the width of the bars
    multiplier = 0

    for i in range(index):
        offset = width * multiplier
        ax.bar(x + offset, vals[i], width, label=keys[i], color=colors[i])
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    font_dict = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 12}
    ax.set_ylabel('AUC', fontdict=font_dict)
    ax.set_xticks(x + width, species, fontdict=font_dict)
    ax.set_title(fig_title, fontdict=font_dict)

    # 设置坐标轴数字为Times New Roman字体
    text = ax.get_yticklabels()
    for tx in text:
        tx.set_fontproperties('Times New Roman')

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_fig_num(ax, num: str):
    # 在指定区域插入文本框
    param_dict = {
        's': num,
        'fontdict': {
            'family': 'Times New Roman',
            'size': 12,
            'weight': 'bold',
            'horizontalalignment': 'center',
            'verticalalignment': 'center'
        }}
    ax.text(-1.55, 0.95, **param_dict)


if __name__ == '__main__':
    # 设置公式字体
    config = {
        "font.family": 'serif',
        "font.size": 10,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun']  # simsun字体中文版就是宋体
    }
    rcParams.update(config)
    # 建立公式样本
    sample = '$^{*}$'

    fig, ax = plt.subplots(2, 2, figsize=(14, 14), dpi=300)

    # NCRT治疗前AE
    source_1_1 = open_specific_path('ceus_pre', 'Mean', 'PCA', 'Relief_17', 'SVM')
    source_1_2 = open_specific_path('ceus_pre', 'Mean', 'PCA', 'Relief_17', 'AE')  # **
    source_1_3 = open_specific_path('ceus_pre', 'Mean', 'PCA', 'Relief_17', 'LDA')
    source_1_4 = open_specific_path('ceus_pre', 'Mean', 'PCA', 'Relief_17', 'RF')
    source_1_5 = open_specific_path('ceus_pre', 'Mean', 'PCA', 'Relief_17', 'LR')
    source_1_6 = open_specific_path('ceus_pre', 'Mean', 'PCA', 'Relief_17', 'LRLasso')
    source_1_7 = open_specific_path('ceus_pre', 'Mean', 'PCA', 'Relief_17', 'GP')
    data_1 = {
        'cross-validation training set': (
            source_1_1['cross-validation training'], source_1_2['cross-validation training'],
            source_1_3['cross-validation training'], source_1_4['cross-validation training'],
            source_1_5['cross-validation training'], source_1_6['cross-validation training'],
            source_1_7['cross-validation training']
        ),
        'cross-validation validation set': (
            source_1_1['cross-validation validation'], source_1_2['cross-validation validation'],
            source_1_3['cross-validation validation'], source_1_4['cross-validation validation'],
            source_1_5['cross-validation validation'], source_1_6['cross-validation validation'],
            source_1_7['cross-validation validation']
        ),
        'testing set': (
            source_1_1['testing'], source_1_2['testing'],
            source_1_3['testing'], source_1_4['testing'],
            source_1_5['testing'], source_1_6['testing'],
            source_1_7['testing']
        )
    }
    species = ['SVM', 'AE' + sample, 'LDA', 'RF', 'LR', 'LR-Lasso', 'GP']
    draw_grouped_histogram(ax[0, 0], species, data_1, 'AE model before NCRT')
    add_fig_num(ax[0, 0], 'A')
    ax[0, 0].legend(loc='upper right', ncols=1, prop={
        'family': 'Times New Roman',
        'size': 10,
        'weight': 'bold'
    })

    # NCRT治疗前RF
    source_2_1 = open_specific_path('ceus_pre', 'Zscore', 'PCA', 'RFE_7', 'SVM')
    source_2_2 = open_specific_path('ceus_pre', 'Zscore', 'PCA', 'RFE_7', 'AE')
    source_2_3 = open_specific_path('ceus_pre', 'Zscore', 'PCA', 'RFE_7', 'LDA')
    source_2_4 = open_specific_path('ceus_pre', 'Zscore', 'PCA', 'RFE_7', 'RF')  # **
    source_2_5 = open_specific_path('ceus_pre', 'Zscore', 'PCA', 'RFE_7', 'LR')
    source_2_6 = open_specific_path('ceus_pre', 'Zscore', 'PCA', 'RFE_7', 'LRLasso')
    source_2_7 = open_specific_path('ceus_pre', 'Zscore', 'PCA', 'RFE_7', 'GP')
    data_2 = {
        'cross-validation training set': (
            source_2_1['cross-validation training'], source_2_2['cross-validation training'],
            source_2_3['cross-validation training'], source_2_4['cross-validation training'],
            source_2_5['cross-validation training'], source_2_6['cross-validation training'],
            source_2_7['cross-validation training']
        ),
        'cross-validation validation set': (
            source_2_1['cross-validation validation'], source_2_2['cross-validation validation'],
            source_2_3['cross-validation validation'], source_2_4['cross-validation validation'],
            source_2_5['cross-validation validation'], source_2_6['cross-validation validation'],
            source_2_7['cross-validation validation']
        ),
        'testing set': (
            source_2_1['testing'], source_2_2['testing'],
            source_2_3['testing'], source_2_4['testing'],
            source_2_5['testing'], source_2_6['testing'],
            source_2_7['testing']
        )
    }
    species = ['SVM', 'AE', 'LDA', 'RF' + sample, 'LR', 'LR-Lasso', 'GP']
    draw_grouped_histogram(ax[0, 1], species, data_2, 'RF model before NCRT')
    add_fig_num(ax[0, 1], 'B')

    # NCRT治疗后AE
    source_3_1 = open_specific_path('ceus_post', 'Zscore', 'PCC', 'KW_8', 'SVM')
    source_3_2 = open_specific_path('ceus_post', 'Zscore', 'PCC', 'KW_8', 'AE')  # **
    source_3_3 = open_specific_path('ceus_post', 'Zscore', 'PCC', 'KW_8', 'LDA')
    source_3_4 = open_specific_path('ceus_post', 'Zscore', 'PCC', 'KW_8', 'RF')
    source_3_5 = open_specific_path('ceus_post', 'Zscore', 'PCC', 'KW_8', 'LR')
    source_3_6 = open_specific_path('ceus_post', 'Zscore', 'PCC', 'KW_8', 'LRLasso')
    source_3_7 = open_specific_path('ceus_post', 'Zscore', 'PCC', 'KW_8', 'GP')
    data_3 = {
        'cross-validation training set': (
            source_3_1['cross-validation training'], source_3_2['cross-validation training'],
            source_3_3['cross-validation training'], source_3_4['cross-validation training'],
            source_3_5['cross-validation training'], source_3_6['cross-validation training'],
            source_3_7['cross-validation training']
        ),
        'cross-validation validation set': (
            source_3_1['cross-validation validation'], source_3_2['cross-validation validation'],
            source_3_3['cross-validation validation'], source_3_4['cross-validation validation'],
            source_3_5['cross-validation validation'], source_3_6['cross-validation validation'],
            source_3_7['cross-validation validation']
        ),
        'testing set': (
            source_3_1['testing'], source_3_2['testing'],
            source_3_3['testing'], source_3_4['testing'],
            source_3_5['testing'], source_3_6['testing'],
            source_3_7['testing']
        )
    }
    species = ['SVM', 'AE' + sample, 'LDA', 'RF', 'LR', 'LR-Lasso', 'GP']
    draw_grouped_histogram(ax[1, 0], species, data_3, 'AE model after NCRT')
    add_fig_num(ax[1, 0], 'C')

    # NCRT治疗后RF
    source_4_1 = open_specific_path('ceus_post', 'Mean', 'PCC', 'KW_4', 'SVM')
    source_4_2 = open_specific_path('ceus_post', 'Mean', 'PCC', 'KW_4', 'AE')
    source_4_3 = open_specific_path('ceus_post', 'Mean', 'PCC', 'KW_4', 'LDA')
    source_4_4 = open_specific_path('ceus_post', 'Mean', 'PCC', 'KW_4', 'RF')  # **
    source_4_5 = open_specific_path('ceus_post', 'Mean', 'PCC', 'KW_4', 'LR')
    source_4_6 = open_specific_path('ceus_post', 'Mean', 'PCC', 'KW_4', 'LRLasso')
    source_4_7 = open_specific_path('ceus_post', 'Mean', 'PCC', 'KW_4', 'GP')
    data_4 = {
        'cross-validation training set': (
            source_4_1['cross-validation training'], source_4_2['cross-validation training'],
            source_4_3['cross-validation training'], source_4_4['cross-validation training'],
            source_4_5['cross-validation training'], source_4_6['cross-validation training'],
            source_4_7['cross-validation training']
        ),
        'cross-validation validation set': (
            source_4_1['cross-validation validation'], source_4_2['cross-validation validation'],
            source_4_3['cross-validation validation'], source_4_4['cross-validation validation'],
            source_4_5['cross-validation validation'], source_4_6['cross-validation validation'],
            source_4_7['cross-validation validation']
        ),
        'testing set': (
            source_4_1['testing'], source_4_2['testing'],
            source_4_3['testing'], source_4_4['testing'],
            source_4_5['testing'], source_4_6['testing'],
            source_4_7['testing']
        )
    }
    species = ['SVM', 'AE', 'LDA', 'RF' + sample, 'LR', 'LR-Lasso', 'GP']
    draw_grouped_histogram(ax[1, 1], species, data_4, 'RF model after NCRT')
    add_fig_num(ax[1, 1], 'D')

    plt.savefig('./saved_figs/Figure 4.jpg', bbox_inches='tight')
    plt.show()
