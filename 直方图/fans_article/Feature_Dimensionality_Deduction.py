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
    ax.text(-0.6, 0.95, **param_dict)


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

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=300)

    # NCRT治疗前AE
    source_1_1 = open_specific_path('ceus_pre', 'Mean', 'PCA', 'Relief_17', 'AE')  # **
    source_1_2 = open_specific_path('ceus_pre', 'Mean', 'PCC', 'Relief_17', 'AE')
    data_1 = {
        'cross-validation training set': (
            source_1_1['cross-validation training'], source_1_2['cross-validation training']),
        'cross-validation validation set': (
            source_1_1['cross-validation validation'], source_1_2['cross-validation validation']),
        'testing set': (source_1_1['testing'], source_1_2['testing'])
    }
    species = ['PCA' + sample, 'PCC']
    draw_grouped_histogram(ax[0, 0], species, data_1, 'AE model before NCRT')
    add_fig_num(ax[0, 0], 'A')
    ax[0, 0].legend(loc='upper left', ncols=1, prop={
        'family': 'Times New Roman',
        'size': 10,
        'weight': 'bold'
    })

    # NCRT治疗前RF
    source_2_1 = open_specific_path('ceus_pre', 'Zscore', 'PCA', 'RFE_7', 'RF')  # **
    source_2_2 = open_specific_path('ceus_pre', 'Zscore', 'PCC', 'RFE_7', 'RF')
    data_2 = {
        'cross-validation training set': (
            source_2_1['cross-validation training'], source_2_2['cross-validation training']),
        'cross-validation validation set': (
            source_2_1['cross-validation validation'], source_2_2['cross-validation validation']),
        'testing set': (source_2_1['testing'], source_2_2['testing'])
    }
    species = ['PCA' + sample, 'PCC']
    draw_grouped_histogram(ax[0, 1], species, data_2, 'RF model before NCRT')
    add_fig_num(ax[0, 1], 'B')

    # NCRT治疗后AE
    source_3_1 = open_specific_path('ceus_post', 'Zscore', 'PCA', 'KW_8', 'AE')
    source_3_2 = open_specific_path('ceus_post', 'Zscore', 'PCC', 'KW_8', 'AE')  # **
    data_3 = {
        'cross-validation training set': (
            source_3_1['cross-validation training'], source_3_2['cross-validation training']),
        'cross-validation validation set': (
            source_3_1['cross-validation validation'], source_3_2['cross-validation validation']),
        'testing set': (source_3_1['testing'], source_3_2['testing'])
    }
    species = ['PCA', 'PCC' + sample]
    draw_grouped_histogram(ax[1, 0], species, data_3, 'AE model after NCRT')
    add_fig_num(ax[1, 0], 'C')

    # NCRT治疗后RF
    source_4_1 = open_specific_path('ceus_post', 'Mean', 'PCA', 'KW_4', 'RF')
    source_4_2 = open_specific_path('ceus_post', 'Mean', 'PCC', 'KW_4', 'RF')  # **
    data_4 = {
        'cross-validation training set': (
            source_4_1['cross-validation training'], source_4_2['cross-validation training']),
        'cross-validation validation set': (
            source_4_1['cross-validation validation'], source_4_2['cross-validation validation']),
        'testing set': (source_4_1['testing'], source_4_2['testing'])
    }
    species = ['PCA', 'PCC' + sample]
    draw_grouped_histogram(ax[1, 1], species, data_4, 'RF model after NCRT')
    add_fig_num(ax[1, 1], 'D')

    plt.savefig('./saved_figs/Figure 2.jpg', bbox_inches='tight')
    plt.show()
