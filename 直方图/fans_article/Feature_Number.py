import matplotlib.pyplot as plt
import numpy as np

from open_tools import open_specific_path


def get_data(pre_or_post, norm, fdr, fs, classifier):
    auc_values = []
    # lower_limits = []
    # upper_limits = []
    stds = []
    for i in range(1, 31):
        result_dict = open_specific_path(pre_or_post, norm, fdr, f'{fs}_{i}', classifier)
        auc_values.append(result_dict['cross-validation validation'])
        # lower_limits.append(result_dict['cross-validation lower_limit'])
        # upper_limits.append(result_dict['cross-validation upper_limit'])
        stds.append(result_dict['cross-validation std'])

    auc_values = np.array(auc_values)
    stds = np.array(stds)
    lower_limits = auc_values - stds
    upper_limits = auc_values + stds
    best_performance_minus_se = get_std(auc_values, pre_or_post, norm, fdr, fs, classifier)

    return auc_values, lower_limits, upper_limits, best_performance_minus_se


def get_std(auc_values, pre_or_post, norm, fdr, fs, classifier):
    # 计算1-se
    best_auc = np.max(auc_values)
    best_auc_fm = np.argmax(auc_values) + 1
    result_dict = open_specific_path(pre_or_post, norm, fdr, f'{fs}_{best_auc_fm}', classifier)
    std_ = result_dict['cross-validation std']
    # 计算1-SE规则中的阈值
    best_performance_minus_se = best_auc - std_
    return best_performance_minus_se


def draw_error_bar(ax, auc_values: np.array, lower_limits: np.array, upper_limits: np.array, title, curr_num,
                   best_performance, show_legend):
    # x是特征数量
    x = np.arange(1, 31)

    # 计算误差条长度
    yerr = np.vstack((auc_values - lower_limits, upper_limits - auc_values))

    # 绘制误差图
    line1 = ax.errorbar(x, auc_values, yerr=yerr, fmt='.-', capsize=2, ecolor='#6989b9', color='#203378',
                        label='cross-validation validation set')
    # 绘制当前选择的点
    line2 = ax.scatter(x[curr_num - 1], auc_values[curr_num - 1], color='#e3bf2b', label='current model')
    # 绘制1-SE线
    line3 = ax.axhline(best_performance, color='#a9c1a3', ls='--', label='best performance - SE')

    # 设置图形标题和坐标轴标签
    font_dict = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 12}
    ax.set_ylabel('AUC', fontdict=font_dict)
    ax.set_xlabel('Feature number', fontdict=font_dict)
    ax.set_title(title, fontdict=font_dict)

    # 设置坐标轴数字为Times New Roman字体
    text = ax.get_yticklabels()
    for tx in text:
        tx.set_fontproperties('Times New Roman')
    text = ax.get_xticklabels()
    for tx in text:
        tx.set_fontproperties('Times New Roman')

    # 设置y轴范围和locator
    # ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))

    ax.set_xlim(0, 31)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置图例
    if show_legend:
        ax.legend(loc='lower right', ncols=1, prop={
            'family': 'Times New Roman',
            'size': 10,
            'weight': 'bold'
        }, fancybox=True, handles=[line1, line2, line3], framealpha=0.4)


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
    y_min, y_max = ax.get_ylim()
    ax.text(-5.5, (y_max - y_min) * 0.95 + y_min, **param_dict)


if __name__ == '__main__':
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=300)

    auc_values, lower_limits, upper_limits, best_performance = get_data('ceus_pre', 'Mean', 'PCA', 'Relief', 'AE')
    draw_error_bar(ax[0, 0], auc_values, lower_limits, upper_limits, 'AE model before NCRT', 17, best_performance,
                   False)
    add_fig_num(ax[0, 0], 'A')

    auc_values, lower_limits, upper_limits, best_performance = get_data('ceus_pre', 'Zscore', 'PCA', 'RFE', 'RF')
    draw_error_bar(ax[0, 1], auc_values, lower_limits, upper_limits, 'RF model before NCRT', 7, best_performance, True)
    add_fig_num(ax[0, 1], 'B')

    auc_values, lower_limits, upper_limits, best_performance = get_data('ceus_post', 'Zscore', 'PCC', 'KW', 'AE')
    draw_error_bar(ax[1, 0], auc_values, lower_limits, upper_limits, 'AE model after NCRT', 8, best_performance, False)
    add_fig_num(ax[1, 0], 'C')

    auc_values, lower_limits, upper_limits, best_performance = get_data('ceus_post', 'Mean', 'PCC', 'KW', 'RF')
    draw_error_bar(ax[1, 1], auc_values, lower_limits, upper_limits, 'RF model after NCRT', 4, best_performance, False)
    add_fig_num(ax[1, 1], 'D')

    plt.savefig('./saved_figs/Figure 5.jpg', bbox_inches='tight')
    plt.show()
