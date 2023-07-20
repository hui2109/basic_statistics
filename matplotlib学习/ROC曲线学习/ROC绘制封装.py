import os
import time
from collections import OrderedDict
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class MyROCPlot:
    SingleMode = 0
    MultipleMode = 1

    def __init__(self, mode, csv_path: Union[str, list]):
        self.mode = mode
        self.csv_path = csv_path

        # 汉字字体，优先使用楷体，找不到则使用黑体
        plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']
        # 正常显示负号
        plt.rcParams['axes.unicode_minus'] = False

    def _get_data(self, csv_path, label_name, score_name):
        df = pd.read_csv(csv_path)
        y_true = np.array(df[label_name])
        y_score = np.array(df[score_name])
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        ac = auc(fpr, tpr)
        return fpr, tpr, ac

    def get_data(self, label_name, score_name: Union[str, list]):
        if type(self.csv_path) == str and type(score_name) == str:
            fpr, tpr, ac = self._get_data(self.csv_path, label_name, score_name)
            return fpr, tpr, ac
        elif type(self.csv_path) == list and type(score_name) == str:
            data = []
            for path in self.csv_path:
                fpr, tpr, ac = self._get_data(path, label_name, score_name)
                data.append((fpr, tpr, ac))
            return data
        elif type(self.csv_path) == str and type(score_name) == list:
            data = []
            for score in score_name:
                fpr, tpr, ac = self._get_data(self.csv_path, label_name, score)
                data.append((fpr, tpr, ac))
            return data

    def single_plot(self, label_name, score_name: str, fig_title, *, c_roc='#203378',
                    c_area='#6989b9', c_base='#da927c', c_text_bk='#a9c1a3'):
        if self.mode != MyROCPlot.SingleMode:
            return 'SingleMode才能调用该方法'

        figure, ax = plt.subplots(figsize=(6, 6), dpi=300)
        fpr, tpr, ac = self.get_data(label_name, score_name)
        self.fig_title = fig_title

        # 参数设置
        self._set_parameters(ax, fig_title)
        # 绘制ROC曲线
        self._roc_curve_plot(ax, fpr, tpr, c_roc)
        # 绘制曲线下面积
        self._roc_area_plot(ax, fpr, tpr, c_area)
        # 找到最靠近左上角的点并绘制
        self._youden_point_plot(ax, fpr, tpr, label='Current classifier')
        # [约登文本框]绘制
        self._youden_text_plot(ax)
        # 绘制基线
        self._baseline_plot(ax, c_base)
        # 定义[AUC文本框]的位置和内容
        self._auc_text_plot(ax, ac, c_text_bk)
        # 设置图例
        self._legend_plot(ax)
        # 保存图片
        self._save_fig()

    def multiple_plot(self, label_name, score_name: Union[str, list], roc_curve_name: tuple,
                      fig_title, rol=1, col=1, *, c_roc: tuple, c_base='#da927c'):
        if self.mode != MyROCPlot.MultipleMode:
            return 'MultipleMode才能调用该方法'

        data = self.get_data(label_name, score_name)
        self.fig_title = fig_title

        total_figures = rol * col
        if total_figures > 1:
            figure, ax = plt.subplots(figsize=(14, 14), dpi=300, nrows=rol, ncols=col)
            # 数据整理
            plot_data = OrderedDict()
            for i in range(len(data)):
                plot_data[roc_curve_name[i]] = data[i]  # data[i]是个tuple

            for i in range(len(roc_curve_name)):
                # 参数设置
                self._set_parameters(ax[i // col, i % col], fig_title)
                # 绘制ROC曲线
                fpr, tpr, ac = plot_data[roc_curve_name[i]]
                self._roc_curves_plot(ax[i // col, i % col], fpr, tpr, c_roc[i], roc_curve_name[i], ac)
                # 找到最靠近左上角的点并绘制
                self._youden_point_plot(ax[i // col, i % col], fpr, tpr)
                # 绘制基线
                self._baseline_plot(ax[i // col, i % col], c_base)
                # 设置图例
                self._legend_plot(ax[i // col, i % col], font='Microsoft YaHei')
            # 保存图片
            self._save_fig()
        else:
            figure, ax = plt.subplots(figsize=(8, 8), dpi=300, nrows=rol, ncols=col)
            # 数据整理
            plot_data = OrderedDict()
            for i in range(len(data)):
                plot_data[roc_curve_name[i]] = data[i]  # data[i]是个tuple

            # 参数设置
            self._set_parameters(ax, fig_title)
            for i in range(len(roc_curve_name)):
                # 绘制ROC曲线
                fpr, tpr, ac = plot_data[roc_curve_name[i]]
                self._roc_curves_plot(ax, fpr, tpr, c_roc[i], roc_curve_name[i], ac)
                if i != len(roc_curve_name) - 1:
                    # 找到最靠近左上角的点并绘制
                    self._youden_point_plot(ax, fpr, tpr)
                else:
                    # 找到最靠近左上角的最后一个点并绘制
                    self._youden_point_plot(ax, fpr, tpr, label='Current classifier')
            # 绘制基线
            self._baseline_plot(ax, c_base)
            # 设置图例
            self._legend_plot(ax, font='Microsoft YaHei')
            # 保存图片
            self._save_fig()

    def _set_parameters(self, ax, fig_title):
        # 参数设置
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(which='both', linestyle='--', color='darkgrey', alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

        fontdict = {
            'family': 'Times New Roman',
            'weight': 'bold'}
        ax.set_xlabel('False Positive Rate', fontsize=12, fontdict=fontdict)
        ax.set_ylabel('True Positive Rate', fontsize=12, fontdict=fontdict)
        ax.set_title(fig_title, fontsize=14, fontdict=fontdict)

    def _roc_curve_plot(self, ax, fpr, tpr, c_roc):
        # 绘制ROC曲线
        param_dict = {
            'color': c_roc,
            'lw': 2,
            'marker': '.',
            'label': 'ROC curve',
        }
        ax.plot(fpr, tpr, **param_dict)

    def _roc_curves_plot(self, ax, fpr, tpr, c_roc, roc_curve_name, ac):
        # 绘制ROC曲线
        param_dict = {
            'color': c_roc,
            'lw': 2,
            'marker': '.',
            'label': roc_curve_name + f' (AUC = {ac:.2f})',
        }
        ax.plot(fpr, tpr, **param_dict)

    def _roc_area_plot(self, ax, fpr, tpr, c_area):
        # 绘制曲线下面积
        param_dict = {
            'facecolor': c_area,
            'alpha': 0.5,
            'label': 'Area under curve (AUC)',
        }
        ax.fill_between(fpr, tpr, 0, **param_dict)

    def _youden_point_plot(self, ax, fpr, tpr, label=None):
        # 找到最靠近左上角的点并绘制
        distance = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
        closest_point_idx = np.argmin(distance)
        self.closest_point = (fpr[closest_point_idx], tpr[closest_point_idx])
        param_dict = {
            'c': 'red',
            'label': label,
        }
        ax.scatter(self.closest_point[0], self.closest_point[1], **param_dict)

    def _youden_text_plot(self, ax):
        # 定义[约登文本框]的位置和内容
        param_dict = {
            's': f'({self.closest_point[0]:.2f}, {self.closest_point[1]:.2f})',
            'bbox': {
                'boxstyle': 'round',
                'alpha': 0
            },
            'fontdict': {
                'family': 'Times New Roman',
                'size': 12,
                'weight': 'bold',
                'horizontalalignment': 'center',
                'verticalalignment': 'center'
            }}
        # 在指定区域插入文本框
        ax.text(self.closest_point[0] + 0.1, self.closest_point[1] - 0.025, **param_dict)

    def _baseline_plot(self, ax, c_base):
        # 绘制辅助线
        param_dict = {
            'color': c_base,
            'lw': 2,
            'linestyle': ':',
            'alpha': 0.5,
            'label': 'Baseline',
        }
        ax.plot([0, 1], [0, 1], **param_dict)

    def _auc_text_plot(self, ax, ac, c_text_bk):
        # 定义[AUC文本框]的位置和内容
        param_dict = {
            's': f'Positive class: 1\nAUC: {ac:.2f}',
            'bbox': {
                'boxstyle': 'round',
                'alpha': 0.5,
                'facecolor': c_text_bk,
                'edgecolor': c_text_bk
            },
            'fontdict': {
                'family': 'Times New Roman',
                'size': 12,
                'weight': 'bold',
                'horizontalalignment': 'center',
                'verticalalignment': 'center'
            }}
        # 在指定区域插入文本框
        ax.text(0.6, 0.35, **param_dict)

    def _legend_plot(self, ax, font='Times New Roman'):
        # 设置图例
        param_dict = {
            'prop': {
                'family': font,
                'size': 10,
                'weight': 'bold'
            },
            'loc': 'lower right',
            'facecolor': 'white',
            'edgecolor': 'black',
            'framealpha': 0.5
        }
        ax.legend(**param_dict)

    def _save_fig(self):
        # 保存图片
        plt.savefig(f'../../aresources/{self.mode}_{self.fig_title}_{time.time():.0f}.png')
        plt.show()


if __name__ == '__main__':
    # file_path = '../../aresources/fans_data/data/Post_Zsocre_PCC_KW_8_AE_test.csv'
    # m = MyROCPlot(MyROCPlot.SingleMode, file_path)
    # m.single_plot('Label', 'Pred', 'SVM')

    # root = '../../aresources/my_data/plot_data'
    # file_path = [os.path.join(root, i) for i in os.listdir(root)]
    # m = MyROCPlot(MyROCPlot.MultipleMode, file_path)
    # m.multiple_plot('label', 'scores',
    #                 tuple([i.split('/')[-1].split('.')[0] for i in file_path]),
    #                 fig_title='ROC curves',
    #                 c_roc=('red', 'green', 'blue', 'yellow',
    #                        'orange', 'pink', 'purple', 'cyan', 'darkblue'),
    #                 )

    # file_path = '../../aresources/clinic_data.csv'
    # m = MyROCPlot(MyROCPlot.MultipleMode, file_path)
    # m.multiple_plot('组别',
    #                 ['长径变化率', '厚径变化率', 'PSV变化率', 'RI变化率', 'PRE_1', 'PRE_2', '降级修改'],
    #                 ('长径变化率', '厚径变化率', 'PSV变化率', 'RI变化率', 'PRE_1', 'PRE_2', '降级修改'),
    #                 fig_title='ROC curves',
    #                 c_roc=('red', 'green', 'blue', 'yellow', 'orange', 'pink', 'cyan'),
    #                 )

    file_path = '../../aresources/clinic_data.csv'
    m = MyROCPlot(MyROCPlot.MultipleMode, file_path)
    m.multiple_plot('组别',
                    ['长径变化率', '厚径变化率', 'PSV变化率', 'RI变化率', 'PRE_1', 'PRE_2', '降级修改'],
                    ('长径变化率', '厚径变化率', 'PSV变化率', 'RI变化率', 'PRE_1', 'PRE_2', '降级修改'),
                    fig_title='ROC curves',
                    c_roc=('red', 'green', 'blue', 'yellow', 'orange', 'pink', 'cyan'),
                    rol=3,
                    col=3)
