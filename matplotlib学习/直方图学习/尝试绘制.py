import matplotlib.pyplot as plt
import numpy as np


def draw_grouped_histogram(species: list, data: dict):
    plt.rcParams['mathtext.default'] = 'regular'

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
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

    # 设置坐标轴数字为Times New Roman字体
    text = ax.get_yticklabels()
    for tx in text:
        tx.set_fontproperties('Times New Roman')

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper left', ncols=1, prop={
        'family': 'Times New Roman',
        'size': 10,
        'weight': 'bold'
    })

    plt.show()


if __name__ == '__main__':
    species = ['mean', 'Z-Score']
    data = {
        'cv_train': (0.5050, 0.5715),
        'cv_val': (0.6318, 0.4825),
        'test': (0.5016, 0.6377),
    }
    draw_grouped_histogram(species, data)
