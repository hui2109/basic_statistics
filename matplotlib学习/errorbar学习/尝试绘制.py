import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
x = np.arange(1, 11)
auc_values = np.random.rand(10)  # 随机生成10个AUC值
lower_limits = auc_values - np.random.rand(10) * 0.2  # 随机生成10个置信区间下限
upper_limits = auc_values + np.random.rand(10) * 0.2  # 随机生成10个置信区间上限

# 计算误差条长度
yerr = np.vstack((auc_values - lower_limits, upper_limits - auc_values))

# 绘制误差图
plt.errorbar(x, auc_values, yerr=yerr, fmt='o', capsize=3)

# 绘制折线图
# plt.plot(x, auc_values, marker='o')

# 设置图形标题和坐标轴标签
plt.title('AUC with Confidence Intervals')
plt.xlabel('X')
plt.ylabel('AUC')

# 显示图形
plt.show()


