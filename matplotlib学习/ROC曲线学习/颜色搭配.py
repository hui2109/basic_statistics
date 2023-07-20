# 使用字符串指定颜色映射
# x = np.random.rand(10)
# y = np.random.rand(10)
# colors = np.random.rand(10)
# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.scatter(x, y, c=colors, cmap='viridis')
#
# # 使用Colormap对象指定颜色映射
# cmap = cm.get_cmap('cool')
# ax2.plot(x, y, c=colors, cmap=cmap)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 安装Seaborn库（如果尚未安装）
# !pip install seaborn

# 生成随机数据
fpr = np.linspace(0, 1, 100)
tpr = np.random.uniform(0.5, 1, 100)

# 设置Seaborn绘图风格
sns.set()

# 创建一个新的图形窗口
plt.figure()

# 绘制ROC曲线
plt.plot(fpr, tpr)

# 设置图形标题和坐标轴标签
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# 显示图形
plt.show()


