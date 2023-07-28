# import matplotlib.pyplot as plt
#
# # 绘制线条1
# x1 = [1, 2, 3, 4, 5]
# y1 = [2, 4, 6, 8, 10]
# line1, = plt.plot(x1, y1, label='line1')
#
# # 绘制线条2
# x2 = [1, 2, 3, 4, 5]
# y2 = [5, 4, 3, 2, 1]
# line2, = plt.plot(x2, y2, label='line2')
#
#

# ax = plt.gca()
# handles, labels = ax.get_legend_handles_labels()
# print(handles, labels)
#
# custom_legend = plt.legend([tx2, tx1], ['line1 and line2', ''], loc='upper left')
#
#
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Custom Legend Example')
# plt.grid(True)
# plt.show()


# import matplotlib.pyplot as plt
# from matplotlib.legend_handler import HandlerTuple
#
# fig, ax = plt.subplots()
# p3 = plt.scatter(0.75, 1, marker="*", color='blue', label='star2')
# p4 = plt.scatter(1.75, 1, marker="*", color='red', label='star1')
# p1, = ax.plot([1, 2.5, 3], 'r-d')
# p2, = ax.plot([3, 2, 1], 'k-o')
#
# l = ax.legend([p1, p2, (p3, p4)],
#               ['tt', 'yy', 'Two keys'],
#               numpoints=1,
#               handler_map={tuple: HandlerTuple(ndivide=None)}
#               )
#
# plt.show()


import matplotlib.pyplot as plt

# 设置Matplotlib使用LaTeX解析器
# plt.rcParams['text.usetex'] = True

plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 设置标题，使用斜体P
plt.title(r'This is an example with italic $P$')

plt.grid(True)
plt.show()

