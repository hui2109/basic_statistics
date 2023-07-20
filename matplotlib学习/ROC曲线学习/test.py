import matplotlib.pyplot as plt

figure, ax = plt.subplots(figsize=(6, 6), ncols=3, nrows=3)
ax[0, 0].plot([1, 2, 3], [1, 2, 3])

print(plt.show())