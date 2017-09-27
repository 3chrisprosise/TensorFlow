import matplotlib.pyplot as plt
import numpy as np

n = 1024
X = np.random.normal(0, 1, 1024)
Y = np.random.normal(0, 1, 1024) # 均值0 方差1 1024 个点

T = np.arctan2(Y, X)

plt.scatter(X, Y, s=75, c=T, alpha=0.5)  # s means SIZE  alpha  透明度

plt.xlim((-1.5, 1.5))
plt.ylim((-1.5, 1.5))

plt.xticks(())
plt.yticks(())  # 坐标设置，传入参数为tuple

plt.show()


