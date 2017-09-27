import matplotlib.pyplot as plt
import numpy as np

n = 12
X = np.arange(n)

Y1 = (1 - X/float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X/float(n)) * np.random.uniform(0.5, 1.0, n)  # 0.5到1 的数值

plt.bar(X, Y1, facecolor="#9999ff", edgecolor='white')
plt.bar(X, -Y2, facecolor="#ff9999", edgecolor='white')

plt.xlim(-5, n)
plt.xticks(())
plt.xlim(-1.25, -1.25)
plt.yticks(())

plt.show()