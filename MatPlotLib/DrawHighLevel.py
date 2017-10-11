# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f(x,y):
    return (1-x/2 + x**5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

n = 256

x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X,Y = np.meshgrid(x,y)

plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap=plt.cm.hot)  # 这里的8 代表分段数
# plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap=plt.cm.cool)

C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=5)
plt.clabel(C, inline=True, fontsize=5)

plt.xticks(())
plt.yticks(())
plt.show()