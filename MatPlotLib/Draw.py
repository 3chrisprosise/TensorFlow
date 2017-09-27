import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.show()
x = np.linspace(-3, 3, 50)  #  -1 到1 50 个点
y1 = 2*x + 1
y2 = x **2
plt.figure(num=3)  # 重名的图会导致图像被覆盖，可以用作动画？
plt.plot(x,y1)
plt.figure(num=3,figsize=(8,5))  # figsize设置图片长宽

plt.xlim((-1,2))  # 设置x，y轴范围
plt.ylim((-1,3))

plt.xlabel("hx")  # 设置x，y轴标签
plt.ylabel("hy")

new_ticks = np.linspace(-1,2,5)  # -1 到 2  分 5 个点
print(new_ticks)
plt.xticks(new_ticks)

plt.yticks([-2, -1.8, -1, 1.2, 2.3],
           ['1\ $','2\ $','3\ $','4\ $','5\ $'])  # 更改y轴坐标的显示方式  空格要进行转义,符合正则表达式

# gca = 'get current axis'

ax = plt.gca()   # 获取图片的坐标轴

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')  # 隐藏右方和上方的坐标轴

ax.xaxis.set_ticks_position('bottom')  # 指明刻度的标注位置
ax.yaxis.set_ticks_position('left')

ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))  # 设置坐标轴位置 ，将坐标轴移动到(0,0)坐标位置


for label in ax.get_xticklabels() + ax.get_yticklabels():  # 更改X,Y轴图标样式
    label.set_fontsize(5)
    label.set_bbox(dict(
        facecolor='white', edgecolor='None', alpha=0.7
    ))

plt.plot(x, y1)
plt.plot(x, y2, color='red', linewidth=5, linestyle='-')  # 设置线宽，线型
plt.show()


