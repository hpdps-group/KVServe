import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 模拟数据 [Prefill, Comp, Trans, Decomp, Decode]
# 假设 Ours 的 Trans 非常短
data = {
    'Baseline': [10, 0, 100, 0, 20],
    'Method A': [10, 5, 60, 5, 20],
    'Method B': [10, 8, 45, 8, 20],
    'Ours':     [10, 12, 5, 12, 20]  # Trans 只有 5，Comp 增加了
}
methods = list(data.keys())
colors = ['#dddddd', '#4e79a7', '#f28e2b', '#76b7b2', '#dddddd'] 
# 颜色：Prefill(灰), Comp(蓝), Trans(橙), Decomp(青), Decode(灰)

fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(methods))
height = 0.5

# 记录 Trans 的坐标，用于画漏斗
trans_coords = {} 

# 1. 循环绘制堆叠图
for i, method in enumerate(methods):
    # 倒序排列，Baseline 在最上面 (y=0对应Ours, y=3对应Baseline，稍微反直觉，画图时翻转即可)
    y = len(methods) - 1 - i 
    values = data[method]
    
    left = 0
    for j, val in enumerate(values):
        ax.barh(y, val, left=left, height=height, color=colors[j], edgecolor='white')
        
        # 记录 Trans 的起止点 (索引2是Trans)
        if j == 2:
            trans_coords[method] = {'left': left, 'right': left + val, 'y': y}
        
        left += val

# 2. 绘制漏斗 (Funnel)
# 连接 Baseline (Top) 和 Ours (Bottom) 的 Trans 区域
top_method = 'Baseline'
bot_method = 'Ours'

# 定义多边形的四个顶点
points = [
    [trans_coords[top_method]['left'], trans_coords[top_method]['y'] + height/2],  # 左上
    [trans_coords[top_method]['right'], trans_coords[top_method]['y'] + height/2], # 右上
    [trans_coords[bot_method]['right'], trans_coords[bot_method]['y'] - height/2], # 右下
    [trans_coords[bot_method]['left'], trans_coords[bot_method]['y'] - height/2]   # 左下
]

# 添加半透明遮罩
polygon = patches.Polygon(points, closed=True, color=colors[2], alpha=0.15, linestyle='--')
ax.add_patch(polygon)

# 3. 辅助标注 (解决 Trans 看不见的问题)
# 在 Ours 的 Trans 位置加一个箭头和文字
ours_trans_center = trans_coords['Ours']['left'] + (trans_coords['Ours']['right'] - trans_coords['Ours']['left'])/2
ax.annotate('Trans: 5ms', 
            xy=(ours_trans_center, trans_coords['Ours']['y']), 
            xytext=(ours_trans_center, trans_coords['Ours']['y'] + 0.5),
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            ha='center')

# 美化
ax.set_yticks(np.arange(len(methods)))
ax.set_yticklabels(reversed(methods)) # 翻转标签以匹配视觉
ax.set_xlabel('Latency (ms)')
ax.set_title('Breakdown with "Funnel" highlighting Trans reduction')

# 手动添加图例 (略)
# plt.show()
plt.savefig('test.png')