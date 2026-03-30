import matplotlib.pyplot as plt
import numpy as np

# 1. 数据准备
platforms = ['A10', 'V100', 'T4', 'L4', 'A100']
prefill_time = np.array([25, 20, 35, 30, 15])
comm_time = np.array([20, 40, 20, 20, 5])
decode_time = np.array([55, 40, 45, 50, 80])

# 计算总和用于归一化
totals = prefill_time + comm_time + decode_time

# 2. 颜色策略 (关键：同色系表达层级关系)
# Compute 类：使用蓝色系
c_compute_main = '#4e79a7'  # 内环：Compute 总颜色
c_prefill = '#a0cbe8'       # 外环：Prefill (浅蓝)
c_decode = '#59a14f'        # 外环：Decode (绿色) -> 或者用深蓝 '#4e79a7' 保持单色系
# 这里我建议 Decode 用绿色，以区分 Prefill，但在逻辑上它们被内环的蓝色包围

# Comm 类：使用橙红色系
c_comm_main = '#f28e2b'     # 内环：Comm 总颜色
c_comm_sub = '#ffbe7d'      # 外环：KV Transfer (浅橙)

# 3. 绘图设置
fig, axs = plt.subplots(1, 5, figsize=(15, 3.5)) # 1行5列
plt.rcParams['font.family'] = 'serif'

for i, ax in enumerate(axs):
    # 当前平台的数据
    p = prefill_time[i]
    d = decode_time[i]
    c = comm_time[i]
    total = totals[i]
    
    # --- 构建数据 ---
    # 外环数据 (细节): [Prefill, Decode, Comm]
    # 注意：顺序很重要，要和内环对应
    # 为了让 Compute 在一起，我们将顺序设为: Prefill, Decode, Comm
    outer_sizes = [p, d, c]
    outer_colors = [c_prefill, '#76b7b2', c_comm_sub] # Decode换了个青色区分
    outer_labels = ['Prefill', 'Decode', 'Comm']
    
    # 内环数据 (宏观): [Compute总和, Comm总和]
    # Compute总和 = Prefill + Decode
    inner_sizes = [p + d, c]
    inner_colors = [c_compute_main, c_comm_main]
    
    # --- 绘制外环 (Detail) ---
    # radius=1.0, width=0.3 (即从 0.7 到 1.0)
    wedges_out, texts_out = ax.pie(outer_sizes, radius=1.0, colors=outer_colors, 
                                   startangle=90, counterclock=False,
                                   wedgeprops=dict(width=0.3, edgecolor='w'))
    
    # --- 绘制内环 (Category) ---
    # radius=0.7, width=0.3 (即从 0.4 到 0.7)
    wedges_in, texts_in = ax.pie(inner_sizes, radius=0.7, colors=inner_colors, 
                                 startangle=90, counterclock=False,
                                 wedgeprops=dict(width=0.3, edgecolor='w'))
    
    # --- 中心文字 (硬件名称) ---
    ax.text(0, 0, platforms[i], ha='center', va='center', fontsize=12, fontweight='bold')
    
    # --- 智能标注 (百分比) ---
    # 只在切片够大时显示百分比，避免拥挤
    
    # 内环标注 (Compute vs Comm)
    if c / total > 0.15: # 如果Comm占比大于15%，在内环显示
        # 计算角度位置... 这里简化处理，直接用图例
        pass 

# 4. 全局图例 (Legend)
# 创建自定义图例句柄
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=c_compute_main, label='Compute (Aggregated)'),
    Patch(facecolor=c_comm_main, label='Communication (Aggregated)'),
    Patch(facecolor=c_prefill, label='Prefill Phase'),
    Patch(facecolor='#76b7b2', label='Decode Phase'),
    Patch(facecolor=c_comm_sub, label='KV Transfer'),
]

# 将图例放在正下方
fig.legend(handles=legend_elements, loc='lower center', 
           bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False, fontsize=11)

plt.tight_layout()
# plt.show()
plt.savefig('test.pdf', bbox_inches='tight')