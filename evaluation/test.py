import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

# --- 1. 设置学术风格 (类似 LaTeX) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # 使用衬线字体
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,        # 高分辨率
    'lines.linewidth': 2.5,
    'lines.markersize': 10
})

# --- 2. 定义数据 ---
# 阶段描述 (X轴)
stages = [
    "Baseline", 
    "+ Trans.\nMethods", 
    "+ Trans.\nParams", 
    "+ Quant.\nMethods", 
    "+ Quant.\nParams", 
    "+ Codec\nMethods"
]

# 搜索空间大小估算 (Y轴数据)
# 逻辑推演:
# 1. Baseline: 1 (无操作)
# 2. + Transformer Algos: 3 (Affine, Hadamard, Delta) -> 3
# 3. + Transformer Params: 3 * ~5 (Rotation angles etc.) -> 15
# 4. + Quantizer Algos: 15 * 2 (Mixed, Various Dim) -> 30
# 5. + Quantizer Params: 30 * 225 (Bit-width maps) -> 6750 (爆炸点!)
# 6. + Codec Algos: 6750 * 2 (ANS, Packing) -> 13500
search_space_sizes = [1, 3, 15, 30, 6750, 13500]

x = np.arange(len(stages))

# --- 3. 绘图 ---
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制主曲线
# 使用 's-' (方形点 + 实线), 颜色选用深蓝色
ax.plot(x, search_space_sizes, 's-', color='#003366', label='Search Space Size', clip_on=False)

# --- 4. 关键区域高亮与标注 ---

# 设置 Y 轴为对数坐标，否则看不出爆炸效果
ax.set_yscale('log')

# 标注 "Explosion" 区域
# 在 "Quant. Params" 这一步发生了巨大的跳跃
ax.annotate(
    'Combinatorial Explosion\n(Parameters)', 
    xy=(4, 6750), xycoords='data',
    xytext=(1.5, 4000), textcoords='data',
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='#cc0000', lw=2),
    fontsize=14, color='#cc0000', fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cc0000", alpha=0.9)
)

# 辅助虚线：标示出量化参数这一步带来的增量
ax.vlines(x=4, ymin=1, ymax=6750, colors='gray', linestyles='--', alpha=0.5)

# --- 5. 坐标轴美化 ---

# X轴设置
ax.set_xticks(x)
ax.set_xticklabels(stages, rotation=0) # 如果文字太长可以设为 30
ax.set_xlabel("Pipeline Configuration Depth (Complexity)", fontweight='bold', labelpad=15)

# Y轴设置
ax.set_ylabel("Total Configurations (Log Scale)", fontweight='bold')
ax.set_ylim(0.8, 100000) # 稍微留一点头部空间

# 网格线 (仅 Y 轴需要，方便看数量级)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 移除上方和右侧的边框 (Spines)，更符合现代学术风格
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- 6. 保存与显示 ---
plt.tight_layout()
plt.savefig("search_space_explosion.pdf", bbox_inches='tight') # 建议保存为 PDF 矢量图
plt.show()