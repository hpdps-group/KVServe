import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})
# --- 1. 数据准备 ---
data_raw = [
    # gsm8k, humaneval, multi_news, qasper
    ["84.91/1.00", "66.46/1.00", "26.87/1.00", "45.66/1.00"], # Default
    ["83.47/5.85", "62.80/3.98", "26.08/6.20", "44.46/6.65"], # Cachegen
    ["81.35/4.32", "60.37/2.61", "26.44/4.45", "44.97/4.89"], # KIVI
    ["83.93/2.16", "65.24/1.06", "23.41/2.89", "43.90/3.77"], # DuoAttn
    ["76.99/6.19", "59.76/5.36", "26.47/6.05", "44.83/6.60"]  # KVServe
]
rows = ["Default", "Cachegen", "KIVI", "DuoAttn", "MixHQ"]
cols = ["GSM8K", "HumanEval", "Multi-News", "Qasper"]

acc_data = []
comp_data = []
for r in data_raw:
    acc_row = []
    comp_row = []
    for cell in r:
        a, c = map(float, cell.split('/'))
        acc_row.append(a)
        comp_row.append(c)
    acc_data.append(acc_row)
    comp_data.append(comp_row)

df_acc_raw = pd.DataFrame(acc_data, index=rows, columns=cols)
df_comp_raw = pd.DataFrame(comp_data, index=rows, columns=cols)

# --- 2. 排名映射计算 (Score Calculation) ---

def calculate_diverging_score(df_raw, metric_type='acc', default_label='Default'):
    """
    计算用于绘图的分数。
    metric_type='acc': 正值 (红色), Default=0.
    metric_type='comp': 负值 (绿色), Default=0.
    逻辑: N个非Default数据, 分成 N+1 段.
    Rank 1 (Worst) -> 1/(N+1)
    Rank N (Best)  -> N/(N+1)
    """
    plot_df = df_raw.copy().astype(float)
    rows_idx = plot_df.index
    cols_idx = plot_df.columns

    actual_default_label = default_label
    if default_label not in rows_idx:
        actual_default_label = rows_idx[0]

    for col in cols_idx:
        # 1. 强制 Default 为 0
        plot_df.loc[actual_default_label, col] = 0.0

        # 2. 处理其他行
        other_rows = rows_idx[rows_idx != actual_default_label]
        if len(other_rows) > 0:
            other_vals = df_raw.loc[other_rows, col]
            
            # 计算排名 (从小到大, 1开始)
            ranks = other_vals.rank(ascending=True, method='min')
            
            # 计算总的非Default数量
            N = len(other_rows)
            # 使用 ranks / (N + 1) 进行分段映射
            # 例如 N=4: 分数分别为 0.2, 0.4, 0.6, 0.8
            scores = ranks / (N + 1)
            
            if metric_type == 'comp':
                # 压缩率用负值 (绿色)
                scores = -scores
            
            plot_df.loc[other_rows, col] = scores
            
    return plot_df

# 计算绘图数据
# Acc: 正值 (0 ~ 1) -> 红色区域
plot_df_acc = calculate_diverging_score(df_acc_raw, metric_type='acc')
# Comp: 负值 (-1 ~ 0) -> 绿色区域
plot_df_comp = calculate_diverging_score(df_comp_raw, metric_type='comp')

# --- 3. 绘图设置 ---
sns.set_theme(style="white", font_scale=1.0)

# 定义统一的发散色谱：绿色 (-1) -> 灰色 (0) -> 红色 (1)
color_green = "#1a9850"
color_gray = "#f7f7f7"
color_red = "#e46a61"

cmap_diverging = mcolors.LinearSegmentedColormap.from_list(
    "GreenGrayRed", [color_green, color_gray, color_red]
)

# 设定统一范围 [-1, 1]
vmin = -1.0
vmax = 1.0

# 创建画布
fig = plt.figure(figsize=(13, 6))
gs_outer = GridSpec(1, 2, width_ratios=[2, 0.03], figure=fig, wspace=0.05)
gs_inner = GridSpecFromSubplotSpec(1, 2, width_ratios=[1, 0.8], subplot_spec=gs_outer[0, 0], wspace=-0.08)
ax1 = fig.add_subplot(gs_inner[0, 0])
ax2 = fig.add_subplot(gs_inner[0, 1], sharey=ax1)
# 手动创建 colorbar axes，增加高度
cbar_ax_temp = fig.add_subplot(gs_outer[0, 1])
cbar_pos = cbar_ax_temp.get_position()
cbar_ax_temp.remove()  # 移除临时 axes
# 增加 colorbar 的高度：保持宽度不变，但增加高度（向上和向下扩展）
height_scale = 1.1  # 高度缩放因子，可以调整这个值来改变高度
new_height = cbar_pos.height * height_scale
new_y0 = cbar_pos.y0 - (new_height - cbar_pos.height) / 2
cbar_ax = fig.add_axes([cbar_pos.x0, new_y0, cbar_pos.width, new_height])

# --- 4. 自定义标注函数 ---
def add_bold_annotations(ax, df_data):
    rows_idx = df_data.index
    cols_idx = df_data.columns
    for i, row_label in enumerate(rows_idx):
        for j, col_label in enumerate(cols_idx):
            val = df_data.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}",
                    ha="center", va="center", color="black", weight="bold", fontsize=16)

# --- 5. 绘制热力图 ---

# 左图: Accuracy (数据为正，显示为红色系)
sns.heatmap(plot_df_acc, ax=ax1, cmap=cmap_diverging, vmin=vmin, vmax=vmax, center=0,
            annot=False, cbar=False, square=False, linewidths=1.5, linecolor='white')
ax1.set_aspect(1.0 / 1.25)
add_bold_annotations(ax1, df_acc_raw)
ax1.set_title("Accuracy", fontsize=20, weight='bold', pad=15)
ax1.set_ylabel("Cache Type", fontsize=18, weight='bold', labelpad=15)
ax1.set_xlabel("")
ax1.tick_params(labelsize=13.5)

# 右图: Compression (数据为负，显示为绿色系)
# 我们传入完整的 vmin/vmax，确保它使用整个色谱的绿色部分
hm_ax2 = sns.heatmap(plot_df_comp, ax=ax2, cmap=cmap_diverging, vmin=vmin, vmax=vmax, center=0,
            annot=False, cbar=True, cbar_ax=cbar_ax, 
            square=False, linewidths=1.5, linecolor='white')
ax2.set_aspect(1.0 / 1.25)
add_bold_annotations(ax2, df_comp_raw)
ax2.set_title("Compression Ratio", fontsize=20, weight='bold', pad=15)
ax2.set_ylabel("")
ax2.set_xlabel("")
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.tick_params(labelsize=13.5)

# --- Colorbar 设置 ---
# 这是一个统一的 colorbar：-1 (绿) ... 0 (灰) ... 1 (红)
cbar = hm_ax2.collections[0].colorbar
# 设置刻度，对应 最绿(Best CR)，中间(Default)，最红(Best Acc)
cbar.set_ticks([-0.9, 0.9]) 
cbar.set_ticklabels(['Best\nCR', 'Best\nAcc'])
cbar_ax.tick_params(labelsize=15, length=0)  # length=0 隐藏刻度线
cbar_ax.set_ylabel("Metric Performance Rank", rotation=270, labelpad=10, weight='bold', fontsize=18)

# 布局
fig.text(0.52, 0.03, 'Dataset', ha='center', fontsize=20, weight='bold')
plt.subplots_adjust(bottom=0.15, right=0.9)
filename = "heatmap.pdf"
plt.savefig(filename, bbox_inches='tight')
print(f"Generated: {filename}")
plt.show()