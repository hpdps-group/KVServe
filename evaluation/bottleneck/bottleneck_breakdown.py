import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ================= 配置参数 (参考 bandwidth_latency_breakdown.py) =================
plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
})

# 纹理样式
HATCH_PREFILL = '/'
HATCH_COMM = '..'
HATCH_DECODE = '\\'

# 颜色配置 (参考参考代码的色板)
COLOR_PREFILL = '#FFBB78'  # Grey-ish
COLOR_COMM = '#e41a1c'     # Purple-ish
COLOR_DECODE = '#C5B0D5'   # Grey-ish

COLORS = [COLOR_PREFILL, COLOR_COMM, COLOR_DECODE]
HATCHES = [HATCH_PREFILL, HATCH_COMM, HATCH_DECODE]
LABELS = ['Prefill', 'Transmission', 'Decode']
COLUMNS = ['Prefill', 'Transmission', 'Decode']

# 纵向柱子间距控制参数 (1.0 为默认间距，小于1.0更紧凑，大于1.0更稀疏)
Y_AXIS_SPACING = 0.65 
BAR_HEIGHT = 0.45

# 最小显示宽度 (5%)
MIN_DISPLAY_WIDTH = 0.01

import matplotlib.ticker as mtick

def adjust_ratios_for_min_width(ratios, min_width):
    """
    调整比例列表，确保非零元素的最小宽度为 min_width。
    不足的部分由其他非零且不需要拉伸的阶段平分扣除。
    """
    ratios = np.array(ratios)
    # 找出需要拉伸的阶段 (0 < ratio < min_width)
    small_mask = (ratios > 0) & (ratios < min_width)
    
    if not np.any(small_mask):
        return ratios

    # 计算总共缺少的宽度
    current_small_vals = ratios[small_mask]
    needed_total = np.sum(min_width - current_small_vals)
    
    # 找出可以贡献宽度的阶段 (即不需要拉伸且大于0的阶段)
    donor_mask = (~small_mask) & (ratios > 0)
    
    if not np.any(donor_mask):
        # 如果没有可以贡献的阶段，直接返回（或者只能归一化，这里保持原样）
        return ratios
        
    # 计算每个贡献者需要减少的量 (平分)
    num_donors = np.sum(donor_mask)
    reduction_per_donor = needed_total / num_donors
    
    new_ratios = ratios.copy()
    
    # 将小阶段设置为最小宽度
    new_ratios[small_mask] = min_width
    
    # 减少贡献者的宽度
    new_ratios[donor_mask] -= reduction_per_donor
    
    # 防止减过头变成负数 (简单截断为0，虽然理论上应该递归调整，但在当前场景下通常够用)
    new_ratios[new_ratios < 0] = 0
    
    return new_ratios

def plot_panel(ax, csv_path, title):
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 为了让表格第一行显示在最上面，我们需要反转数据顺序（因为barh是从下往上画的）
    df = df.iloc[::-1].reset_index(drop=True)
    
    devices = df['Device'].tolist()
    # 应用间距参数
    y_pos = np.arange(len(devices)) * Y_AXIS_SPACING
    
    # 计算总数值用于归一化
    total_latency = df[COLUMNS].sum(axis=1).values
    
    # --- 预计算和调整比例 ---
    # 1. 计算原始比例矩阵 (Rows: Devices, Cols: Stages)
    raw_ratios_list = []
    for col in COLUMNS:
        raw_ratios_list.append(df[col].values / total_latency)
    # 转置为 (Devices, Stages)
    ratios_matrix = np.array(raw_ratios_list).T
    
    # 2. 对每一行(每个设备)进行调整
    adjusted_ratios_matrix = np.zeros_like(ratios_matrix)
    for i in range(ratios_matrix.shape[0]):
        adjusted_ratios_matrix[i] = adjust_ratios_for_min_width(ratios_matrix[i], MIN_DISPLAY_WIDTH)
    
    # --- 绘图 ---
    # 初始化 left 偏移量
    left_offsets = np.zeros(len(devices))
    
    # 循环绘制每一层 (Prefill, Communication, Decode)
    for i, col in enumerate(COLUMNS):
        # 使用调整后的比例进行绘图
        ratios = adjusted_ratios_matrix[:, i]
        # 使用原始比例进行文本显示
        orig_ratios = ratios_matrix[:, i]
        
        ax.barh(y_pos, ratios, left=left_offsets, height=BAR_HEIGHT,
                color=COLORS[i], edgecolor='black', hatch=HATCHES[i], 
                linewidth=1.5, zorder=3, label=LABELS[i] if i < 3 else "")
        
        # 添加内部百分比标签 (带边框矩形)
        for j, (ratio, orig_ratio) in enumerate(zip(ratios, orig_ratios)):
            # 只有当原始比例大于0时才显示 (或者设置一个很小的阈值)
            if ratio > 0.005 and col == 'Transmission': 
                # Determine colors based on component
                text_color = 'white' if i == 1 else 'black'
                edge_color = 'white' if i == 1 else 'black'

                # 文本显示原始比例
                pct_text = f"{orig_ratio*100:.0f}"
                if pct_text == "0" and orig_ratio > 0:
                     pct_text = "<1"

                ax.text(left_offsets[j] + ratio/2, y_pos[j], pct_text, 
                        ha='center', va='center', fontsize=11, fontweight='bold', color=text_color,
                        bbox=dict(facecolor=COLORS[i], edgecolor=edge_color, boxstyle='round,pad=0.3', alpha=1.0),
                        zorder=4)
        
        left_offsets += ratios

    # 设置轴标签和样式
    ax.set_yticks(y_pos)
    # 只有左侧图 (title包含Qasper) 显示Y轴标签，或者通过参数控制
    if "Qasper" in title:
        ax.set_yticklabels(devices, fontsize=12, fontweight='bold')
    else:
        ax.set_yticklabels([]) # 隐藏右侧图的Y轴标签
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10, loc='center', y=0.95, x=0.5)
    
    # 调整Y轴范围以适应新的间距
    margin = BAR_HEIGHT
    if len(y_pos) > 0:
        ax.set_ylim(min(y_pos) - 0.35, max(y_pos) + 0.35)
    
    # 样式调整：去边框，加网格
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
    
    # 设置X轴为百分比格式，但不显示%号，数值为 0-100
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{int(x*100)}'))

def main():
    # 文件路径
    file_qasper = "qasper.csv"
    file_wiki = "2wikimqa.csv"
    
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    
    # 绘制左图 (Qasper)
    plot_panel(ax1, file_qasper, "Qasper")
    ax1.title.set_fontsize(18)
    ax1.tick_params(axis='y', labelsize=15)

    # 绘制右图 (2WikiMQA)
    plot_panel(ax2, file_wiki, "2WikiMQA")
    ax2.title.set_fontsize(18)
    ax2.tick_params(axis='y', labelsize=15)
    
    # 设置公共 X 轴标签
    fig.supxlabel("Percentage of Total Latency (%)", fontweight='bold', fontsize=18, y=-0.02, x=0.55)
    
    # 创建图例 (手动创建 Patch 以保证样式一致)
    legend_handles = [
        Patch(facecolor=COLORS[0], edgecolor='black', hatch=HATCHES[0], label='Prefill'),
        Patch(facecolor=COLORS[1], edgecolor='black', hatch=HATCHES[1], label='Communication'),
        Patch(facecolor=COLORS[2], edgecolor='black', hatch=HATCHES[2], label='Decode')
    ]
    
    # 将图例放在顶部居中
    leg = fig.legend(handles=legend_handles, loc='upper center', 
               bbox_to_anchor=(0.54, 0.98), ncol=3, 
               frameon=True, prop={'size': 13, 'weight': 'bold'}, fancybox=True, borderpad=0.05)
    
    # 设置图例边框样式 (类似小矩形的圆角)
    leg.get_frame().set_boxstyle("round,pad=0.2")
    leg.get_frame().set_linewidth(1.2)
    leg.get_frame().set_edgecolor("black")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.75, bottom=0.18, wspace=0.08) 
    # 保存图片
    output_filename = "bottleneck_breakdown.pdf"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Plot generated: {output_filename}")

if __name__ == "__main__":
    main()
