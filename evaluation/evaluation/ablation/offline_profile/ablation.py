import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea, DrawingArea
import os

# ================= GLOBAL CONFIGURATION =================
OUTPUT_PDF = "ablation_study.pdf"
DATA_FILE = "data.csv"
FIGURE_SIZE = (5, 4)
GRID_ALPHA = 0.3
LINE_WIDTH = 2

# 颜色配置 (Removed, moved to STYLES)
# BAR_COLOR = '#1f77b4'      # 蓝色用于柱状图 (Iters)
# LINE_COLOR = '#d62728'     # 红色用于折线图 (Max CR)

# ================= STYLE CONFIGURATION =================
# 配置前四组和最后一组的样式
# 顺序对应: [w/o Exp, w/o Enc, w/o Prune, w/o Stop, Full]

STYLES = [
    # 1. w/o Exp
    {"bar_color": "#c7c7c7", "bar_edge": "black", "hatch": "..", "marker": "o", "markersize": 8, "alpha": 0.8},
    # 2. w/o Enc
    {"bar_color": "#c7c7c7", "bar_edge": "black", "hatch": "..", "marker": "o", "markersize": 8, "alpha": 0.8},
    # 3. w/o Prune
    {"bar_color": "#c7c7c7", "bar_edge": "black", "hatch": "..", "marker": "o", "markersize": 8, "alpha": 0.8},
    # 4. w/o Stop
    {"bar_color": "#c7c7c7", "bar_edge": "black", "hatch": "..", "marker": "o", "markersize": 8, "alpha": 0.8},
    # 5. Full (Highlights)
    {"bar_color": "#e41a1c", "bar_edge": "black", "hatch": "xx", "marker": "*", "markersize": 14, "alpha": 1.0},
]

LINE_COLOR = '#e41a1c' # Keep the red line for consistency

# 自定义刻度配置 (Custom Ticks)
# 左轴 (Max CR) 的刻度值，根据数据范围 8.5 - 9.3 设置
CUSTOM_Y_TICKS_LEFT = [0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4]
# 右轴 (Iterations) 的刻度值，根据数据范围 194 - 300 设置
CUSTOM_Y_TICKS_RIGHT = [150, 250, 300, 350]

# 自定义 X 轴标签 (如果要覆盖 data.csv 中的 method 名称，请在此填入字符串列表)
# 列表长度必须与数据点数量一致
CUSTOM_X_LABELS = ["w/o\nExp", "w/o\nEnc", "w/o\nPrune", "w/o\nStop", "Full"] 
# 例如: ["Base", "Opt1", "Opt2", "Opt3", "Final"]

# Matplotlib rcParams
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.5,
    "lines.linewidth": LINE_WIDTH,
    "grid.linestyle": "--",
    "grid.alpha": GRID_ALPHA,
    "xtick.direction": "out",  # 修改为朝外
    "ytick.direction": "out",  # 修改为朝外
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# ================= HELPERS =================

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def load_data(file_name):
    file_path = os.path.join(get_script_dir(), file_name)
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None, None, None
    
    methods = []
    max_crs = []
    iters = []
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            methods.append(row['method'])
            max_crs.append(float(row['max_cr']))
            iters.append(int(row['iters']))
            
    return methods, np.array(max_crs), np.array(iters)

def map_values_to_linear_ticks(values, ticks):
    """将真实值映射到基于刻度索引的线性空间"""
    ticks = np.array(ticks)
    # np.interp(x, xp, fp) -> 将 values (x) 映射到 [0, 1, 2, ...] (fp) 空间，基于 ticks (xp)
    return np.interp(values, ticks, np.arange(len(ticks)))

def create_legend_box(fig, handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.95), frameon=True, orientation='vertical'):
    """
    使用 offsetbox 构建单行居中图例 (参考 draw_bandwidth_latency.py)
    """
    def create_legend_item(handle, label):
        da = DrawingArea(width=15, height=10, xdescent=0, ydescent=0)
        
        if isinstance(handle, Line2D):
            line = Line2D([0, 11, 15], [5, 5, 5],
                          color=handle.get_color(),
                          linewidth=handle.get_linewidth(),
                          linestyle=handle.get_linestyle(),
                          marker=handle.get_marker(),
                          markersize=handle.get_markersize(),
                          markeredgecolor=handle.get_markeredgecolor(),
                          markeredgewidth=handle.get_markeredgewidth(),
                          markevery=None)
            # 对于 Line2D，设置 marker 在中间
            line.set_markevery([1]) 
            da.add_artist(line)
        elif isinstance(handle, (Patch, Rectangle)):
            # 对于柱状图 (Patch / Rectangle)
            # 确保获取颜色属性
            fc = handle.get_facecolor()
            ec = handle.get_edgecolor()
            lw = handle.get_linewidth()
            alpha = handle.get_alpha()
            hatch = handle.get_hatch()
            
            # 在 DrawingArea 中画一个小矩形
            # width=22, height=10. 画一个居中的矩形
            # rectangle (x, y), width, height
            r = Rectangle((0, 2), 15, 6, 
                          facecolor=fc,
                          edgecolor=ec,
                          hatch=hatch,
                          linewidth=lw if lw else 0,
                          alpha=alpha)
            da.add_artist(r)
        
        ta = TextArea(label, textprops=dict(color="black", size=10, family="DejaVu Sans", fontweight='bold'))
        return HPacker(children=[da, ta], align="center", pad=0, sep=5)

    # 构建 Items
    items = [create_legend_item(h, l) for h, l in zip(handles, labels)]
    
    if orientation == 'vertical':
        packer = VPacker(children=items, align="left", pad=0, sep=1)
    else:
        packer = HPacker(children=items, align="center", pad=0, sep=10) # sep是图例项之间的间距
    
    # 放入容器
    anchored_box = AnchoredOffsetbox(
        loc=loc,
        child=packer,
        pad=0,
        frameon=frameon,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=fig.transFigure,
        borderpad=0
    )
    
    # 设置边框样式
    if frameon:
        anchored_box.patch.set_boxstyle("round,pad=0.2")
        anchored_box.patch.set_linewidth(1.2)
        anchored_box.patch.set_edgecolor('black')
        anchored_box.patch.set_facecolor('white')
        anchored_box.patch.set_alpha(0.1) # 背景透明度
    
    return anchored_box

# ================= PLOTTING =================

def plot_ablation():
    methods, max_crs, iters = load_data(DATA_FILE)
    if methods is None:
        return

    # 将数据映射到自定义刻度的线性空间
    max_crs_mapped = map_values_to_linear_ticks(max_crs, CUSTOM_Y_TICKS_LEFT)
    iters_mapped = map_values_to_linear_ticks(iters, CUSTOM_Y_TICKS_RIGHT)

    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE)
    
    x_indices = np.arange(len(methods))
    bar_width = 0.5

    # --- 右轴：柱状图 (Iterations) ---
    ax2 = ax1.twinx()
    
    # 逐个绘制 Bar 以应用不同的样式
    bars = []
    for i in range(len(methods)):
        # 获取样式，如果数据点多于样式定义，默认使用最后一个
        style = STYLES[i] if i < len(STYLES) else STYLES[-1]
        
        bar = ax2.bar(x_indices[i], iters_mapped[i], width=bar_width, 
                      color=style['bar_color'], 
                      edgecolor=style['bar_edge'],
                      hatch=style['hatch'],
                      alpha=style['alpha'],
                      zorder=1)
        bars.append(bar[0]) # bar 返回 container，取第一个元素

    # 标注真实数值
    for bar, val in zip(bars, iters):
        height = bar.get_height()
        # 在映射空间的高度上加一点偏移
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{val}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

    # --- 左轴：折线图 (Max CR) ---
    # 1. 绘制连线 (统一颜色，无 Marker，底层)
    ax1.plot(x_indices, max_crs_mapped, color=LINE_COLOR, linestyle='-', linewidth=2, zorder=10)
    
    # 2. 绘制不同的 Marker (顶层)
    for i in range(len(methods)):
        style = STYLES[i] if i < len(STYLES) else STYLES[-1]
        # 绘制单个点
        ax1.plot(x_indices[i], max_crs_mapped[i], 
                 color=LINE_COLOR, 
                 marker=style['marker'], 
                 markersize=style['markersize'], 
                 markeredgecolor='white', # 可选：加个白边让点更清晰
                 markeredgewidth=0.5,
                 zorder=11)
    
    # 标注真实数值
    for x, y_mapped, y_real in zip(x_indices, max_crs_mapped, max_crs):
        ax1.text(x - 0.1, y_mapped + 0.2, f'{y_real:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color=LINE_COLOR)

    # --- 轴标签与刻度设置 ---
    
    # X 轴
    # ax1.set_xlabel("Ablation Settings", fontweight='bold', fontsize=13)
    ax1.set_xticks(x_indices)
    
    # 使用自定义刻度内容
    if CUSTOM_X_LABELS and len(CUSTOM_X_LABELS) == len(methods):
        ax1.set_xticklabels(CUSTOM_X_LABELS, rotation=0, fontweight='bold')
    else:
        ax1.set_xticklabels(methods, rotation=0)
        
    ax1.set_xlim(min(x_indices) - 0.6, max(x_indices) + 0.6)

    # 左 Y 轴 (Max CR) - 使用自定义刻度
    ax1.set_ylabel("Compression Ratio", fontweight='bold', fontsize=13)
    ax1.set_yticks(np.arange(len(CUSTOM_Y_TICKS_LEFT)))
    ax1.set_yticklabels([f"{y:.1f}" for y in CUSTOM_Y_TICKS_LEFT], fontsize=11)
    ax1.tick_params(axis='y')
    # 设置显示范围对应刻度索引范围
    ax1.set_ylim(0, len(CUSTOM_Y_TICKS_LEFT) - 1)
    
    # 右 Y 轴 (Iterations) - 使用自定义刻度
    ax2.set_ylabel("Iterations", fontweight='bold', fontsize=13, rotation=270, labelpad=15)
    ax2.set_yticks(np.arange(len(CUSTOM_Y_TICKS_RIGHT)))
    ax2.set_yticklabels([str(y) for y in CUSTOM_Y_TICKS_RIGHT], fontsize=11)
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, len(CUSTOM_Y_TICKS_RIGHT) - 1)

    # --- 样式调整 ---
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # 网格线 (仅基于左轴画横线)
    ax1.grid(True, axis='y', linestyle='--', alpha=GRID_ALPHA)
    
    # --- 图例 (修改为使用 create_legend_box) ---
    
    # 手动创建图例 handles
    legend_handles = []
    legend_labels = []
    
    # 1. Max CR (Red Line)
    proxy_line = Line2D([0], [0], color=LINE_COLOR, markersize=8, linestyle='-')
    legend_handles.append(proxy_line)
    legend_labels.append("Max CR")

    # 2. Iterations (w/o) - 灰色样式
    style_wo = STYLES[0]
    proxy_rect_wo = Rectangle((0, 0), 1, 1, 
                           facecolor=style_wo['bar_color'], 
                           edgecolor=style_wo['bar_edge'],
                           hatch=style_wo['hatch'],
                           alpha=style_wo['alpha'])
    legend_handles.append(proxy_rect_wo)
    legend_labels.append("Iters (w/o)")

    # 3. Iterations (Full) - 红色样式
    style_full = STYLES[-1]
    proxy_rect_full = Rectangle((0, 0), 1, 1, 
                           facecolor=style_full['bar_color'], 
                           edgecolor=style_full['bar_edge'],
                           hatch=style_full['hatch'],
                           alpha=style_full['alpha'])
    legend_handles.append(proxy_rect_full)
    legend_labels.append("Iters (Full)")
    
    # 调整位置，例如放在顶部中间
    # 增加 bbox_to_anchor 的宽度适应 3 个图例
    legend_box = create_legend_box(fig, legend_handles, legend_labels, 
                                   loc='upper center', bbox_to_anchor=(0.27, 0.87), frameon=True)
    fig.add_artist(legend_box)

    # 保存
    plt.tight_layout()
    # 调整顶部边距以容纳图例
    plt.subplots_adjust(top=0.9)
    output_path = os.path.join(get_script_dir(), OUTPUT_PDF)
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    plot_ablation()
