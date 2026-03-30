import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea, DrawingArea
from matplotlib.patches import Rectangle
import csv
import os

# ================= 配置参数 (参考 style) =================
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

# ================= 配置参数 (自定义) =================
LINE_WIDTH = 3  # 折线图线宽

# ================= 1. 数据准备区域 (从CSV读取) =================

def load_data_from_csv(file_path):
    data_list = []
    if not os.path.exists(file_path):
        # Fallback for relative path if script is run from different dir
        file_path = os.path.join(os.path.dirname(__file__), "device.csv")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_list.append(row)
            
    # 提取 X_LABELS (保持出现顺序且去重)
    x_labels = []
    for row in data_list:
        if row['X'] not in x_labels:
            x_labels.append(row['X'])
            
    # 提取 METHODS (保持出现顺序且去重)
    methods = []
    for row in data_list:
        if row['Method'] not in methods:
            methods.append(row['Method'])
            
    # 构建 DATA_MS
    data_ms = {m: [] for m in methods}
    for method in methods:
        for x_label in x_labels:
            # 查找对应的值
            val = 0.0
            found = False
            for row in data_list:
                if row['X'] == x_label and row['Method'] == method:
                    val = float(row['Y'])
                    found = True
                    break
            if not found:
                print(f"Warning: Missing data for {x_label} - {method}")
            data_ms[method].append(val)
            
    return x_labels, methods, data_ms

# CSV 文件路径
CSV_PATH = "/root/workspace/KVServe/evaluation/evaluation/jct_latency/device.csv"

# 加载数据
X_LABELS, METHODS, DATA_MS = load_data_from_csv(CSV_PATH)

print("Loaded X_LABELS:", X_LABELS)
print("Loaded METHODS:", METHODS)
print("Loaded DATA:", DATA_MS)

# 颜色池 (按顺序分配)
# 对应: Default(BF16), CacheGen, KIVI, KVServe
METHOD_COLORS = ['#868686', '#FFBB78', '#C5B0D5', '#e41a1c', '#9575cd', '#4dd0e1'] 
# 截取需要的颜色数量
current_colors = METHOD_COLORS[:len(METHODS)]
COLOR_MAP = {name: col for name, col in zip(METHODS, current_colors)}

# Marker 池 (用于折线图)
MARKERS = ['o', 's', 'p', 'h', 'v', '<', '>']
MARKER_MAP = {name: m for name, m in zip(METHODS, MARKERS)}

LINES = ['--', '--', '--', '-']
LINES_MAP = {name: l for name, l in zip(METHODS, LINES)}

# Hatch 池 (用于柱状图)
# 可选样式: '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
HATCHES = ['..', '//', '\\\\', 'xx', '..', '++']
HATCH_MAP = {name: h for name, h in zip(METHODS, HATCHES)}

# ================= 2. 纵轴刻度配置 (自定义均匀刻度) =================

# 左轴 (JCT, s)
CUSTOM_Y_TICKS_SEC = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# 右轴 (Speedup)
# 自定义刻度，例如 0 到 4 倍
CUSTOM_Y2_TICKS_SPEEDUP = [0, 1.0, 1.8]

# 右轴起始位置 (0.0 ~ 1.0)
# 例如 0.75 表示右轴的第一个刻度从图表高度的 75% 处开始显示
RIGHT_AXIS_START_POS = 0.3

# ================= 3. 辅助函数 (坐标轴变换逻辑) =================

def map_values_to_linear_ticks(values, ticks):
    """将真实数值映射到线性索引空间 (0, 1, 2...)"""
    ticks = np.array(ticks)
    # 使用插值将数值映射到 tick 的 index 上
    return np.interp(values, ticks, np.arange(len(ticks)))

def transform_y(y_values, ticks):
    return map_values_to_linear_ticks(y_values, ticks)

# ================= 核心修改函数：构建自定义图例 =================
def create_combined_legend_box(fig, handles_row1, labels_row1, handles_row2, labels_row2):
    """
    使用 offsetbox 构建两行居中对齐、每行元素数量不同的统一图例
    """
    
    def create_legend_item(handle, label):
        # 1. 创建图例图标区域 (DrawingArea)
        # width=22, height=12 大致对应标准图例图标的大小
        da = DrawingArea(width=22, height=10, xdescent=0, ydescent=0)
        
        if isinstance(handle, Patch):
            # 提取 handle 的样式 (颜色、边框、纹理) 并画一个矩形
            rect = Rectangle((0, 0), width=22, height=10,
                             facecolor=handle.get_facecolor(),
                             edgecolor=handle.get_edgecolor(),
                             hatch=handle.get_hatch(),
                             linewidth=handle.get_linewidth())
            da.add_artist(rect)
        elif isinstance(handle, Line2D):
             # 提取 Line2D 的样式
            # 在 DrawingArea 中画线和点
            # 线: (0, 5) -> (11, 5) -> (22, 5) (垂直居中)
            # markevery=[1] 确保只在中间点绘制 marker
            line = Line2D([0, 11, 22], [5, 5, 5],
                          color=handle.get_color(),
                          linewidth=handle.get_linewidth(),
                          linestyle=handle.get_linestyle(),
                          marker=handle.get_marker(),
                          markersize=handle.get_markersize(),
                          markeredgecolor=handle.get_markeredgecolor(),
                          markeredgewidth=handle.get_markeredgewidth(),
                          markevery=[1])
            da.add_artist(line)
        
        # 2. 创建文字区域 (TextArea)
        ta = TextArea(label, textprops=dict(color="black", size=13, family="DejaVu Sans", fontweight='bold'))
        
        # 3. 将图标和文字水平打包 (HPacker), 类似 "icon  label"
        return HPacker(children=[da, ta], align="center", pad=0, sep=5)

    # --- 构建第一行 (Methods) ---
    row1_items = [create_legend_item(h, l) for h, l in zip(handles_row1, labels_row1)]
    # 使用 HPacker 将它们水平排列，sep 是列间距
    row1_packer = HPacker(children=row1_items, align="center", pad=0, sep=20)
    
    # --- 构建第二行 (Types) ---
    row2_items = [create_legend_item(h, l) for h, l in zip(handles_row2, labels_row2)]
    row2_packer = HPacker(children=row2_items, align="center", pad=0, sep=20)
    
    # --- 将两行垂直打包 (VPacker) ---
    # align="center" 保证两行相对于彼此居中
    vbox = VPacker(children=[row1_packer, row2_packer], align="center", pad=1, sep=7)
    
    # --- 放入带边框的容器 (AnchoredOffsetbox) ---
    anchored_box = AnchoredOffsetbox(
        loc='lower center',
        child=vbox,
        pad=0,          # 内部边距
        frameon=True,     # 开启边框
        bbox_to_anchor=(0.5, 0.83), # 整体位置 (相对于 fig)
        bbox_transform=fig.transFigure,
        borderpad=0.
    )
    
    # 设置边框样式
    anchored_box.patch.set_boxstyle("round,pad=0.4")
    anchored_box.patch.set_linewidth(1.2)
    anchored_box.patch.set_edgecolor('black')
    anchored_box.patch.set_facecolor('white')
    
    return anchored_box

# ================= 4. 绘图主逻辑 =================

def draw_chart():
    fig, ax = plt.subplots(figsize=(10, 5)) # 稍微加宽一点

    # 布局参数
    num_groups = len(X_LABELS)
    num_methods = len(METHODS)
    indices = np.arange(num_groups)
    
    # 柱状图宽度配置
    group_width = 0.8
    bar_width = group_width / num_methods

    # 准备基准数据 (Default)
    baseline_name = "Default(BF16)"
    if baseline_name in DATA_MS:
        baseline_vals = np.array(DATA_MS[baseline_name])
    else:
        # 如果找不到确切名字，尝试第一个或者报错
        print(f"Warning: Baseline '{baseline_name}' not found. Using first method as baseline.")
        baseline_name = METHODS[0]
        baseline_vals = np.array(DATA_MS[baseline_name])

    # 创建右轴
    ax2 = ax.twinx()

    # 收集图例 Handles
    legend_handles = []
    legend_labels = []

    # 绘制每个方法的柱子和折线
    for i, method in enumerate(METHODS):
        # --- 1. 柱状图部分 ---
        # 计算每个柱子的 X 坐标
        x_positions = indices - (group_width / 2) + (i * bar_width) + (bar_width / 2)
        
        # 获取数据 (ms) 并转换为 (s)
        values_ms = np.array(DATA_MS[method])
        values_sec = values_ms / 1000.0
        
        # 核心变换：计算柱子在自定义坐标系下的高度
        y_bottom_transformed = transform_y(0, CUSTOM_Y_TICKS_SEC)
        y_top_transformed = transform_y(values_sec, CUSTOM_Y_TICKS_SEC)
        bar_heights = y_top_transformed - y_bottom_transformed
        
        # 绘图 (柱子)
        bar = ax.bar(x_positions, bar_heights, 
               width=bar_width * 0.8, 
               bottom=y_bottom_transformed,
               color=COLOR_MAP[method], 
               edgecolor='black', 
               linewidth=1.5, 
               hatch=HATCH_MAP[method],
               zorder=3,) # 稍微透明一点，让折线更清楚？或者保持原样

        # 在柱子上方添加数值标签
        for x, y, val in zip(x_positions, y_top_transformed, values_sec):
            # 使用 ax2.text 并指定 transform=ax.transData，确保文字在右轴图层（上层），但位置跟随左轴数据
            ax2.text(x, y + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold', zorder=20, transform=ax.transData)

        # --- 2. 折线图部分 (Speedup) ---
        # 计算加速比: Baseline / Current
        # 避免除以0
        safe_values_ms = np.where(values_ms == 0, 1e-9, values_ms)
        speedup = baseline_vals / safe_values_ms
        
        # 变换到右轴坐标系
        speedup_transformed = transform_y(speedup, CUSTOM_Y2_TICKS_SPEEDUP)
        
        # 绘制折线点
        # X 坐标使用组的中心，以便于在同一垂直线上比较不同方法的加速比
        line_x = indices 

        # 绘制折线
        # 注意：这里的折线是连接各组中该方法的点
        # 颜色与柱子一致，使用特定 marker
        # Default 的层级 (zorder) 稍微高一点
        line_zorder = 12 if method == baseline_name else 10
        line, = ax2.plot(line_x, speedup_transformed, 
                 color=COLOR_MAP[method],
                 marker=MARKER_MAP[method],
                 markersize=10,
                 linewidth=LINE_WIDTH,
                 linestyle=LINES_MAP[method],
                 markeredgecolor='white',
                 markeredgewidth=1.0,
                 zorder=line_zorder) # 保证在柱子上面
        
        # 特殊处理：为 KVServe 添加 Speedup 数值标签
        if method == "KVServe":
             for j, (sx, sy, val) in enumerate(zip(line_x, speedup_transformed, speedup)):
                # 右上方显示，颜色与折线一致
                ax2.text(sx + 0.02, sy + 0.05, f"{val:.2f}x", 
                         color=COLOR_MAP[method], 
                         fontweight='bold', 
                         fontsize=12, 
                         ha='left', va='bottom',
                         zorder=30)

        # 添加到图例 (使用 Line2D 对象，它包含了 marker 和 color)
        legend_handles.append(line)
        legend_labels.append(method)

    # ================= 5. 坐标轴与样式设置 =================
    
    # 设置左 Y 轴 (JCT)
    ax.set_yticks(np.arange(len(CUSTOM_Y_TICKS_SEC)))
    ax.set_yticklabels([f"{y}" for y in CUSTOM_Y_TICKS_SEC])
    ax.set_ylim(0, len(CUSTOM_Y_TICKS_SEC) - 1)
    ax.set_ylabel("JCT (s)", fontweight='bold', fontsize=16)
    
    # 设置右 Y 轴 (Speedup)
    r_max_idx = len(CUSTOM_Y2_TICKS_SPEEDUP) - 1
    ax2.set_yticks(np.arange(len(CUSTOM_Y2_TICKS_SPEEDUP)))
    # 不要绘制0刻度
    # 找到第一个非0的位置（通常是1.0）
    y2_ticks_no_zero = [y for y in CUSTOM_Y2_TICKS_SPEEDUP if y != 0]
    ax2.set_yticks(np.arange(1, len(CUSTOM_Y2_TICKS_SPEEDUP)))  # 从1开始，跳过0刻度
    ax2.set_yticklabels([f"{y:.1f}" for y in y2_ticks_no_zero])
    # 调整右轴 ylim 以控制起始位置
    if 0 < RIGHT_AXIS_START_POS < 1:
        # 公式推导: 
        # 设可视范围为 [Y_min, R_max]
        # 刻度 0 (对应 CUSTOM_Y2_TICKS_SPEEDUP[0]) 位于可视范围的 RIGHT_AXIS_START_POS 处
        # 0 - Y_min = RIGHT_AXIS_START_POS * (R_max - Y_min)
        # 解得 Y_min = - (P * R_max) / (1 - P)
        y_min_right = - (RIGHT_AXIS_START_POS * r_max_idx) / (1 - RIGHT_AXIS_START_POS)
        ax2.set_ylim(y_min_right, r_max_idx)
    else:
        ax2.set_ylim(0, r_max_idx)

    ax2.set_ylabel("Speedup (x)", fontweight='bold', fontsize=16, rotation=270, labelpad=18)

    # 设置 X 轴
    ax.set_xticks(indices)
    ax.set_xticklabels(X_LABELS, fontsize=16, fontweight='bold')
    
    # 网格线 (仅左轴 Y)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    # 去除边框
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False) # 右轴需要显示 spine 吗？通常双轴图保留右 spine
    # ax2.spines['right'].set_visible(True)

    # 图例 (放置在上方，使用自定义带边框图例)
    
    # --- Row 1: Methods (4个) ---
    method_handles = []
    method_labels = []
    for m in METHODS:
        # 使用 Patch 展示颜色和纹理
        p = Patch(facecolor=COLOR_MAP[m], edgecolor='black', hatch=HATCH_MAP[m], linewidth=1.5)
        method_handles.append(p)
        method_labels.append(m)

    # --- Row 2: Types (2个) ---
    # 1. Latency (柱状图代表) - 用白色带边框的方块表示
    h_latency = Patch(facecolor='white', edgecolor='black', linewidth=1.5)
    l_latency = 'JCT (Left)'
    
    # 2. Speedup (折线图代表) - 用黑色点线表示
    h_speedup = Line2D([0], [0], color='black', marker='o', markersize=8, 
                       linestyle='-', markeredgecolor='white', markeredgewidth=1.5)
    l_speedup = 'Speedup (Right)'
    
    type_handles = [h_latency, h_speedup]
    type_labels = [l_latency, l_speedup]

    # 创建并添加自定义图例
    custom_legend = create_combined_legend_box(fig, method_handles, method_labels, type_handles, type_labels)
    fig.add_artist(custom_legend)

    # 保存
    output_filename = "jct_latency_bar_chart.pdf"
    plt.tight_layout()
    # 压缩下方图的位置，给双排图例留出空间 (top=0.80)
    plt.subplots_adjust(top=0.8, bottom=0.18)
    plt.savefig(output_filename)
    print(f"Plot generated: {output_filename}")

if __name__ == "__main__":
    draw_chart()
