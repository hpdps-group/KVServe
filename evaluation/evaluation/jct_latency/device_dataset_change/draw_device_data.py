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

# 标题配置 (大写字母自定义变量)
PLOT_TITLES = [
    "Llama-8B (2WikiMQA)",
    "Llama-8B (HotpotQA)",
    "Qwen-32B (2WikiMQA)",
    "Qwen-32B (HotpotQA)"
]

# CSV 文件列表 (假设在当前目录下)
CSV_FILES = [
    "device_llama8_2wikimqa.csv",
    "device_llama8_hotpotqa.csv",
    "device_qwen32_2wikimqa.csv",
    "device_qwen32_hotpotqa.csv"
]

# ================= 2. 纵轴刻度配置 (自定义均匀刻度 - 8个list) =================

# 左轴 (JCT, s) - 4组配置
LEFT_TICKS_LIST = [
    [1, 2, 4, 6], # Plot 1
    [1, 2, 4, 6], # Plot 2
    [1, 2, 4, 6], # Plot 3
    [1, 2, 4, 6], # Plot 4
]

# 右轴 (Speedup) - 4组配置
RIGHT_TICKS_LIST = [
    [0, 1, 4], # Plot 1
    [0, 1, 4], # Plot 2
    [0, 1, 4], # Plot 3
    [0, 1, 4], # Plot 4
]

# 右轴起始位置 (0.0 ~ 1.0)
RIGHT_AXIS_START_POS = 0.25

# ================= 1. 数据准备区域 (从CSV读取) =================

def load_data_from_csv(file_path):
    data_list = []
    # Adjust path if relative
    if not os.path.isabs(file_path):
         file_path = os.path.join(os.path.dirname(__file__), file_path)
         
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return [], [], {}
    
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
                print(f"Warning: Missing data for {x_label} - {method} in {os.path.basename(file_path)}")
            data_ms[method].append(val)
            
    return x_labels, methods, data_ms

# 颜色池 (按顺序分配)
METHOD_COLORS = ['#868686', '#FFBB78', '#C5B0D5', '#e41a1c', '#9575cd', '#4dd0e1'] 
MARKERS = ['o', 's', 'p', 'h', 'v', '<', '>']
LINES = ['--', '--', '--', '-']
HATCHES = ['..', '//', '\\\\', 'xx', '..', '++']

# ================= 3. 辅助函数 (坐标轴变换逻辑) =================

def map_values_to_linear_ticks(values, ticks):
    ticks = np.array(ticks)
    # 使用插值将数值映射到 tick 的 index 上
    return np.interp(values, ticks, np.arange(len(ticks)))

def transform_y(y_values, ticks):
    return map_values_to_linear_ticks(y_values, ticks)

# ================= 核心修改函数：构建自定义图例 =================
def create_combined_legend_box(fig, handles_row1, labels_row1, handles_row2, labels_row2):
    """
    使用 offsetbox 构建两行居中对齐、每行元素数量不同的统一图例
    修改为：单行显示，row2 接在 row1 后面
    """
    
    def create_legend_item(handle, label):
        # 1. 创建图例图标区域 (DrawingArea)
        da = DrawingArea(width=22, height=10, xdescent=0, ydescent=0)
        
        if isinstance(handle, Patch):
            rect = Rectangle((0, 0), width=22, height=10,
                             facecolor=handle.get_facecolor(),
                             edgecolor=handle.get_edgecolor(),
                             hatch=handle.get_hatch(),
                             linewidth=handle.get_linewidth())
            da.add_artist(rect)
        elif isinstance(handle, Line2D):
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
        
        # 3. 将图标和文字水平打包 (HPacker)
        return HPacker(children=[da, ta], align="center", pad=0, sep=5)

    # --- 合并两行内容到一行 ---
    all_handles = handles_row1 + handles_row2
    all_labels = labels_row1 + labels_row2

    row_items = [create_legend_item(h, l) for h, l in zip(all_handles, all_labels)]
    packer = HPacker(children=row_items, align="center", pad=0, sep=20)
    
    # --- 放入带边框的容器 (AnchoredOffsetbox) ---
    anchored_box = AnchoredOffsetbox(
        loc='lower center',
        child=packer,
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
    # 画布大小长度变成两倍 (10->20)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # 用于图例的 handles (只从第一个图获取即可)
    legend_method_handles = []
    legend_method_labels = []
    
    # 遍历4个子图
    for i, ax in enumerate(axes):
        # 1. 获取对应配置
        csv_file = CSV_FILES[i]
        title = PLOT_TITLES[i]
        left_ticks = LEFT_TICKS_LIST[i]
        right_ticks = RIGHT_TICKS_LIST[i]
        
        print(f"Processing Plot {i+1}: {csv_file}")
        x_labels, methods, data_ms = load_data_from_csv(csv_file)
        
        # 颜色映射
        current_colors = METHOD_COLORS[:len(methods)]
        color_map = {name: col for name, col in zip(methods, current_colors)}
        marker_map = {name: m for name, m in zip(methods, MARKERS)}
        lines_map = {name: l for name, l in zip(methods, LINES)}
        hatch_map = {name: h for name, h in zip(methods, HATCHES)}
        
        # 如果是第一个图，收集 Method 图例信息
        if i == 0:
             for m in methods:
                p = Patch(facecolor=color_map[m], edgecolor='black', hatch=hatch_map[m], linewidth=1.5)
                legend_method_handles.append(p)
                legend_method_labels.append(m)

        # 布局参数
        num_groups = len(x_labels)
        num_methods = len(methods)
        indices = np.arange(num_groups)
        group_width = 0.9
        bar_width = group_width / num_methods
        
        # 准备基准数据 (Default)
        baseline_name = "Default(BF16)"
        if baseline_name not in data_ms:
            baseline_name = methods[0]
            # print(f"Warning: Baseline '{baseline_name}' not found in {csv_file}. Using {baseline_name}.")
        baseline_vals = np.array(data_ms[baseline_name])
        
        # 创建右轴
        ax2 = ax.twinx()
        
        # 绘图循环
        for j, method in enumerate(methods):
            # --- 1. 柱状图部分 ---
            x_positions = indices - (group_width / 2) + (j * bar_width) + (bar_width / 2)
            values_ms = np.array(data_ms[method])
            values_sec = values_ms / 1000.0
            
            y_bottom_transformed = transform_y(0, left_ticks)
            y_top_transformed = transform_y(values_sec, left_ticks)
            bar_heights = y_top_transformed - y_bottom_transformed
            
            ax.bar(x_positions, bar_heights, 
                   width=bar_width * 0.8, 
                   bottom=y_bottom_transformed,
                   color=color_map[method], 
                   edgecolor='black', 
                   linewidth=1.5, 
                   hatch=hatch_map[method],
                   zorder=3)
            
            # 数值标签 (柱子上) - 已移除
            # for x, y, val in zip(x_positions, y_top_transformed, values_sec):
            #      ax2.text(x, y + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=20, transform=ax.transData)

            # --- 2. 折线图部分 (Speedup) ---
            safe_values_ms = np.where(values_ms == 0, 1e-9, values_ms)
            speedup = baseline_vals / safe_values_ms
            speedup_transformed = transform_y(speedup, right_ticks)
            
            line_zorder = 12 if method == baseline_name else 10
            line, = ax2.plot(indices, speedup_transformed, 
                     color=color_map[method],
                     marker=marker_map[method],
                     markersize=10,
                     linewidth=LINE_WIDTH,
                     linestyle=lines_map[method],
                     markeredgecolor='white',
                     markeredgewidth=1.0,
                     zorder=line_zorder)
            
            # 特殊处理：KVServe Speedup 数值标签
            if method == "KVServe":
                 for sx, sy, val in zip(indices, speedup_transformed, speedup):
                    ax2.text(sx + 0.02, sy + 0.05, f"{val:.2f}x", 
                             color=color_map[method], 
                             fontweight='bold', 
                             fontsize=10, 
                             ha='left', va='bottom',
                             zorder=30)

        # --- 坐标轴设置 ---
        # 标题
        ax.set_title(title, fontweight='bold', fontsize=14, pad=10, x=0.65, y=0.9)

        # 左轴 (JCT)
        ax.set_yticks(np.arange(len(left_ticks)))
        ax.set_yticklabels([f"{y}" for y in left_ticks])
        ax.set_ylim(0, len(left_ticks) - 1)
        
        if i == 0:
            ax.set_ylabel("JCT (s)", fontweight='bold', fontsize=16)
        else:
            ax.set_ylabel("")
            
        # 右轴 (Speedup)
        r_max_idx = len(right_ticks) - 1
        # 隐藏索引为0的刻度 (通常是0值)
        ax2.set_yticks(np.arange(1, len(right_ticks)))
        ax2.set_yticklabels([f"{y}" for y in right_ticks[1:]])
        
        if 0 < RIGHT_AXIS_START_POS < 1:
            y_min_right = - (RIGHT_AXIS_START_POS * r_max_idx) / (1 - RIGHT_AXIS_START_POS)
            ax2.set_ylim(y_min_right, r_max_idx)
        else:
            ax2.set_ylim(0, r_max_idx)

        if i == 3: # 最后一个图显示右轴标签
            ax2.set_ylabel("Speedup (x)", fontweight='bold', fontsize=16, rotation=270, labelpad=18)
        else:
            ax2.set_ylabel("")

        # X轴
        ax.set_xticks(indices)
        ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        
        # 边框
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
    # --- 图例 (共用) ---
    h_latency = Patch(facecolor='white', edgecolor='black', linewidth=1.5)
    l_latency = 'JCT (Left)'
    h_speedup = Line2D([0], [0], color='black', marker='o', markersize=8, 
                       linestyle='-', markeredgecolor='white', markeredgewidth=1.5)
    l_speedup = 'Speedup (Right)'
    
    type_handles = [h_latency, h_speedup]
    type_labels = [l_latency, l_speedup]
    
    custom_legend = create_combined_legend_box(fig, legend_method_handles, legend_method_labels, type_handles, type_labels)
    fig.add_artist(custom_legend)
    
    # 保存
    output_filename = "jct_latency_bar_chart_combined.pdf"
    plt.tight_layout()
    # 调整布局以适应顶部图例
    plt.subplots_adjust(top=0.78, bottom=0.15, wspace=0.2)
    plt.savefig(output_filename)
    print(f"Plot generated: {output_filename}")

if __name__ == "__main__":
    draw_chart()
