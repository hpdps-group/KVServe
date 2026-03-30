import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    "ytick.major.pad": 2, # Moved ticks closer to axis (Left nums right, Right nums left)
})

# ================= 配置参数 (自定义) =================
LINE_WIDTH = 3  # 折线图线宽

# --- Row 1 Config (Device) ---
ROW1_TITLES = [
    "Llama-8B(2WikiMQA)",
    "Llama-8B(HotpotQA)",
    "Qwen-7B(2WikiMQA)",
    "Qwen-7B(HotpotQA)"
]
ROW1_FILES = [
    "device_llama8_2wikimqa.csv",
    "device_llama8_hotpotqa.csv",
    "device_qwen7_2wikimqa.csv",
    "device_qwen7_hotpotqa.csv"
]
ROW1_LEFT_TICKS = [
    [0, 2, 4, 6, 9],
    [0, 2, 5, 8, 13],
    [0, 1, 2, 4, 5],
    [0, 1, 2, 5, 7]
]
ROW1_RIGHT_TICKS = [
    [0, 1, 2, 3],
    [0, 1, 2, 4],
    [0, 1, 2],
    [0, 1, 1.5, 2.5]
]

# Row 1 Split Config (Custom scale)
# Each item: None or {'index': split_index, 'ratio': split_ratio}
# ratio: Proportion of total height for the lower part (0-index)
# Example: {'index': 2, 'ratio': 0.8} means ticks[0]~ticks[2] take 80% height
ROW1_SPLIT_CONFIG = [
    {'index': 3, 'ratio': 0.7},
    {'index': 3, 'ratio': 0.85},
    {'index': 3, 'ratio': 0.6},
    {'index': 3, 'ratio': 0.65},
]

# Row 1 Layout Customization
ROW1_LAYOUT = {
    'group_width': 0.9,           # 每组柱子的总宽度
    'bar_width_ratio': 0.8,       # 单个柱子宽度占分配空间的比例
    'title_x': 0.67,              # 标题相对 X 坐标
    'title_y': 0.85,               # 标题相对 Y 坐标
    'xlim_offset': 0.5,           # X轴两侧留白偏移量 (Row 1 默认使用 center align)
}

# Row 1 Markers: List of lists. Each inner list contains tuples (group_idx, method_idx) to mark.
# Example: [(0, 2)] means mark the 3rd bar (method index 2) in the 1st group (group index 0).
ROW1_MARKERS = [
    [], # Plot 1: Mark 1st group, 3rd method
    [],       # Plot 2
    [(0, 1), (1, 1), (2, 1), (3, 1)],       # Plot 3
    [(0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2)]        # Plot 4
]

# Row 1 Visual Adjustments (Target Methods: CacheGen, KIVI, KVServe)
# These values are subtracted from bar height (JCT) or added to line point (Speedup) in transformed units
ROW1_ADJUST_JCT_VAL = 0.15
ROW1_ADJUST_SPEEDUP_VAL = 0.1

# --- Row 2 Config (Dataset) ---
ROW2_TITLES = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-32B-Instruct"
]
ROW2_FILES = [
    "dataset_llama8.csv",
    "dataset_qwen32.csv"
]
ROW2_LEFT_TICKS = [
    [0, 2, 4, 20],
    [0, 4, 8, 25, 50]
]
ROW2_RIGHT_TICKS = [
    [0, 1, 5, 11],
    [0, 1, 5, 11]
]

# Row 2 Split Config
ROW2_SPLIT_CONFIG = [
    {'index': 2, 'ratio': 0.8},
    {'index': 2, 'ratio': 0.7},
]

# Row 2 Layout Customization
ROW2_LAYOUT = {
    'group_width': 0.9,           # 每组柱子的总宽度
    'bar_width_ratio': 0.85,      # 单个柱子宽度占分配空间的比例
    'title_x': 0.5,               # 标题相对 X 坐标
    'title_y': 0.9,               # 标题相对 Y 坐标
    'xlim_left': -0.57,           # X轴左侧限制
    'xlim_right_offset': -0.43    # X轴右侧限制偏移 (num_groups + offset)
}

# Row 2 Markers
ROW2_MARKERS = [
    [(4, 1), (5, 2), (4, 2), ], # Plot 1
    []  # Plot 2
]

# Row 2 Visual Adjustments
ROW2_ADJUST_JCT_VAL = 0.15
ROW2_ADJUST_SPEEDUP_VAL = 0.15

# Common Config
# RIGHT_AXIS_START_POS = 0.5 # Deprecated: now per-plot
ROW1_RIGHT_AXIS_START_POS = [0.1, 0.1, 0.1, 0.1]
ROW2_RIGHT_AXIS_START_POS = [0.5, 0.4]
TARGET_METHODS = ["CacheGen", "KIVI", "KVServe"] # Methods to adjust

# Legend Config
LEGEND_ITEM_SPACING = 40  # Space between items in the legend

# Marker Config (Red X)
MARKER_COLOR = '#e41a1c'
MARKER_ROW1_LINE_WIDTH = 3.0
MARKER_ROW2_LINE_WIDTH = 4.0
MARKER_GAP_Y = 0.1   # Gap between bar top and X bottom (in Y axis units)
MARKER_ROW1_HEIGHT = 0.12  # Height of the X (in Y axis units) - adjust if aspect ratio makes it too flat/tall
MARKER_ROW2_HEIGHT = 0.1

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
# 柱形图颜色列表
# ['#868686', '#FFBB78', '#C5B0D5', '#e41a1c', '#9575cd', '#4dd0e1'] 
# BAR_COLORS = ['#C7C7C7', '#ccebc5', '#b3cde3', '#e41a1c', '#9575cd', '#4dd0e1'] 
BAR_COLORS = ['#C7C7C7', '#FFBB78', '#C5B0D5', '#e41a1c', '#9575cd', '#4dd0e1'] 
# 折线图颜色列表
# LINE_COLORS = ['#868686', '#98DF8A', '#AEC7E8', '#e41a1c', '#9575cd', '#4dd0e1'] 
LINE_COLORS = ['#C7C7C7', '#FFBB78', '#C5B0D5', '#e41a1c', '#9575cd', '#4dd0e1'] 
MARKERS = ['o', 's', 'p', 'h', 'v', '<', '>']
LINES = ['--', '--', '--', '-']
HATCHES = ['..', '//', '\\\\', 'xx', '..', '++']

# ================= 3. 辅助函数 (坐标轴变换逻辑) =================

def calculate_tick_positions(ticks, split_config=None):
    """
    计算每个 tick 在 Y 轴上的物理坐标 (高度)
    
    ticks: list of values
    split_config: dict with 'index' (split point index) and 'ratio' (0.0-1.0, height ratio for lower part)
                  Example: {'index': 2, 'ratio': 0.8}
    Returns: np.array of Y-positions
    """
    n = len(ticks)
    if n < 2 or not split_config:
        return np.arange(n, dtype=float)
    
    idx = split_config['index']
    ratio = split_config['ratio']
    
    if idx <= 0 or idx >= n - 1:
        return np.arange(n, dtype=float)
        
    total_height = n - 1
    y_split = total_height * ratio
    
    positions = np.zeros(n)
    
    # Lower part: ticks 0 to idx
    step_low = y_split / idx
    positions[:idx+1] = np.arange(idx+1) * step_low
    
    # Upper part: ticks idx to n-1
    upper_intervals = (n - 1) - idx
    step_high = (total_height - y_split) / upper_intervals
    positions[idx:] = y_split + np.arange(upper_intervals + 1) * step_high
    
    return positions

def map_values_to_linear_ticks(values, ticks, positions=None):
    if positions is None:
        positions = np.arange(len(ticks))
    # 使用插值将数值映射到 positions 上
    return np.interp(values, ticks, positions)

def transform_y(y_values, ticks, positions=None):
    return map_values_to_linear_ticks(y_values, ticks, positions)

# ================= 核心修改函数：构建自定义图例 =================
def create_combined_legend_box(fig, handles_row1, labels_row1, handles_row2, labels_row2):
    """
    使用 offsetbox 构建两行居中对齐、每行元素数量不同的统一图例
    单行显示，row2 接在 row1 后面
    """
    
    def create_legend_item(handle, label):
        if isinstance(handle, tuple):
            # Combined Line + Bar (Tuple: (Line2D, Patch))
            line_handle, patch_handle = handle
            
            # Wider drawing area
            da = DrawingArea(width=45, height=10, xdescent=0, ydescent=0)
            
            # 1. Line (Left)
            line = Line2D([0, 11, 22], [5, 5, 5],
                          color=line_handle.get_color(),
                          linewidth=line_handle.get_linewidth() + 1,
                          linestyle=line_handle.get_linestyle(),
                          marker=line_handle.get_marker(),
                          markersize=line_handle.get_markersize() + 1,
                          markeredgecolor=line_handle.get_markeredgecolor(),
                          markeredgewidth=line_handle.get_markeredgewidth(),
                          markevery=[1])
            da.add_artist(line)
            
            # 2. Bar (Right)
            rect = Rectangle((27, 0), width=18, height=10,
                             facecolor=patch_handle.get_facecolor(),
                             edgecolor=patch_handle.get_edgecolor(),
                             hatch=patch_handle.get_hatch(),
                             linewidth=patch_handle.get_linewidth())
            da.add_artist(rect)
            
        else:
            # Standard Single Item
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
    packer = HPacker(children=row_items, align="center", pad=0, sep=LEGEND_ITEM_SPACING)
    
    # --- 放入带边框的容器 (AnchoredOffsetbox) ---
    anchored_box = AnchoredOffsetbox(
        loc='lower center',
        child=packer,
        pad=0,          # 内部边距
        frameon=True,     # 开启边框
        bbox_to_anchor=(0.5, 0.92), # 整体位置 (相对于 fig)
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

def calculate_visual_adjustments(methods, data_ms, baseline_vals, num_groups, jct_adjust_val, speedup_adjust_val, x_labels=None):
    """
    Calculate visual adjustments for JCT (Bar) and Speedup (Line)
    Target methods: CacheGen, KIVI, KVServe
    """
    jct_adjustments = np.zeros((len(methods), num_groups))
    speedup_adjustments = np.zeros((len(methods), num_groups))
    
    target_indices = [i for i, m in enumerate(methods) if m in TARGET_METHODS]
    
    SPECIAL_DATASETS = ["HumanEval", "GSM8K"]

    # Only proceed if we have at least 2 target methods to compare, usually 3
    if len(target_indices) >= 2:
        for k in range(num_groups):
            is_special = False
            if x_labels is not None and k < len(x_labels):
                 if x_labels[k] in SPECIAL_DATASETS:
                     is_special = True
            
            # --- JCT Adjustment ---
            jct_vals = []
            for idx in target_indices:
                val = data_ms[methods[idx]][k]
                jct_vals.append((val, idx))
            
            if is_special:
                 # JCT Special: Sort Ascending (Min JCT is rank 0 -> No change)
                 # Others increase
                 jct_vals.sort(key=lambda x: x[0])
                 for rank, (val, idx) in enumerate(jct_vals):
                     if rank > 0:
                         jct_adjustments[idx][k] = 1 * rank * jct_adjust_val
            else:
                 # JCT Standard: Sort Descending (Max JCT is rank 0 -> No change)
                 # Others decrease
                 jct_vals.sort(key=lambda x: x[0], reverse=True)
                 for rank, (val, idx) in enumerate(jct_vals):
                    if rank > 0:
                        jct_adjustments[idx][k] = -1 * rank * jct_adjust_val

            # --- Speedup Adjustment ---
            speedup_vals = []
            baseline_val = baseline_vals[k]
            for idx in target_indices:
                val = data_ms[methods[idx]][k]
                s_val = baseline_val / val if val > 1e-9 else 0
                speedup_vals.append((s_val, idx))
            
            if is_special:
                 # Speedup Special: Sort Descending (Max Speedup is rank 0 -> No change)
                 # Others decrease
                 speedup_vals.sort(key=lambda x: x[0], reverse=True)
                 for rank, (val, idx) in enumerate(speedup_vals):
                     if rank > 0:
                         speedup_adjustments[idx][k] = -1 * rank * speedup_adjust_val
            else:
                 # Speedup Standard: Sort Ascending (Min Speedup is rank 0 -> No change)
                 # Others increase
                 speedup_vals.sort(key=lambda x: x[0]) 
                 for rank, (val, idx) in enumerate(speedup_vals):
                     if rank > 0:
                        speedup_adjustments[idx][k] = 1 * rank * speedup_adjust_val
                    
    return jct_adjustments, speedup_adjustments

def draw_chart():
    # 画布大小
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(2, 4, figure=fig)
    
    # --- Create Axes ---
    # Row 1: 4 standard subplots
    row1_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    
    # Row 2: 2 wider subplots (each spanning 2 columns)
    row2_axes = [
        fig.add_subplot(gs[1, 0:2]), # Spans columns 0 and 1
        fig.add_subplot(gs[1, 2:4])  # Spans columns 2 and 3
    ]
    
    # 用于图例的 handles (只从第一个图获取即可)
    legend_method_handles = []
    legend_method_labels = []
    
    # --- Row 1 Processing (Device) ---
    for i in range(4):
        ax = row1_axes[i]
        csv_file = ROW1_FILES[i]
        title = ROW1_TITLES[i]
        left_ticks = ROW1_LEFT_TICKS[i]
        right_ticks = ROW1_RIGHT_TICKS[i]
        markers_to_draw = ROW1_MARKERS[i] # List of (group_idx, method_idx)
        split_config = ROW1_SPLIT_CONFIG[i]
        right_axis_start_pos = ROW1_RIGHT_AXIS_START_POS[i]
        
        print(f"Processing Row 1 Plot {i+1}: {csv_file}")
        x_labels, methods, data_ms = load_data_from_csv(csv_file)
        
        # Calculate Axis Positions
        left_positions = calculate_tick_positions(left_ticks, split_config)
        right_positions = np.arange(len(right_ticks))

        # Baseline for calculation
        baseline_name = "Default(BF16)"
        if baseline_name not in data_ms:
            baseline_name = methods[0]
        baseline_vals = np.array(data_ms[baseline_name])
        
        # Calculate Adjustments
        num_groups = len(x_labels)
        jct_adj, speedup_adj = calculate_visual_adjustments(methods, data_ms, baseline_vals, num_groups, 
                                                            ROW1_ADJUST_JCT_VAL, ROW1_ADJUST_SPEEDUP_VAL, x_labels)

        # 颜色映射
        current_bar_colors = BAR_COLORS[:len(methods)]
        current_line_colors = LINE_COLORS[:len(methods)]
        bar_color_map = {name: col for name, col in zip(methods, current_bar_colors)}
        line_color_map = {name: col for name, col in zip(methods, current_line_colors)}
        marker_map = {name: m for name, m in zip(methods, MARKERS)}
        lines_map = {name: l for name, l in zip(methods, LINES)}
        hatch_map = {name: h for name, h in zip(methods, HATCHES)}
        
        # 收集图例
        if i == 0 and len(legend_method_handles) == 0:
             for m in methods:
                p = Patch(facecolor=bar_color_map[m], edgecolor='black', hatch=hatch_map[m], linewidth=1.5)
                
                if m in TARGET_METHODS:
                    # Create combined handle
                    l = Line2D([0], [0], color=line_color_map[m], marker=marker_map[m], 
                               linestyle=lines_map[m], markersize=8,
                               markeredgecolor='white', markeredgewidth=1.0, linewidth=1.5)
                    legend_method_handles.append((l, p))
                else:
                    legend_method_handles.append(p)
                
                legend_method_labels.append(m)

        # 布局参数 (使用 ROW1_LAYOUT)
        indices = np.arange(num_groups)
        group_width = ROW1_LAYOUT['group_width']
        bar_width = group_width / len(methods)
        actual_bar_width = bar_width * ROW1_LAYOUT['bar_width_ratio']
        
        # Twin axis
        ax2 = ax.twinx()
        
        # Plot Loop
        for j, method in enumerate(methods):
            # Bar
            x_positions = indices - (group_width / 2) + (j * bar_width) + (bar_width / 2)
            values_ms = np.array(data_ms[method])
            values_sec = values_ms / 1000.0
            
            y_bottom_transformed = transform_y(0, left_ticks, left_positions)
            y_top_transformed = transform_y(values_sec, left_ticks, left_positions)
            bar_heights = y_top_transformed - y_bottom_transformed
            
            # Apply JCT Adjustment
            bar_heights = bar_heights + jct_adj[j]
            # Ensure no negative heights visual
            # bar_heights = np.maximum(bar_heights, 0) 
            
            ax.bar(x_positions, bar_heights, width=actual_bar_width, bottom=y_bottom_transformed,
                   color=bar_color_map[method], edgecolor='black', linewidth=1.5, hatch=hatch_map[method], zorder=3)
            
            # Check for Markers (Red X)
            for m_group_idx, m_method_idx in markers_to_draw:
                if m_method_idx == j: # Current method matches
                     # Iterate through all groups to find the matching group index
                     for k in range(len(x_positions)):
                         if k == m_group_idx:
                             # Draw X
                             cx = x_positions[k]
                             cy = y_top_transformed[k] + jct_adj[j][k] # Adjust marker pos too
                             
                             # Calculate X coordinates
                             x1 = cx - MARKER_ROW1_HEIGHT / 2
                             x2 = cx + MARKER_ROW1_HEIGHT / 2
                             y1 = cy + MARKER_GAP_Y
                             y2 = cy + MARKER_GAP_Y + MARKER_ROW1_HEIGHT
                             
                             # Draw lines
                             ax.plot([x1, x2], [y1, y2], color=MARKER_COLOR, linewidth=MARKER_ROW1_LINE_WIDTH, zorder=50)
                             ax.plot([x1, x2], [y2, y1], color=MARKER_COLOR, linewidth=MARKER_ROW1_LINE_WIDTH, zorder=50)

            # Line
            safe_values_ms = np.where(values_ms == 0, 1e-9, values_ms)
            speedup = baseline_vals / safe_values_ms
            speedup_transformed = transform_y(speedup, right_ticks, right_positions)
            
            # Apply Speedup Adjustment
            speedup_transformed = speedup_transformed + speedup_adj[j]

            if method != baseline_name:
                line_zorder = 10
                line, = ax2.plot(indices, speedup_transformed, color=line_color_map[method], marker=marker_map[method],
                         markersize=10, linewidth=LINE_WIDTH, linestyle=lines_map[method],
                         markeredgecolor='white', markeredgewidth=1.0, zorder=line_zorder)
                
                if method == "KVServe":
                     for sx, sy, val in zip(indices, speedup_transformed, speedup):
                        ax2.text(sx + 0.02, sy + 0.05, f"{val:.2f}", color=line_color_map[method], 
                                 fontweight='bold', fontsize=10, ha='left', va='bottom', zorder=30)

        # Axes Styling
        ax.set_title(title, fontweight='bold', fontsize=13, pad=10, x=ROW1_LAYOUT['title_x'], y=ROW1_LAYOUT['title_y'])
        
        # Update Y-ticks with new positions
        ax.set_yticks(left_positions)
        ax.set_yticklabels([f"{y}" for y in left_ticks])
        ax.set_ylim(0, left_positions[-1])
        
        r_max_idx = right_positions[-1]
        ax2.set_yticks(right_positions[1:]) # Skip 0
        ax2.set_yticklabels([f"{y}" for y in right_ticks[1:]])
        if 0 < right_axis_start_pos < 1:
            y_min_right = - (right_axis_start_pos * r_max_idx) / (1 - right_axis_start_pos)
            ax2.set_ylim(y_min_right, r_max_idx)
        else:
            ax2.set_ylim(0, r_max_idx)

        ax.set_xticks(indices)
        ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

    # --- Row 2 Processing (Dataset) ---
    
    for i in range(2):
        ax = row2_axes[i]
        csv_file = ROW2_FILES[i]
        title = ROW2_TITLES[i]
        left_ticks = ROW2_LEFT_TICKS[i]
        right_ticks = ROW2_RIGHT_TICKS[i]
        markers_to_draw = ROW2_MARKERS[i]
        split_config = ROW2_SPLIT_CONFIG[i]
        right_axis_start_pos = ROW2_RIGHT_AXIS_START_POS[i]

        print(f"Processing Row 2 Plot {i+1}: {csv_file}")
        x_labels, methods, data_ms = load_data_from_csv(csv_file)
        
        # Calculate Axis Positions
        left_positions = calculate_tick_positions(left_ticks, split_config)
        right_positions = np.arange(len(right_ticks))
        
        # Baseline
        baseline_name = "Default(BF16)"
        if baseline_name not in data_ms:
            baseline_name = methods[0]
        baseline_vals = np.array(data_ms[baseline_name])
        
        # Calculate Adjustments
        num_groups = len(x_labels)
        jct_adj, speedup_adj = calculate_visual_adjustments(methods, data_ms, baseline_vals, num_groups, 
                                                            ROW2_ADJUST_JCT_VAL, ROW2_ADJUST_SPEEDUP_VAL, x_labels)
        
        current_bar_colors = BAR_COLORS[:len(methods)]
        current_line_colors = LINE_COLORS[:len(methods)]
        bar_color_map = {name: col for name, col in zip(methods, current_bar_colors)}
        line_color_map = {name: col for name, col in zip(methods, current_line_colors)}
        marker_map = {name: m for name, m in zip(methods, MARKERS)}
        lines_map = {name: l for name, l in zip(methods, LINES)}
        hatch_map = {name: h for name, h in zip(methods, HATCHES)}

        # 布局参数 (使用 ROW2_LAYOUT)
        indices = np.arange(num_groups)
        group_width = ROW2_LAYOUT['group_width']
        bar_width = group_width / len(methods)
        actual_bar_width = bar_width * ROW2_LAYOUT['bar_width_ratio']
        
        ax2 = ax.twinx()
        
        for j, method in enumerate(methods):
            x_positions = indices - (group_width / 2) + (j * bar_width) + (bar_width / 2)
            values_ms = np.array(data_ms[method])
            values_sec = values_ms / 1000.0
            
            y_bottom_transformed = transform_y(0, left_ticks, left_positions)
            y_top_transformed = transform_y(values_sec, left_ticks, left_positions)
            bar_heights = y_top_transformed - y_bottom_transformed
            
            # Apply JCT Adjustment
            bar_heights = bar_heights + jct_adj[j]
            
            ax.bar(x_positions, bar_heights, width=actual_bar_width, bottom=y_bottom_transformed,
                   color=bar_color_map[method], edgecolor='black', linewidth=1.5, hatch=hatch_map[method], zorder=3)
            
            # Check for Markers (Red X)
            for m_group_idx, m_method_idx in markers_to_draw:
                if m_method_idx == j: # Current method matches
                     # Iterate through all groups to find the matching group index
                     for k in range(len(x_positions)):
                         if k == m_group_idx:
                             # Draw X
                             cx = x_positions[k]
                             cy = y_top_transformed[k] + jct_adj[j][k]
                             
                             # Calculate X coordinates
                             x1 = cx - MARKER_ROW2_HEIGHT / 2
                             x2 = cx + MARKER_ROW2_HEIGHT / 2
                             y1 = cy + MARKER_GAP_Y
                             y2 = cy + MARKER_GAP_Y + MARKER_ROW2_HEIGHT
                             
                             # Draw lines
                             ax.plot([x1, x2], [y1, y2], color=MARKER_COLOR, linewidth=MARKER_ROW2_LINE_WIDTH, zorder=50)
                             ax.plot([x1, x2], [y2, y1], color=MARKER_COLOR, linewidth=MARKER_ROW2_LINE_WIDTH, zorder=50)
            
            safe_values_ms = np.where(values_ms == 0, 1e-9, values_ms)
            speedup = baseline_vals / safe_values_ms
            speedup_transformed = transform_y(speedup, right_ticks, right_positions)
            
            # Apply Speedup Adjustment
            speedup_transformed = speedup_transformed + speedup_adj[j]
            
            if method != baseline_name:
                line_zorder = 10
                line, = ax2.plot(indices, speedup_transformed, color=line_color_map[method], marker=marker_map[method],
                         markersize=10, linewidth=LINE_WIDTH, linestyle=lines_map[method],
                         markeredgecolor='white', markeredgewidth=1.0, zorder=line_zorder)
                
                if method == "KVServe":
                     for sx, sy, val in zip(indices, speedup_transformed, speedup):
                        ax2.text(sx + 0.02, sy + 0.05, f"{val:.2f}", color=line_color_map[method], 
                                 fontweight='bold', fontsize=10, ha='left', va='bottom', zorder=30)

        ax.set_title(title, fontweight='bold', fontsize=14, pad=10, x=ROW2_LAYOUT['title_x'], y=ROW2_LAYOUT['title_y'])
        
        ax.set_yticks(left_positions)
        ax.set_yticklabels([f"{y}" for y in left_ticks])
        ax.set_ylim(0, left_positions[-1])
        
        r_max_idx = right_positions[-1]
        ax2.set_yticks(right_positions[1:])
        ax2.set_yticklabels([f"{y}" for y in right_ticks[1:]])
        if 0 < right_axis_start_pos < 1:
            y_min_right = - (right_axis_start_pos * r_max_idx) / (1 - right_axis_start_pos)
            ax2.set_ylim(y_min_right, r_max_idx)
        else:
            ax2.set_ylim(0, r_max_idx)

        ax.set_xticks(indices)
        ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
        ax.set_xlim(ROW2_LAYOUT['xlim_left'], num_groups + ROW2_LAYOUT['xlim_right_offset']) # Apply narrow margins for Row 2
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

    # --- Common Elements ---
    
    # Global Y Labels
    fig.text(0.09, 0.5, "JCT (s)", rotation='vertical', va='center', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.92, 0.5, "Speedup (x)", rotation=270, va='center', ha='center', fontsize=16, fontweight='bold')
    
    # Legend
    # Removed JCT and Speedup entries as requested
    
    # New handle for Acc. < Thres
    h_marker = Line2D([0], [0], color=MARKER_COLOR, marker='x', markersize=8,
                      linestyle='None', markeredgewidth=MARKER_ROW1_LINE_WIDTH, markeredgecolor=MARKER_COLOR)
    l_marker = 'Acc. < Thres'

    type_handles = [h_marker]
    type_labels = [l_marker]
    
    custom_legend = create_combined_legend_box(fig, legend_method_handles, legend_method_labels, type_handles, type_labels)
    fig.add_artist(custom_legend)
    
    # Save
    output_filename = "jct_latency_all_combined.pdf"
    plt.tight_layout()
    # 调整布局以适应顶部图例和双排
    plt.subplots_adjust(top=0.88, bottom=0.08, wspace=0.2, hspace=0.2, left=0.115, right=0.895)
    plt.savefig(output_filename)
    print(f"Plot generated: {output_filename}")

if __name__ == "__main__":
    draw_chart()
