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
    "ytick.major.pad": 2, 
})

# ================= 配置参数 (自定义) =================
LINE_WIDTH = 3  # (Unused now)

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
    [0, 2, 4, 6, 13],
    [0, 1, 2, 3, 4],
    [0, 2, 4, 6, 7]
]

# Row 1 Split Config (Custom scale)
ROW1_SPLIT_CONFIG = [
    {'index': 3, 'ratio': 0.85},
    {'index': 3, 'ratio': 0.85},
    {'index': 3, 'ratio': 0.85},
    {'index': 3, 'ratio': 0.85},
]

# Row 1 Layout Customization
ROW1_LAYOUT = {
    'group_width': 0.9,           
    'bar_width_ratio': 0.8,       
    'title_x': 0.7,              
    'title_y': 0.85,               
    'xlim_offset': 0.5,           
}

# Row 1 Markers: List of lists. Each inner list contains tuples (group_idx, method_idx) to mark.
ROW1_MARKERS = [
    [], 
    [],       
    [(0, 1), (1, 1), (2, 1), (3, 1)],       
    [(0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2)]        
]

# Row 1 Annotation Targets (Indices to annotate for each plot)
ROW1_ANNOTATE_INDICES = [
    [0], # Plot 1: Only 1st group (Leftmost, 5090)
    [0], # Plot 2
    [0], # Plot 3
    [0]  # Plot 4
]

# Row 1 Annotation Offsets (X, Y)
# One list per subplot, containing (x_offset, y_offset) tuples for the first 4 groups
ROW1_TEXT_OFFSETS = [
    [(-0.05, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    [(-0.05, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    [(-0.05, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    [(-0.05, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
]

# Row 1 Visual Adjustments (Target Methods: CacheGen, KIVI, KVServe)
ROW1_ADJUST_JCT_VAL = 0.2

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

# Row 2 Split Config
ROW2_SPLIT_CONFIG = [
    {'index': 2, 'ratio': 0.8},
    {'index': 2, 'ratio': 0.7},
]

# Row 2 Layout Customization
ROW2_LAYOUT = {
    'group_width': 0.9,           
    'bar_width_ratio': 0.85,      
    'title_x': 0.5,               
    'title_y': 0.8,               
    'xlim_left': -0.57,           
    'xlim_right_offset': -0.43    
}

# Row 2 Markers
ROW2_MARKERS = [
    [(4, 1), (5, 2), (4, 2), ], 
    []  
]

# Row 2 Annotation Targets (X Labels to annotate)
ROW2_ANNOTATE_LABELS = [
    ["HotpotQA"],
    ["HotpotQA"]
]

# Row 2 Annotation Offsets (X, Y)
ROW2_TEXT_OFFSETS = [
    [(0.05, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    [(0.05, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
]

# Row 2 Visual Adjustments
ROW2_ADJUST_JCT_VAL = 0.2

# Common Config
TARGET_METHODS = ["CacheGen", "KIVI", "KVServe"] 
LEGEND_ITEM_SPACING = 60  

# Marker Config (Red X)
MARKER_COLOR = '#e41a1c'
MARKER_ROW1_LINE_WIDTH = 3.0
MARKER_ROW2_LINE_WIDTH = 4.0
MARKER_GAP_Y = 0.1   
MARKER_ROW1_WIDTH = 0.1
MARKER_ROW1_HEIGHT = 0.15  
MARKER_ROW2_WIDTH = 0.1
MARKER_ROW2_HEIGHT = 0.16

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
            
    # 提取 X_LABELS
    x_labels = []
    for row in data_list:
        if row['X'] not in x_labels:
            x_labels.append(row['X'])
            
    # 提取 METHODS
    methods = []
    for row in data_list:
        if row['Method'] not in methods:
            methods.append(row['Method'])
            
    # 构建 DATA_MS
    data_ms = {m: [] for m in methods}
    for method in methods:
        for x_label in x_labels:
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

# 颜色池
BAR_COLORS = ['#C7C7C7', '#FFBB78', '#C5B0D5', '#e41a1c', '#9575cd', '#4dd0e1'] 
MARKERS = ['o', 's', 'p', 'h', 'v', '<', '>']
LINES = ['--', '--', '--', '-']
HATCHES = ['..', '//', '\\\\', 'xx', '..', '++']

# ================= 3. 辅助函数 (坐标轴变换逻辑) =================

def calculate_tick_positions(ticks, split_config=None):
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
    
    step_low = y_split / idx
    positions[:idx+1] = np.arange(idx+1) * step_low
    
    upper_intervals = (n - 1) - idx
    step_high = (total_height - y_split) / upper_intervals
    positions[idx:] = y_split + np.arange(upper_intervals + 1) * step_high
    
    return positions

def map_values_to_linear_ticks(values, ticks, positions=None):
    if positions is None:
        positions = np.arange(len(ticks))
    return np.interp(values, ticks, positions)

def transform_y(y_values, ticks, positions=None):
    return map_values_to_linear_ticks(y_values, ticks, positions)

# ================= 核心修改函数：构建自定义图例 =================
def create_combined_legend_box(fig, handles_row1, labels_row1, handles_row2, labels_row2):
    """
    使用 offsetbox 构建两行居中对齐、每行元素数量不同的统一图例 (仅柱状图)
    """
    
    def create_legend_item(handle, label):
        # Standard Single Item (Bar only)
        da = DrawingArea(width=22, height=10, xdescent=0, ydescent=0)
        
        if isinstance(handle, Patch):
            rect = Rectangle((0, 0), width=22, height=10,
                                facecolor=handle.get_facecolor(),
                                edgecolor=handle.get_edgecolor(),
                                hatch=handle.get_hatch(),
                                linewidth=handle.get_linewidth())
            da.add_artist(rect)
        elif isinstance(handle, Line2D):
             # For the marker 'Acc. < Thres'
             line = Line2D([11], [5], 
                          color=handle.get_color(),
                          marker=handle.get_marker(),
                          markersize=handle.get_markersize(),
                          linestyle='None',
                          markeredgewidth=handle.get_markeredgewidth(),
                          markeredgecolor=handle.get_markeredgecolor())
             da.add_artist(line)

        # 2. 创建文字区域 (TextArea)
        ta = TextArea(label, textprops=dict(color="black", size=13, family="DejaVu Sans", fontweight='bold'))
        
        # 3. 将图标和文字水平打包 (HPacker)
        return HPacker(children=[da, ta], align="center", pad=0, sep=15)

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
        bbox_to_anchor=(0.54, 0.92), # 整体位置 (相对于 fig)
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

def calculate_visual_adjustments(methods, data_ms, num_groups, jct_adjust_val, x_labels=None):
    """
    Calculate visual adjustments for JCT (Bar) only.
    """
    jct_adjustments = np.zeros((len(methods), num_groups))
    target_indices = [i for i, m in enumerate(methods) if m in TARGET_METHODS]
    
    SPECIAL_DATASETS = ["HumanEval", "GSM8K"]

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
                 # Ascending
                 jct_vals.sort(key=lambda x: x[0])
                 for rank, (val, idx) in enumerate(jct_vals):
                     if rank > 0:
                         jct_adjustments[idx][k] = 1 * rank * jct_adjust_val
            else:
                 # Descending
                 jct_vals.sort(key=lambda x: x[0], reverse=True)
                 for rank, (val, idx) in enumerate(jct_vals):
                    if rank > 0:
                        jct_adjustments[idx][k] = -1 * rank * jct_adjust_val
                    
    return jct_adjustments

def draw_speedup_annotation(ax, x_pos, y_base, y_target, speedup_val, bar_width, color, text_offset_x=0.0, text_offset_y=0.0):
    """
    Draw speedup annotation with double arrows and a horizontal cap line at the top.
    x_pos: Center X of the target (KVServe) bar.
    y_base: Visual height of Baseline bar (The top reference).
    y_target: Visual height of Target bar (The bottom reference).
    """
    # 1. Horizontal cap line at y_base (representing baseline height)
    # Projecting over the KVServe bar
    line_width = bar_width * 1.0
    ax.plot([x_pos - line_width/2, x_pos + line_width/2], [y_base, y_base], 
            color=color, linewidth=3, zorder=20)
            
    # 2. Arrows (Double headed, leaving gap for text)
    mid_y = (y_base + y_target) / 2 + text_offset_y
    mid_x = x_pos + text_offset_x
    
    # Arrow UP (pointing to y_base)
    ax.annotate('', xy=(x_pos, y_base), xytext=(x_pos, mid_y + 0.2),
                arrowprops=dict(arrowstyle='->', color=color, lw=3.0), zorder=20)
                
    # Arrow DOWN (pointing to y_target)
    ax.annotate('', xy=(x_pos, y_target), xytext=(x_pos, mid_y - 0.15),
                arrowprops=dict(arrowstyle='->', color=color, lw=3.0), zorder=20)
                
    # 3. Text
    ax.text(mid_x, mid_y, f"{speedup_val:.2f}x", ha='center', va='center',
            color=color, fontweight='bold', fontsize=12, zorder=30, 
            bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.6))

def draw_chart():
    # 画布大小
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(2, 4, figure=fig)
    
    # Row 1 axes
    row1_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    
    # Row 2 axes
    row2_axes = [
        fig.add_subplot(gs[1, 0:2]), 
        fig.add_subplot(gs[1, 2:4])  
    ]
    
    legend_method_handles = []
    legend_method_labels = []
    
    # --- Row 1 Processing (Device) ---
    for i in range(4):
        ax = row1_axes[i]
        csv_file = ROW1_FILES[i]
        title = ROW1_TITLES[i]
        left_ticks = ROW1_LEFT_TICKS[i]
        markers_to_draw = ROW1_MARKERS[i]
        split_config = ROW1_SPLIT_CONFIG[i]
        offsets = ROW1_TEXT_OFFSETS[i]
        indices_to_annotate = ROW1_ANNOTATE_INDICES[i]
        
        print(f"Processing Row 1 Plot {i+1}: {csv_file}")
        x_labels, methods, data_ms = load_data_from_csv(csv_file)
        
        # Calculate Axis Positions
        left_positions = calculate_tick_positions(left_ticks, split_config)

        # Baseline for calculation
        baseline_name = "Default(BF16)"
        if baseline_name not in data_ms:
            baseline_name = methods[0]
        
        # Calculate Adjustments (JCT only)
        num_groups = len(x_labels)
        jct_adj = calculate_visual_adjustments(methods, data_ms, num_groups, ROW1_ADJUST_JCT_VAL, x_labels)

        # 颜色映射
        current_bar_colors = BAR_COLORS[:len(methods)]
        bar_color_map = {name: col for name, col in zip(methods, current_bar_colors)}
        hatch_map = {name: h for name, h in zip(methods, HATCHES)}
        
        # 收集图例
        if i == 0 and len(legend_method_handles) == 0:
             for m in methods:
                p = Patch(facecolor=bar_color_map[m], edgecolor='black', hatch=hatch_map[m], linewidth=1.5)
                legend_method_handles.append(p)
                legend_method_labels.append(m)

        # 布局参数
        indices = np.arange(num_groups)
        group_width = ROW1_LAYOUT['group_width']
        bar_width = group_width / len(methods)
        actual_bar_width = bar_width * ROW1_LAYOUT['bar_width_ratio']
        
        # Store data for annotation: method -> list of (x_pos, y_visual_top)
        annotation_data = {m: [] for m in methods}
        baseline_vals = data_ms[baseline_name]
        
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
            y_final_tops = y_bottom_transformed + bar_heights
            
            # Store for annotation
            for grp_idx in range(num_groups):
                annotation_data[method].append((x_positions[grp_idx], y_final_tops[grp_idx]))
            
            ax.bar(x_positions, bar_heights, width=actual_bar_width, bottom=y_bottom_transformed,
                   color=bar_color_map[method], edgecolor='black', linewidth=1.5, hatch=hatch_map[method], zorder=3)
            
            # Markers (Red X)
            for m_group_idx, m_method_idx in markers_to_draw:
                if m_method_idx == j: 
                     for k in range(len(x_positions)):
                         if k == m_group_idx:
                             cx = x_positions[k]
                             cy = y_top_transformed[k] + jct_adj[j][k]
                             x1 = cx - MARKER_ROW1_WIDTH / 2
                             x2 = cx + MARKER_ROW1_WIDTH / 2
                             y1 = cy + MARKER_GAP_Y
                             y2 = cy + MARKER_GAP_Y + MARKER_ROW1_HEIGHT
                             ax.plot([x1, x2], [y1, y2], color=MARKER_COLOR, linewidth=MARKER_ROW1_LINE_WIDTH, zorder=50)
                             ax.plot([x1, x2], [y2, y1], color=MARKER_COLOR, linewidth=MARKER_ROW1_LINE_WIDTH, zorder=50)
        
        # --- Speedup Annotation ---
        target_name = "KVServe"
        if target_name in annotation_data and baseline_name in annotation_data:
            kvserve_color = bar_color_map[target_name]
            
            # Annotate specified indices
            for k in indices_to_annotate:
                if k >= num_groups:
                    continue
                
                x_kvs, y_kvs = annotation_data[target_name][k]
                x_base, y_base = annotation_data[baseline_name][k]
                
                # Raw speedup calc
                val_base = baseline_vals[k]
                val_kvs = data_ms[target_name][k]
                speedup = val_base / val_kvs if val_kvs > 1e-9 else 0
                
                # Only annotate if valid and meaningful
                if speedup > 0:
                    off_x, off_y = offsets[k] if k < len(offsets) else (0.0, 0.0)
                    draw_speedup_annotation(ax, x_kvs, y_base, y_kvs, speedup, actual_bar_width, kvserve_color, off_x, off_y)

        # Axes Styling
        ax.set_title(title, fontweight='bold', fontsize=13, pad=10, x=ROW1_LAYOUT['title_x'], y=ROW1_LAYOUT['title_y'])
        ax.set_yticks(left_positions)
        ax.set_yticklabels([f"{y}" for y in left_ticks])
        ax.set_ylim(0, left_positions[-1])
        ax.set_xticks(indices)
        ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False) 

    # --- Row 2 Processing (Dataset) ---
    for i in range(2):
        ax = row2_axes[i]
        csv_file = ROW2_FILES[i]
        title = ROW2_TITLES[i]
        left_ticks = ROW2_LEFT_TICKS[i]
        markers_to_draw = ROW2_MARKERS[i]
        split_config = ROW2_SPLIT_CONFIG[i]
        offsets = ROW2_TEXT_OFFSETS[i]
        labels_to_annotate = ROW2_ANNOTATE_LABELS[i]

        print(f"Processing Row 2 Plot {i+1}: {csv_file}")
        x_labels, methods, data_ms = load_data_from_csv(csv_file)
        
        left_positions = calculate_tick_positions(left_ticks, split_config)
        
        baseline_name = "Default(BF16)"
        if baseline_name not in data_ms:
            baseline_name = methods[0]
        
        num_groups = len(x_labels)
        jct_adj = calculate_visual_adjustments(methods, data_ms, num_groups, ROW2_ADJUST_JCT_VAL, x_labels)
        
        current_bar_colors = BAR_COLORS[:len(methods)]
        bar_color_map = {name: col for name, col in zip(methods, current_bar_colors)}
        hatch_map = {name: h for name, h in zip(methods, HATCHES)}

        indices = np.arange(num_groups)
        group_width = ROW2_LAYOUT['group_width']
        bar_width = group_width / len(methods)
        actual_bar_width = bar_width * ROW2_LAYOUT['bar_width_ratio']
        
        annotation_data = {m: [] for m in methods}
        baseline_vals = data_ms[baseline_name]
        
        for j, method in enumerate(methods):
            x_positions = indices - (group_width / 2) + (j * bar_width) + (bar_width / 2)
            values_ms = np.array(data_ms[method])
            values_sec = values_ms / 1000.0
            
            y_bottom_transformed = transform_y(0, left_ticks, left_positions)
            y_top_transformed = transform_y(values_sec, left_ticks, left_positions)
            bar_heights = y_top_transformed - y_bottom_transformed
            
            bar_heights = bar_heights + jct_adj[j]
            y_final_tops = y_bottom_transformed + bar_heights
            
            for grp_idx in range(num_groups):
                annotation_data[method].append((x_positions[grp_idx], y_final_tops[grp_idx]))
            
            ax.bar(x_positions, bar_heights, width=actual_bar_width, bottom=y_bottom_transformed,
                   color=bar_color_map[method], edgecolor='black', linewidth=1.5, hatch=hatch_map[method], zorder=3)
            
            for m_group_idx, m_method_idx in markers_to_draw:
                if m_method_idx == j:
                     for k in range(len(x_positions)):
                         if k == m_group_idx:
                             cx = x_positions[k]
                             cy = y_top_transformed[k] + jct_adj[j][k]
                             x1 = cx - MARKER_ROW2_WIDTH / 2
                             x2 = cx + MARKER_ROW2_WIDTH / 2
                             y1 = cy + MARKER_GAP_Y
                             y2 = cy + MARKER_GAP_Y + MARKER_ROW2_HEIGHT
                             ax.plot([x1, x2], [y1, y2], color=MARKER_COLOR, linewidth=MARKER_ROW2_LINE_WIDTH, zorder=50)
                             ax.plot([x1, x2], [y2, y1], color=MARKER_COLOR, linewidth=MARKER_ROW2_LINE_WIDTH, zorder=50)
        
        # --- Speedup Annotation ---
        target_name = "KVServe"
        if target_name in annotation_data and baseline_name in annotation_data:
            kvserve_color = bar_color_map[target_name]
            
            for k in range(num_groups):
                if x_labels[k] not in labels_to_annotate:
                    continue
                
                x_kvs, y_kvs = annotation_data[target_name][k]
                x_base, y_base = annotation_data[baseline_name][k]
                
                val_base = baseline_vals[k]
                val_kvs = data_ms[target_name][k]
                speedup = val_base / val_kvs if val_kvs > 1e-9 else 0
                
                if speedup > 0:
                    off_x, off_y = offsets[k] if k < len(offsets) else (0.0, 0.0)
                    draw_speedup_annotation(ax, x_kvs, y_base, y_kvs, speedup, actual_bar_width, kvserve_color, off_x, off_y)

        ax.set_title(title, fontweight='bold', fontsize=14, pad=10, x=ROW2_LAYOUT['title_x'], y=ROW2_LAYOUT['title_y'])
        ax.set_yticks(left_positions)
        ax.set_yticklabels([f"{y}" for y in left_ticks])
        ax.set_ylim(0, left_positions[-1])
        ax.set_xticks(indices)
        ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
        ax.set_xlim(ROW2_LAYOUT['xlim_left'], num_groups + ROW2_LAYOUT['xlim_right_offset'])
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # --- Common Elements ---
    
    # Global Y Labels
    fig.text(0.09, 0.5, "JCT (s)", rotation='vertical', va='center', ha='center', fontsize=16, fontweight='bold')
    
    # Legend
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
    plt.subplots_adjust(top=0.88, bottom=0.08, wspace=0.1, hspace=0.2, left=0.115, right=0.98) 
    plt.savefig(output_filename)
    print(f"Plot generated: {output_filename}")

if __name__ == "__main__":
    draw_chart()
