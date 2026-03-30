import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea, DrawingArea
from matplotlib.patches import Rectangle
import csv
import os

# ================= 通用配置参数 (Style) =================
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

LINE_WIDTH = 4  # 折线图线宽

# ================= 样式配置字典 (用户修改处) =================
# key: Method Name (必须与CSV中的Method字段一致)
# value: { 'color': ..., 'marker': ..., 'linestyle': ... }
# 可以在此处添加更多方法或修改现有方法的样式
STYLE_CONFIG = {
    "Default(BF16)": {
        "color": "#868686",
        "marker": "o",
        "linestyle": "--"
    },
    "CacheGen": {
        "color": "#FFBB78",
        "marker": "s",
        "linestyle": "--"
    },
    # 示例：如果有其他方法，请在此处添加
    "KIVI": {
        "color": "#C5B0D5",
        "marker": "p",
        "linestyle": "--"
    },
    "KVServe": {
        "color": "#e41a1c",
        "marker": "h",
        "linestyle": "-"
    },
}

# ==============================================================================
#                               PART 1: JCT 配置与函数
# ==============================================================================

# --- JCT 配置 ---
JCT_CSV_PATH = "/root/workspace/KVServe/evaluation/evaluation/jct_latency/bandwidth_change/jct_bandwidth.csv"

# 左图 (Llama) JCT (s) 刻度
JCT_Y_TICKS_LEFT = [1.3, 1.4, 1.5, 1.6, 2.0, 3.0]

# 右图 (Qwen) JCT (s) 刻度
JCT_Y_TICKS_RIGHT = [1.0, 1.5, 2.0, 2.5, 3.0]

# --- JCT 数据加载 ---
def load_jct_data(file_path):
    data_list = []
    if not os.path.exists(file_path):
        print(f"Warning: JCT file not found at {file_path}")
        return [], [], [], {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_list.append(row)
            
    models = []
    for row in data_list:
        if row['Model'] not in models:
            models.append(row['Model'])
            
    bandwidths = []
    for row in data_list:
        if int(row['Bandwidth']) not in bandwidths:
            bandwidths.append(int(row['Bandwidth']))
    bandwidths.sort()

    methods = []
    for row in data_list:
        if row['Method'] not in methods:
            methods.append(row['Method'])
            
    data_structure = {model: {method: {} for method in methods} for model in models}
    
    for row in data_list:
        m = row['Model']
        bw = int(row['Bandwidth'])
        method = row['Method']
        val = float(row['JCT'])
        data_structure[m][method][bw] = val
            
    return models, bandwidths, methods, data_structure

# ==============================================================================
#                               PART 2: TTFT 配置与函数
# ==============================================================================

# --- TTFT 配置 ---
TTFT_CSV_PATH = "/root/workspace/KVServe/evaluation/evaluation/jct_latency/prefix_caching/ttft_bandwidth.csv"

# 用户指定 SLO (秒)
TTFT_SLO_LEFT = 1.5
TTFT_SLO_RIGHT = 1.5

# 左图 (2WikiMQA) TTFT (s) 刻度
TTFT_Y_TICKS_LEFT = [1.3, 1.4, 1.5, 1.6, 2.0, 3.0]

# 右图 (HotpotQA) TTFT (s) 刻度
TTFT_Y_TICKS_RIGHT = [1.0, 1.5, 2.0, 2.5, 3.0]

# --- TTFT 数据加载 ---
def load_ttft_data(file_path):
    data_list = []
    if not os.path.exists(file_path):
        print(f"Warning: TTFT file not found at {file_path}")
        return [], [], [], {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_list.append(row)
            
    datasets = []
    for row in data_list:
        val = row.get('Dateset') or row.get('Dataset') or row.get('Model')
        if val and val not in datasets:
            datasets.append(val)
            
    bandwidths = []
    for row in data_list:
        if int(row['Bandwidth']) not in bandwidths:
            bandwidths.append(int(row['Bandwidth']))
    bandwidths.sort()

    methods = []
    for row in data_list:
        if row['Method'] not in methods:
            methods.append(row['Method'])
            
    data_structure = {d: {method: {} for method in methods} for d in datasets}
    
    for row in data_list:
        d = row.get('Dateset') or row.get('Dataset') or row.get('Model')
        bw = int(row['Bandwidth'])
        method = row['Method']
        val = float(row['JCT'])
        data_structure[d][method][bw] = val
            
    return datasets, bandwidths, methods, data_structure

# ==============================================================================
#                               PART 3: 辅助函数 (共享)
# ==============================================================================

def map_values_to_linear_ticks(values, ticks):
    ticks = np.array(ticks)
    return np.interp(values, ticks, np.arange(len(ticks)))

def transform_y(y_values, ticks):
    return map_values_to_linear_ticks(y_values, ticks)

def create_legend_box(fig, handles, labels):
    def create_legend_item(handle, label):
        da = DrawingArea(width=22, height=10, xdescent=0, ydescent=0)
        if isinstance(handle, Line2D):
            line = Line2D([0, 11, 22], [5, 5, 5],
                          color=handle.get_color(),
                          linewidth=handle.get_linewidth()-1,
                          linestyle=handle.get_linestyle(),
                          marker=handle.get_marker(),
                          markersize=handle.get_markersize(),
                          markeredgecolor=handle.get_markeredgecolor(),
                          markeredgewidth=handle.get_markeredgewidth(),
                          markevery=[1])
            da.add_artist(line)
        ta = TextArea(label, textprops=dict(color="black", size=13, family="DejaVu Sans", fontweight='bold'))
        return HPacker(children=[da, ta], align="center", pad=0, sep=5)

    items = [create_legend_item(h, l) for h, l in zip(handles, labels)]
    packer = HPacker(children=items, align="center", pad=0, sep=60)
    
    anchored_box = AnchoredOffsetbox(
        loc='lower center',
        child=packer,
        pad=0,
        frameon=True,
        bbox_to_anchor=(0.54, 0.92), # 放在顶部
        bbox_transform=fig.transFigure,
        borderpad=0.
    )
    
    anchored_box.patch.set_boxstyle("round,pad=0.4")
    anchored_box.patch.set_linewidth(1.2)
    anchored_box.patch.set_edgecolor('black')
    anchored_box.patch.set_facecolor('white')
    
    return anchored_box

# ==============================================================================
#                               PART 4: 绘图逻辑
# ==============================================================================

def plot_jct_ax(ax, model_name, y_ticks, data_dict, bandwidths, methods, color_map, marker_map, lines_map):
    x_indices = np.arange(len(bandwidths))
    y_data_trans = {}
    y_data_raw = {}

    for method in methods:
        y_vals_ms = []
        for bw in bandwidths:
            val = data_dict[model_name][method].get(bw, np.nan)
            y_vals_ms.append(val)
        
        y_vals_ms = np.array(y_vals_ms)
        y_vals_sec = y_vals_ms / 1000.0
        y_transformed = transform_y(y_vals_sec, y_ticks)
        
        y_data_trans[method] = y_transformed
        y_data_raw[method] = y_vals_sec
        
        ax.plot(x_indices, y_transformed,
                color=color_map[method],
                marker=marker_map[method],
                markersize=10,
                linewidth=LINE_WIDTH,
                linestyle=lines_map[method],
                markeredgecolor='white',
                markeredgewidth=1.0,
                label=method,
                zorder=10)

    # 阴影与加速比
    target_default = "Default(BF16)"
    target_kvserve = "KVServe"
    if target_default in y_data_trans and target_kvserve in y_data_trans:
        ax.fill_between(x_indices, y_data_trans[target_default], y_data_trans[target_kvserve], 
                        color=color_map[target_kvserve], alpha=0.1, zorder=0)
        
        idx = 0
        y_def_trans = y_data_trans[target_default][idx]
        y_kvs_trans = y_data_trans[target_kvserve][idx]
        y_def_raw = y_data_raw[target_default][idx]
        y_kvs_raw = y_data_raw[target_kvserve][idx]
        speedup = y_def_raw / y_kvs_raw
        mid_y = (y_def_trans + y_kvs_trans) / 2
        
        ax.text(idx + 0.05, mid_y, f"{speedup:.1f}x", ha='center', va='center', 
                fontsize=13, fontweight='bold', color=color_map[target_kvserve], zorder=20)
        ax.annotate('', xy=(idx + 0.05, y_def_trans - 0.15), xytext=(idx + 0.05, mid_y + 0.15),
                    arrowprops=dict(arrowstyle='->', color=color_map[target_kvserve], lw=2.0), zorder=20)
        ax.annotate('', xy=(idx + 0.05, y_kvs_trans + 0.15), xytext=(idx + 0.05, mid_y - 0.15),
                    arrowprops=dict(arrowstyle='->', color=color_map[target_kvserve], lw=2.0), zorder=20)

    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_yticklabels([f"{y}" for y in y_ticks])
    ax.set_ylim(0, len(y_ticks) - 1)
    ax.set_ylabel("JCT (s)", fontweight='bold', fontsize=16)
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(bw) for bw in bandwidths], fontsize=14, fontweight='bold')
    ax.set_xlim(-0.3, len(bandwidths) - 0.7)
    
    ax.text(0.5, 0.95, model_name, transform=ax.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_ttft_ax(ax, dataset_name, y_ticks, slo_val, data_dict, bandwidths, methods, color_map, marker_map, lines_map):
    x_indices = np.arange(len(bandwidths))
    y_data_trans = {}
    y_data_raw = {}

    for method in methods:
        y_vals_ms = []
        for bw in bandwidths:
            val = data_dict[dataset_name][method].get(bw, np.nan)
            y_vals_ms.append(val)
        
        y_vals_ms = np.array(y_vals_ms)
        y_vals_sec = y_vals_ms / 1000.0
        y_transformed = transform_y(y_vals_sec, y_ticks)
        
        y_data_trans[method] = y_transformed
        y_data_raw[method] = y_vals_sec
        
        ax.plot(x_indices, y_transformed,
                color=color_map[method],
                marker=marker_map[method],
                markersize=10,
                linewidth=LINE_WIDTH,
                linestyle=lines_map[method],
                markeredgecolor='white',
                markeredgewidth=1.0,
                label=method,
                zorder=10)

    if slo_val is not None:
        slo_trans = transform_y([slo_val], y_ticks)[0]
        slo_color = color_map.get("KVServe", "red")
        ax.axhline(y=slo_trans, color=slo_color, linestyle='--', linewidth=2.5, zorder=5)
        ax.text(len(bandwidths)-1, slo_trans + 0.2, f"SLO={slo_val}s", 
                color=slo_color, ha='right', va='bottom', fontweight='bold', fontsize=13)

    target_default = "Default(BF16)"
    target_kvserve = "KVServe"
    if target_default in y_data_trans and target_kvserve in y_data_trans:
        ax.fill_between(x_indices, y_data_trans[target_default], y_data_trans[target_kvserve], 
                        color=color_map[target_kvserve], alpha=0.1, zorder=0)
        
        idx = 0
        y_def_trans = y_data_trans[target_default][idx]
        y_kvs_trans = y_data_trans[target_kvserve][idx]
        y_def_raw = y_data_raw[target_default][idx]
        y_kvs_raw = y_data_raw[target_kvserve][idx]
        speedup = y_def_raw / y_kvs_raw
        mid_y = (y_def_trans + y_kvs_trans) / 2
        
        ax.text(idx + 0.05, mid_y, f"{speedup:.1f}x", ha='center', va='center', 
                fontsize=13, fontweight='bold', color=color_map[target_kvserve], zorder=20)
        ax.annotate('', xy=(idx + 0.05, y_def_trans - 0.15), xytext=(idx + 0.05, mid_y + 0.15),
                    arrowprops=dict(arrowstyle='->', color=color_map[target_kvserve], lw=2.0), zorder=20)
        ax.annotate('', xy=(idx + 0.05, y_kvs_trans + 0.15), xytext=(idx + 0.05, mid_y - 0.15),
                    arrowprops=dict(arrowstyle='->', color=color_map[target_kvserve], lw=2.0), zorder=20)

    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_yticklabels([f"{y}" for y in y_ticks])
    ax.set_ylim(0, len(y_ticks) - 1)
    ax.set_ylabel("TTFT (s)", fontweight='bold', fontsize=16)
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(bw) for bw in bandwidths], fontsize=14, fontweight='bold')
    ax.set_xlim(-0.3, len(bandwidths) - 0.7)
    
    ax.text(0.5, 0.95, dataset_name, transform=ax.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def main():
    # 1. 加载数据
    jct_models, jct_bandwidths, jct_methods, jct_data = load_jct_data(JCT_CSV_PATH)
    ttft_datasets, ttft_bandwidths, ttft_methods, ttft_data = load_ttft_data(TTFT_CSV_PATH)
    
    # 2. 统一 Methods 和 样式映射
    # 获取所有出现的方法
    all_methods = sorted(list(set(jct_methods + ttft_methods)))
    
    # 构建映射 (从 STYLE_CONFIG 中读取)
    color_map = {}
    marker_map = {}
    lines_map = {}
    
    for m in all_methods:
        if m in STYLE_CONFIG:
            color_map[m] = STYLE_CONFIG[m]['color']
            marker_map[m] = STYLE_CONFIG[m]['marker']
            lines_map[m] = STYLE_CONFIG[m]['linestyle']
        else:
            print(f"Warning: Method '{m}' not found in STYLE_CONFIG. Using default fallback style.")
            # Fallback style
            color_map[m] = 'black'
            marker_map[m] = 'x'
            lines_map[m] = '-'
    
    # 3. 创建画布 (2行2列)
    # 纵向长度变成两倍 (原 10x4 -> 10x8)
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    (ax1, ax2), (ax3, ax4) = axes
    
    # --- Row 1: JCT ---
    print("Plotting JCT Row...")
    if len(jct_models) > 0:
        plot_jct_ax(ax1, jct_models[0], JCT_Y_TICKS_LEFT, jct_data, jct_bandwidths, jct_methods, color_map, marker_map, lines_map)
    if len(jct_models) > 1:
        plot_jct_ax(ax2, jct_models[1], JCT_Y_TICKS_RIGHT, jct_data, jct_bandwidths, jct_methods, color_map, marker_map, lines_map)
        ax2.set_ylabel("") # 右侧不显示Y轴标签

    # --- Row 2: TTFT ---
    print("Plotting TTFT Row...")
    # 假设顺序: 2WikiMQA, HotpotQA
    name1 = "2WikiMQA"
    if name1 not in ttft_datasets and len(ttft_datasets) > 0:
        name1 = ttft_datasets[0]
    
    name2 = "HotpotQA"
    if name2 not in ttft_datasets and len(ttft_datasets) > 1:
        name2 = ttft_datasets[1]
        
    plot_ttft_ax(ax3, name1, TTFT_Y_TICKS_LEFT, TTFT_SLO_LEFT, ttft_data, ttft_bandwidths, ttft_methods, color_map, marker_map, lines_map)
    plot_ttft_ax(ax4, name2, TTFT_Y_TICKS_RIGHT, TTFT_SLO_RIGHT, ttft_data, ttft_bandwidths, ttft_methods, color_map, marker_map, lines_map)
    ax4.set_ylabel("") # 右侧不显示Y轴标签

    # 4. 图例 (共享)
    legend_handles = []
    legend_labels = []
    
    # 优先显示 JCT 的 methods 顺序，如果一致的话
    display_methods = jct_methods if len(jct_methods) >= len(ttft_methods) else all_methods
    
    # 过滤掉不在 STYLE_CONFIG 中的方法，或者确保它们有样式
    valid_display_methods = [m for m in display_methods if m in color_map]

    for m in valid_display_methods:
        line = Line2D([0], [0], 
                      color=color_map[m],
                      marker=marker_map[m],
                      markersize=10,
                      linewidth=LINE_WIDTH,
                      linestyle=lines_map[m],
                      markeredgecolor='white',
                      markeredgewidth=1.5)
        legend_handles.append(line)
        legend_labels.append(m)
        
    custom_legend = create_legend_box(fig, legend_handles, legend_labels)
    fig.add_artist(custom_legend)
    
    # 5. 共享 X 轴标签
    # 放在底部居中
    fig.supxlabel("Bandwidth (Gbps)", fontweight='bold', fontsize=16, x=0.54, y=-0.02)
    
    # 6. 布局调整
    plt.tight_layout()
    # top留给图例，bottom留给xlabel，hspace留给两行之间的间距
    plt.subplots_adjust(top=0.88, bottom=0.10, wspace=0.1, hspace=0.15)
    
    output_filename = "combined_bandwidth_chart.pdf"
    plt.savefig(output_filename)
    print(f"Plot generated: {output_filename}")

if __name__ == "__main__":
    main()
