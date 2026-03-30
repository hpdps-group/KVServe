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
LINE_WIDTH = 4  # 折线图线宽

# 用户指定 SLO (秒)
SLO_VALUE_LEFT = 1.5
SLO_VALUE_RIGHT = 3.0

# ================= 1. 数据准备区域 (从CSV读取) =================

def load_data_from_csv(file_path):
    data_list = []
    if not os.path.exists(file_path):
        # Fallback for relative path
        file_path = os.path.join(os.path.dirname(__file__), "bandwidth.csv")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_list.append(row)
            
    # 提取 Datasets (原 Models)
    datasets = []
    for row in data_list:
        # 兼容 key 名: DateSet / Dataset / Model
        val = row.get('Dateset') or row.get('Dataset') or row.get('Model')
        if val and val not in datasets:
            datasets.append(val)
            
    # 提取 Bandwidths (作为X轴)
    bandwidths = []
    for row in data_list:
        if int(row['Bandwidth']) not in bandwidths:
            bandwidths.append(int(row['Bandwidth']))
    bandwidths.sort()

    # 提取 Methods
    methods = []
    for row in data_list:
        if row['Method'] not in methods:
            methods.append(row['Method'])
            
    # 构建数据字典: data[dataset][method] = {bw: jct_val, ...}
    data_structure = {d: {method: {} for method in methods} for d in datasets}
    
    for row in data_list:
        d = row.get('Dateset') or row.get('Dataset') or row.get('Model')
        bw = int(row['Bandwidth'])
        method = row['Method']
        val = float(row['JCT'])
        data_structure[d][method][bw] = val
            
    return datasets, bandwidths, methods, data_structure

# CSV 文件路径
CSV_PATH = "/root/workspace/KVServe/evaluation/evaluation/jct_latency/prefix_caching/ttft_bandwidth.csv"

# 加载数据
DATASETS, BANDWIDTHS, METHODS, DATA_DICT = load_data_from_csv(CSV_PATH)

print("Loaded DATASETS:", DATASETS)
print("Loaded BANDWIDTHS:", BANDWIDTHS)
print("Loaded METHODS:", METHODS)

# 颜色池
METHOD_COLORS = ['#868686', '#FFBB78', '#e41a1c', '#9575cd', '#4dd0e1'] 
current_colors = METHOD_COLORS[:len(METHODS)]
COLOR_MAP = {name: col for name, col in zip(METHODS, current_colors)}

# Marker 池
MARKERS = ['o', 's', 'p', 'h', 'v', '<', '>']
MARKER_MAP = {name: m for name, m in zip(METHODS, MARKERS)}

LINES = ['--', '--', '-']
LINES_MAP = {name: l for name, l in zip(METHODS, LINES)}

# ================= 2. 纵轴刻度配置 (自定义均匀刻度) =================

# 左图 (Llama) JCT (s)
# 原始数据范围大致在 1300-2700 ms -> 1.3 - 2.7 s
CUSTOM_Y_TICKS_LEFT = [0, 0.5, 1.0, 2.0, 3.0, 5.0]

# 右图 (Qwen) JCT (s)
# 假设范围类似，如果有差异可以在此调整
CUSTOM_Y_TICKS_RIGHT = [0, 1.0, 2.0, 3.0, 5.0]

# ================= 3. 辅助函数 (坐标轴变换逻辑) =================

def map_values_to_linear_ticks(values, ticks):
    ticks = np.array(ticks)
    return np.interp(values, ticks, np.arange(len(ticks)))

def transform_y(y_values, ticks):
    return map_values_to_linear_ticks(y_values, ticks)

# ================= 核心修改函数：构建自定义图例 =================
def create_legend_box(fig, handles, labels):
    """
    使用 offsetbox 构建单行居中图例
    """
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

    # 构建一行 Items
    items = [create_legend_item(h, l) for h, l in zip(handles, labels)]
    packer = HPacker(children=items, align="center", pad=0, sep=60)
    
    # 放入容器
    anchored_box = AnchoredOffsetbox(
        loc='lower center',
        child=packer,
        pad=0,
        frameon=True,
        bbox_to_anchor=(0.54, 0.90), # 放在顶部
        bbox_transform=fig.transFigure,
        borderpad=0.
    )
    
    # 设置边框样式 (圆角)
    anchored_box.patch.set_boxstyle("round,pad=0.4")
    anchored_box.patch.set_linewidth(1.2)
    anchored_box.patch.set_edgecolor('black')
    anchored_box.patch.set_facecolor('white')
    
    return anchored_box

# ================= 4. 绘图主逻辑 =================

def plot_single_ax(ax, dataset_name, y_ticks, slo_val=None):
    # 准备 X 轴坐标 (均匀分布)
    x_indices = np.arange(len(BANDWIDTHS))
    
    # Store data for later use (fill_between and annotation)
    y_data_trans = {}
    y_data_raw = {}

    for method in METHODS:
        # 获取该方法在该模型下的所有带宽对应的数据
        y_vals_ms = []
        for bw in BANDWIDTHS:
            val = DATA_DICT[dataset_name][method].get(bw, np.nan)
            y_vals_ms.append(val)
        
        y_vals_ms = np.array(y_vals_ms)
        y_vals_sec = y_vals_ms / 1000.0
        
        # 转换 Y 坐标
        y_transformed = transform_y(y_vals_sec, y_ticks)
        
        # Store
        y_data_trans[method] = y_transformed
        y_data_raw[method] = y_vals_sec
        
        # 绘制折线
        ax.plot(x_indices, y_transformed,
                color=COLOR_MAP[method],
                marker=MARKER_MAP[method],
                markersize=10,
                linewidth=LINE_WIDTH,
                linestyle=LINES_MAP[method],
                markeredgecolor='white',
                markeredgewidth=1.0,
                label=method,
                zorder=10)

    # --- 新增功能: SLO 虚线 ---
    if slo_val is not None:
        slo_trans = transform_y([slo_val], y_ticks)[0]
        # 使用 KVServe 的颜色
        slo_color = COLOR_MAP.get("KVServe", "red")
        ax.axhline(y=slo_trans, color=slo_color, linestyle='--', linewidth=2.5, zorder=5, xmin=0.05, xmax=0.93)
        
        # 可选：添加 SLO 文字标注
        ax.text(len(BANDWIDTHS) / 2 + 0.25, slo_trans + 0.2, f"SLO={slo_val}s", 
                color=slo_color, ha='right', va='bottom', fontweight='bold', fontsize=13)

    # --- 新增功能: 阴影区域与加速比标注 ---
    target_default = "Default(BF16)"
    target_kvserve = "KVServe"
    
    if target_default in y_data_trans and target_kvserve in y_data_trans:
        # 1. 绘制阴影
        ax.fill_between(x_indices, 
                        y_data_trans[target_default], 
                        y_data_trans[target_kvserve], 
                        color=COLOR_MAP[target_kvserve], 
                        alpha=0.1,
                        zorder=0) # 放在底层

        # 2. 添加箭头和加速比文字 (在 Bandwidth=1 处, index=0) 
        # 注意: Bandwidth可能从1开始，不一定是10，这里取index=0
        idx = 5
        y_def_trans = y_data_trans[target_default][idx]
        y_kvs_trans = y_data_trans[target_kvserve][idx]
        
        y_def_raw = y_data_raw[target_default][idx]
        y_kvs_raw = y_data_raw[target_kvserve][idx]
        
        speedup = y_def_raw / y_kvs_raw
        
        mid_y = (y_def_trans + y_kvs_trans) / 2
        
        # 文字
        ax.text(idx - 0.15, mid_y - 0.3, f"{speedup:.1f}x", 
                ha='center', va='center', 
                fontsize=13, fontweight='bold', 
                color=COLOR_MAP[target_kvserve],
                
                zorder=20)
        
        # 箭头 (向上指 Default)
        ax.annotate('', xy=(idx - 0.05, y_def_trans - 0.15), xytext=(idx - 0.05, mid_y - 0.05),
                    arrowprops=dict(arrowstyle='->', color=COLOR_MAP[target_kvserve], lw=2.0),
                    zorder=20)
        
        # 箭头 (向下指 KVServe)
        ax.annotate('', xy=(idx - 0.05, y_kvs_trans + 0.15), xytext=(idx - 0.05, mid_y - 0.55),
                    arrowprops=dict(arrowstyle='->', color=COLOR_MAP[target_kvserve], lw=2.0),
                    zorder=20)

    # 设置 Y 轴
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_yticklabels([f"{y}" for y in y_ticks])
    ax.set_ylim(0, len(y_ticks) - 1)
    ax.set_ylabel("TTFT (s)", fontweight='bold', fontsize=16)
    
    # 设置 X 轴
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(bw) for bw in BANDWIDTHS], fontsize=14, fontweight='bold')
    
    # 调整 X 轴范围 (增加左侧留白，同时拉长右侧以缩小刻度间距)
    ax.set_xlim(-0.3, len(BANDWIDTHS) - 0.7)

    # 设置标题
    ax.text(0.5, 1.05, dataset_name, transform=ax.transAxes, 
            ha='center', va='top', fontsize=14, fontweight='bold')

    # 网格与边框
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    
    # 左图: 2WikiMQA (假设 DATASETS[0] 是 2WikiMQA)
    name1 = "2WikiMQA"
    if name1 not in DATASETS and len(DATASETS) > 0:
        name1 = DATASETS[0]
        
    plot_single_ax(ax1, name1, CUSTOM_Y_TICKS_LEFT, slo_val=SLO_VALUE_LEFT)
    
    # 右图: HotpotQA
    name2 = "HotpotQA"
    if name2 not in DATASETS and len(DATASETS) > 1:
        name2 = DATASETS[1]
        
    # 不需要显示Y轴label
    plot_single_ax(ax2, name2, CUSTOM_Y_TICKS_RIGHT, slo_val=SLO_VALUE_RIGHT)
    ax2.set_ylabel("")
    
    # 创建图例 Handles
    legend_handles = []
    legend_labels = []
    for m in METHODS:
        # 创建 Line2D 用于图例
        line = Line2D([0], [0], 
                      color=COLOR_MAP[m],
                      marker=MARKER_MAP[m],
                      markersize=10,
                      linewidth=LINE_WIDTH,
                      linestyle=LINES_MAP[m],
                      markeredgecolor='white',
                      markeredgewidth=1.5) # 图例稍微粗一点点
        legend_handles.append(line)
        legend_labels.append(m)
        
    # 添加自定义图例
    custom_legend = create_legend_box(fig, legend_handles, legend_labels)
    fig.add_artist(custom_legend)
    
    # 共用 X 轴标签
    fig.supxlabel("Bandwidth (Gbps)", fontweight='bold', fontsize=16, x=0.54)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.18, wspace=0.15)
    
    output_filename = "bandwidth_jct_line_chart.pdf"
    plt.savefig(output_filename)
    print(f"Plot generated: {output_filename}")

if __name__ == "__main__":
    main()

