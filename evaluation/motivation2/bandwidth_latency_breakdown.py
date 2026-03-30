import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# ================= 新增 Import =================
# 用于构建自定义布局图例
from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea, DrawingArea

# ================= 配置参数 =================
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

# ================= 数据配置 =================
ORIGINAL_DATA_SIZE_GB = 1.25

# 压缩方案数据
COMPRESSION_SCENARIOS = [
    {
        "name": "Cachegen",
        "compressed_size_gb": 0.14492,
        "compression_time": 0.0098 + 0.0473,
        "decompression_time": 0.0074 + 0.0435,
    },
    {
        "name": "KIVI",
        "compressed_size_gb": 0.23443,
        "compression_time": 0.0402,
        "decompression_time": 0.0144,
    },
    {
        "name": "MixHQ",
        "compressed_size_gb": 0.21654,
        "compression_time": 0.0032 + 0.0164,
        "decompression_time": 0.0019 + 0.0079,
    },
]

# 方法列表和颜色映射
method_names = ["Default"] + [s["name"] for s in COMPRESSION_SCENARIOS]

# 左侧折线图颜色列表（可独立修改）
curve_colors = ['#868686', '#FFBB78', '#C5B0D5', '#e41a1c']
# 右侧柱形图颜色列表（可独立修改）
bar_colors   = ['#C7C7C7', '#FFBB78', '#C5B0D5', '#e41a1c']

curve_color_map = {name: col for name, col in zip(method_names, curve_colors)}
bar_color_map   = {name: col for name, col in zip(method_names, bar_colors)}

# 纹理样式
hatch_comp = '//'
hatch_trans = '..'
hatch_decomp = '\\\\'

# ================= 辅助函数 =================
def calculate_transmission_time(data_size_gb, bandwidth_gbps):
    bandwidth_gbps = np.maximum(bandwidth_gbps, 1e-9)
    return (data_size_gb * 8.0) / bandwidth_gbps

# ================= 坐标轴变换逻辑 (独立配置) =================
CUSTOM_X_TICKS = [1, 50, 100, 150, 200, 250, 300, 350]
LEFT_Y_TICKS = [0, 50, 100, 150, 200, 250, 300, 400]
RIGHT_Y_TICKS = [0, 50, 100, 150, 200]

def map_values_to_linear_ticks(values, ticks):
    ticks = np.array(ticks)
    return np.interp(values, ticks, np.arange(len(ticks)))

def transform_y(y_values_ms, ticks):
    return map_values_to_linear_ticks(y_values_ms, ticks)

# ================= 左图配置 (Curve) =================
NUM_POINTS = 300
ZOOM_CONFIG = {
    "x_min": 5, "x_max": 17,
    "y_min": 150, "y_max": 350, # ms
    "loc": "center", "width": "50%", "height": "50%",
    "bbox_to_anchor": (0.05, 0.2, 1, 1)
}

def draw_curve_content(target_ax, y_ticks):
    plot_x_indices = np.linspace(0, len(CUSTOM_X_TICKS) - 1, NUM_POINTS)
    real_bandwidths = np.interp(plot_x_indices, np.arange(len(CUSTOM_X_TICKS)), CUSTOM_X_TICKS)
    y_max_limit = y_ticks[-1]

    # 1. Default
    original_latency = calculate_transmission_time(ORIGINAL_DATA_SIZE_GB, real_bandwidths) * 1000 
    original_plot_data = original_latency.copy()
    original_plot_data[original_plot_data > y_max_limit] = np.nan
    target_ax.plot(
        plot_x_indices,
        transform_y(original_plot_data, y_ticks),
        linewidth=2.5,
        linestyle='--',
        color=curve_color_map["Default"],
    )

    # 2. Compression Scenarios
    for scenario in COMPRESSION_SCENARIOS:
        total_latency = (
            scenario["compression_time"]
            + scenario["decompression_time"]
            + calculate_transmission_time(scenario["compressed_size_gb"], real_bandwidths)
        ) * 1000
        plot_data = total_latency.copy()
        plot_data[plot_data > y_max_limit] = np.nan
        target_ax.plot(
            plot_x_indices,
            transform_y(plot_data, y_ticks),
            linewidth=2.5,
            color=curve_color_map[scenario["name"]],
        )

def plot_left_panel(ax):
    draw_curve_content(ax, LEFT_Y_TICKS)
    ax.set_xticks(np.arange(len(CUSTOM_X_TICKS)))
    ax.set_xticklabels([str(x) for x in CUSTOM_X_TICKS])
    ax.set_xlim(0, len(CUSTOM_X_TICKS) - 1)
    ax.set_yticks(np.arange(len(LEFT_Y_TICKS)))
    ax.set_yticklabels([str(y) for y in LEFT_Y_TICKS])
    ax.set_ylim(0, len(LEFT_Y_TICKS) - 1)
    ax.set_ylabel("Latency (ms)", fontweight='bold', fontsize=15)
    ax.grid(True, linestyle='--', alpha=0.7)

    # 放大镜
    axins = inset_axes(ax, width=ZOOM_CONFIG["width"], height=ZOOM_CONFIG["height"],
                       loc=ZOOM_CONFIG["loc"], bbox_to_anchor=ZOOM_CONFIG["bbox_to_anchor"],
                       bbox_transform=ax.transAxes)
    draw_curve_content(axins, LEFT_Y_TICKS)

    zoom_x_min_idx = np.interp(ZOOM_CONFIG["x_min"], CUSTOM_X_TICKS, np.arange(len(CUSTOM_X_TICKS)))
    zoom_x_max_idx = np.interp(ZOOM_CONFIG["x_max"], CUSTOM_X_TICKS, np.arange(len(CUSTOM_X_TICKS)))
    zoom_y_min_idx = transform_y(ZOOM_CONFIG["y_min"], LEFT_Y_TICKS)
    zoom_y_max_idx = transform_y(ZOOM_CONFIG["y_max"], LEFT_Y_TICKS)

    axins.set_xlim(zoom_x_min_idx, zoom_x_max_idx)
    axins.set_ylim(zoom_y_min_idx, zoom_y_max_idx)
    axins.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                      labelbottom=False, labelleft=False)
    axins.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)

    con1 = ConnectionPatch(xyA=(zoom_x_max_idx, zoom_y_max_idx), coordsA=ax.transData,
                           xyB=(0, 1), coordsB=axins.transAxes, axesA=ax, axesB=axins,
                           arrowstyle="-", linestyle="--", linewidth=1.0, color="k", alpha=0.8)
    con2 = ConnectionPatch(xyA=(zoom_x_max_idx, zoom_y_min_idx), coordsA=ax.transData,
                           xyB=(0, 0), coordsB=axins.transAxes, axesA=ax, axesB=axins,
                           arrowstyle="-", linestyle="--", linewidth=1.0, color="k", alpha=0.8)
    ax.add_artist(con1)
    ax.add_artist(con2)
    rect = Rectangle((zoom_x_min_idx, zoom_y_min_idx),
                     width=(zoom_x_max_idx - zoom_x_min_idx),
                     height=(zoom_y_max_idx - zoom_y_min_idx),
                     linewidth=1.0, edgecolor='k', facecolor='none', linestyle='--', alpha=0.8)
    ax.add_patch(rect)

# ================= 右图配置 (Breakdown) =================
TARGET_BANDWIDTHS = [50, 100, 200]

def plot_right_panel(ax):
    num_methods = len(method_names)
    indices = np.arange(len(TARGET_BANDWIDTHS))
    group_width = 0.9
    bar_width = group_width / num_methods

    for i, bw in enumerate(TARGET_BANDWIDTHS):
        group_center = indices[i]
        for j, method in enumerate(method_names):
            x = group_center - (group_width / 2) + (j * bar_width) + (bar_width / 2)
            
            if method == "Default":
                t_c, t_d = 0.0, 0.0
                size = ORIGINAL_DATA_SIZE_GB
            else:
                scenario = next(s for s in COMPRESSION_SCENARIOS if s["name"] == method)
                t_c = scenario["compression_time"]
                t_d = scenario["decompression_time"]
                size = scenario["compressed_size_gb"]
            
            t_t_ms = calculate_transmission_time(size, bw) * 1000
            t_c_ms = t_c * 1000
            t_d_ms = t_d * 1000
            
            h0, h1 = 0, t_c_ms
            h2 = t_c_ms + t_t_ms
            h3 = t_c_ms + t_t_ms + t_d_ms
            
            y0 = transform_y(h0, RIGHT_Y_TICKS)
            y1 = transform_y(h1, RIGHT_Y_TICKS)
            y2 = transform_y(h2, RIGHT_Y_TICKS)
            y3 = transform_y(h3, RIGHT_Y_TICKS)
            
            c_bar = bar_color_map[method]
            actual_bar_width = bar_width * 0.85
            
            ax.bar(x, y1 - y0, width=actual_bar_width, bottom=y0,
                   color=c_bar, edgecolor='black', hatch=hatch_comp, linewidth=1.5, zorder=3)
            ax.bar(x, y2 - y1, width=actual_bar_width, bottom=y1,
                   color=c_bar, edgecolor='black', hatch=hatch_trans, linewidth=1.5, zorder=3)
            ax.bar(x, y3 - y2, width=actual_bar_width, bottom=y2,
                   color=c_bar, edgecolor='black', hatch=hatch_decomp, linewidth=1.5, zorder=3)

            # 添加数字标签：保留整数，加粗
            ax.text(x, y3 + 0.05, f"{int(h3)}", ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', zorder=4)

    ax.set_yticks(np.arange(len(RIGHT_Y_TICKS)))
    ax.set_yticklabels([str(y) for y in RIGHT_Y_TICKS])
    ax.set_ylim(0, len(RIGHT_Y_TICKS) - 1)
    # ax.set_ylabel("Latency (ms)", fontweight='bold', fontsize=15)
    ax.set_xticks(indices)
    ax.set_xticklabels([f"{bw}" for bw in TARGET_BANDWIDTHS], fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ================= 核心修改函数：构建自定义图例 =================
def create_combined_legend_box(fig, handles_row1, labels_row1, handles_row2, labels_row2):
    """
    使用 offsetbox 构建两行居中对齐、每行元素数量不同的统一图例
    """
    
    def create_breakdown_item(handle, label):
        # 1. 创建图例图标区域 (DrawingArea)
        # width=22, height=12 大致对应标准图例图标的大小
        da = DrawingArea(width=22, height=10, xdescent=0, ydescent=0)
        
        # 提取 handle 的样式 (颜色、边框、纹理) 并画一个矩形
        rect = Rectangle((0, 0), width=22, height=10,
                         facecolor=handle.get_facecolor(),
                         edgecolor=handle.get_edgecolor(),
                         hatch=handle.get_hatch(),
                         linewidth=handle.get_linewidth())
        da.add_artist(rect)
        
        # 2. 创建文字区域 (TextArea)
        ta = TextArea(label, textprops=dict(color="black", size=13, family="DejaVu Sans", fontweight='bold'))
        
        # 3. 将图标和文字水平打包 (HPacker), 类似 "icon  label"
        return HPacker(children=[da, ta], align="center", pad=0, sep=5)

    def create_method_item(handle, label):
        # 1. 创建图例图标区域 (DrawingArea) - 宽度增加以容纳折线和柱状
        da = DrawingArea(width=45, height=10, xdescent=0, ydescent=0)

        # 使用独立的颜色映射：左侧折线用 curve_color_map，右侧柱形用 bar_color_map
        curve_color = curve_color_map[label]
        bar_color = bar_color_map[label]

        # 画折线 (Line) - 左图风格
        linestyle = '--' if label == "Default" else '-'
        line = Line2D([0, 18], [5, 5], color=curve_color, linewidth=2.5, linestyle=linestyle)
        da.add_artist(line)
        
        # 画柱状 (Rect) - 右图风格
        rect = Rectangle((22, 0), width=23, height=10,
                         facecolor=bar_color,
                         edgecolor='black',
                         linewidth=1.5)
        da.add_artist(rect)
        
        # 2. 创建文字区域 (TextArea)
        ta = TextArea(label, textprops=dict(color="black", size=13, family="DejaVu Sans", fontweight='bold'))
        
        # 3. 将图标和文字水平打包 (HPacker)
        return HPacker(children=[da, ta], align="center", pad=0, sep=5)

    # --- 构建第一组 (Methods) ---
    group1_items = [create_method_item(h, l) for h, l in zip(handles_row1, labels_row1)]
    group1_packer = HPacker(children=group1_items, align="center", pad=0, sep=10)
    
    # --- 构建第二组 (Breakdown) ---
    group2_items = [create_breakdown_item(h, l) for h, l in zip(handles_row2, labels_row2)]
    group2_packer = HPacker(children=group2_items, align="center", pad=0, sep=10)
    
    # --- 将两组水平打包 (HPacker) ---
    # sep=40 增加两组之间的间距
    hbox = HPacker(children=[group1_packer, group2_packer], align="center", pad=0, sep=10)
    
    # --- 放入带边框的容器 (AnchoredOffsetbox) ---
    anchored_box = AnchoredOffsetbox(
        loc='lower center',
        child=hbox,
        pad=0,          # 内部边距
        frameon=True,     # 开启边框
        bbox_to_anchor=(0.5, 0.93), # 居中
        bbox_transform=fig.transFigure,
        borderpad=0.
    )
    
    # 设置边框样式
    anchored_box.patch.set_boxstyle("round,pad=0.4")
    anchored_box.patch.set_linewidth(1.2)
    anchored_box.patch.set_edgecolor('black')
    anchored_box.patch.set_facecolor('white')
    
    return anchored_box

# ================= 主函数 =================
def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    
    plot_left_panel(axes[0])
    plot_right_panel(axes[1])
    
    # --- 准备图例数据 ---
    # 第一行：Methods (4个) — 这里只需要一个占位的 handle，具体颜色在 create_method_item 中根据 label 决定
    handles_row1 = [
        Patch(facecolor='white', edgecolor='black', linewidth=1.5) for m in method_names
    ]
    labels_row1 = method_names # ["Default", "Cachegen", ...]
    
    # 第二行：Breakdown (3个)
    handles_row2 = [
        Patch(facecolor='white', edgecolor='black', hatch=hatch_comp, linewidth=1.5),
        Patch(facecolor='white', edgecolor='black', hatch=hatch_trans, linewidth=1.5),
        Patch(facecolor='white', edgecolor='black', hatch=hatch_decomp, linewidth=1.5)
    ]
    labels_row2 = ['Comp.', 'Comm.', 'Decomp.']

    # --- 创建并添加合并后的自定义图例 ---
    custom_legend = create_combined_legend_box(fig, handles_row1, labels_row1, handles_row2, labels_row2)
    fig.add_artist(custom_legend)
    
    # 共用 X 轴标签
    fig.supxlabel("Network Bandwidth (Gbps)", fontweight='bold', fontsize=15, x=0.54)
               
    # 调整 layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.84, bottom=0.18, wspace=0.15) 

    output_filename = "bandwidth_latency_combined_legend.pdf"
    plt.savefig(output_filename)
    print(f"Plot generated: {output_filename}")

if __name__ == "__main__":
    main()