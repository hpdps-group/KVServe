import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle

# ================= 配置参数 =================
plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# 1. 数据与刻度配置
ORIGINAL_DATA_SIZE_GB = 1.25
CUSTOM_X_TICKS = [1, 50, 100, 150, 200, 250, 300, 350]
CUSTOM_Y_TICKS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
NUM_POINTS = 300 

# 2. 压缩方案数据
COMPRESSION_SCENARIOS = [
    {"name": "Cachegen", "compressed_size_gb": 0.14492, "time_overhead": 0.0098 + 0.0473 + 0.0074 + 0.0435},
    {"name": "KIVI", "compressed_size_gb": 0.23443, "time_overhead": 0.0402 + 0.0144},    
    {"name": "KVServe", "compressed_size_gb": 0.21654, "time_overhead": 0.0032 + 0.0164 + 0.0019 + 0.0079},
]

# 3. 放大镜配置
ZOOM_CONFIG = {
    "x_min": 5,     # 横轴起始 (实际带宽值 Gbps)
    "x_max": 17,    # 横轴结束
    "y_min": 0.18,  # 纵轴起始 (实际延迟值 seconds)
    "y_max": 0.35,  # 纵轴结束
    "loc": "center", # 小图位置
    "width": "50%",  # 小图宽度 
    "height": "50%", # 小图高度
    "bbox_to_anchor": (0.05, 0.2, 1, 1) # 相对定位微调
}

# 图表标签
X_LABEL = "Bandwidth (Gbps)"
Y_LABEL = "Latency (seconds)"
TITLE = "Latency-Bandwidth Analysis"

# ========================================================

def calculate_transmission_time(data_size_gb, bandwidth_gbps):
    bandwidth_gbps = np.maximum(bandwidth_gbps, 1e-9) 
    return (data_size_gb * 8.0) / bandwidth_gbps

def map_values_to_linear_ticks(values, ticks):
    ticks = np.array(ticks)
    # np.interp 如果遇到 nan 会返回 nan，这正是我们需要的
    return np.interp(values, ticks, np.arange(len(ticks)))

def plot_bandwidth_latency():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # --- 1. 数据准备 ---
    plot_x_indices = np.linspace(0, len(CUSTOM_X_TICKS) - 1, NUM_POINTS)
    real_bandwidths = np.interp(plot_x_indices, np.arange(len(CUSTOM_X_TICKS)), CUSTOM_X_TICKS)

    def transform_y(y_values):
        return map_values_to_linear_ticks(y_values, CUSTOM_Y_TICKS)

    # --- 2. 绘图函数 (核心修改在这里) ---
    def draw_content(target_ax, show_legend=False):
        # 获取允许显示的 Y 轴最大值 (即 0.4)
        y_max_limit = CUSTOM_Y_TICKS[-1]

        # --- A. 绘制 Original ---
        original_latency = calculate_transmission_time(ORIGINAL_DATA_SIZE_GB, real_bandwidths)
        
        # [关键修改]：创建一个副本，将超过最大限制的值设为 NaN
        original_plot_data = original_latency.copy()
        original_plot_data[original_plot_data > y_max_limit] = np.nan
        
        target_ax.plot(plot_x_indices, transform_y(original_plot_data), 
                       label="Original", linewidth=2, linestyle='--', color='grey')
        
        # --- B. 绘制压缩方案 ---
        colors = plt.cm.tab10(np.linspace(0, 1, len(COMPRESSION_SCENARIOS)))
        for i, scenario in enumerate(COMPRESSION_SCENARIOS):
            total_latency = scenario["time_overhead"] + calculate_transmission_time(scenario["compressed_size_gb"], real_bandwidths)
            
            # [关键修改]：同样对这些线条做截断处理
            plot_data = total_latency.copy()
            plot_data[plot_data > y_max_limit] = np.nan
            
            target_ax.plot(plot_x_indices, transform_y(plot_data), 
                           label=scenario["name"], linewidth=2, color=colors[i])
        
        if show_legend:
            target_ax.legend(prop={'weight': 'bold'}, fontsize=11, loc='upper right')

    # --- 3. 绘制主图 ---
    draw_content(ax, show_legend=True)

    ax.set_xticks(np.arange(len(CUSTOM_X_TICKS)))
    ax.set_xticklabels([str(x) for x in CUSTOM_X_TICKS])
    ax.set_xlim(0, len(CUSTOM_X_TICKS) - 1)

    ax.set_yticks(np.arange(len(CUSTOM_Y_TICKS)))
    ax.set_yticklabels([str(y) for y in CUSTOM_Y_TICKS])
    ax.set_ylim(0, len(CUSTOM_Y_TICKS) - 1)

    ax.set_xlabel(X_LABEL, fontweight='bold', fontsize=12)
    ax.set_ylabel(Y_LABEL, fontweight='bold', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # ================= 放大镜逻辑 (方案2：手动绘制) =================

    # 1. 创建小图
    axins = inset_axes(ax, 
                       width=ZOOM_CONFIG["width"], 
                       height=ZOOM_CONFIG["height"], 
                       loc=ZOOM_CONFIG["loc"],
                       bbox_to_anchor=ZOOM_CONFIG["bbox_to_anchor"],
                       bbox_transform=ax.transAxes)

    # 2. 在小图上绘制
    # 注意：这里的 draw_content 也会自动应用上面的 NaN 截断逻辑，
    # 但如果小图范围本身就很大，这也没关系；如果小图只看局部，NaN 不会影响局部。
    draw_content(axins, show_legend=False)

    # 3. 计算坐标转换
    zoom_x_min_idx = np.interp(ZOOM_CONFIG["x_min"], CUSTOM_X_TICKS, np.arange(len(CUSTOM_X_TICKS)))
    zoom_x_max_idx = np.interp(ZOOM_CONFIG["x_max"], CUSTOM_X_TICKS, np.arange(len(CUSTOM_X_TICKS)))
    
    zoom_y_min_idx = transform_y(ZOOM_CONFIG["y_min"])
    zoom_y_max_idx = transform_y(ZOOM_CONFIG["y_max"])

    # 4. 设置小图范围
    axins.set_xlim(zoom_x_min_idx, zoom_x_max_idx)
    axins.set_ylim(zoom_y_min_idx, zoom_y_max_idx)

    # 5. 样式调整
    axins.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                      labelbottom=False, labelleft=False)
    axins.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)

    # 线1: 主图框的【右上角】 -> 小图的【左上角】
    con1 = ConnectionPatch(xyA=(zoom_x_max_idx, zoom_y_max_idx), coordsA=ax.transData,
                           xyB=(0, 1), coordsB=axins.transAxes,
                           axesA=ax, axesB=axins,
                           arrowstyle="-", linestyle="--", linewidth=1.0, color="k", alpha=0.8)
    ax.add_artist(con1)

    # 线2: 主图框的【右下角】 -> 小图的【左下角】
    con2 = ConnectionPatch(xyA=(zoom_x_max_idx, zoom_y_min_idx), coordsA=ax.transData,
                           xyB=(0, 0), coordsB=axins.transAxes,
                           axesA=ax, axesB=axins,
                           arrowstyle="-", linestyle="--", linewidth=1.0, color="k", alpha=0.8)
    ax.add_artist(con2)

    # 方框
    rect = Rectangle((zoom_x_min_idx, zoom_y_min_idx), 
                     width=(zoom_x_max_idx - zoom_x_min_idx), 
                     height=(zoom_y_max_idx - zoom_y_min_idx),
                     linewidth=1.0, edgecolor='k', facecolor='none', linestyle='--', alpha=0.8)
    ax.add_patch(rect)

    # ==========================================================

    plt.savefig("zoomed_plot_manual_clipped.png")
    print("Plot generated: zoomed_plot_manual_clipped.png")
    # plt.show()

if __name__ == "__main__":
    plot_bandwidth_latency()