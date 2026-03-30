import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea, DrawingArea
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

# ================= 用户自定义区域 =================

# 1. 线条样式配置
# 对应: Bandwidth (上图), Latency1, Latency2, Latency3 (下图)
LINE_STYLES = {
    "Bandwidth": {"color": "#868686", "marker": "", "linestyle": "-", "linewidth": 3, "label": "Bandwidth"},
    "Latency3":  {"color": "#FFBB78", "marker": "", "linestyle": "--", "linewidth": 2, "label": "Method 3"},
    "Latency2":  {"color": "#C5B0D5", "marker": "", "linestyle": "--", "linewidth": 2, "label": "Method 2"},
    "Latency1":  {"color": "#e41a1c", "marker": "", "linestyle": "-", "linewidth": 3, "label": "Method 1"},    
}

# 2. 纵轴刻度配置 (必须是单调递增的列表，绘图时会将其映射为均匀间隔)
# 上图 Bandwidth 刻度
Y_TICKS_TOP = [0, 20, 40, 60]
# 下图 Latency 刻度
Y_TICKS_BOTTOM = [0, 0.3, 0.6, 0.9, 1.2]

# 3. 文件路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATHS = {
    "Bandwidth": os.path.join(BASE_DIR, "bandwidth.csv"),
    "Latency3":  os.path.join(BASE_DIR, "latency3.csv"),
    "Latency2":  os.path.join(BASE_DIR, "latency2.csv"),
    "Latency1":  os.path.join(BASE_DIR, "latency1.csv"),
}

# ================= 数据处理函数 =================

def load_csv(file_path):
    """读取 CSV 文件，返回 Time 和 Value 的数组"""
    times = []
    values = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return np.array([]), np.array([])
        
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # 自动识别列名: Time, Bandwidth/Latency
        headers = reader.fieldnames
        key_time = 'Time'
        key_value = headers[1] if len(headers) > 1 else None
        
        for row in reader:
            try:
                t = float(row[key_time])
                v = float(row[key_value])
                times.append(t)
                values.append(v)
            except ValueError:
                continue
                
    return np.array(times), np.array(values)

def map_values_to_linear_ticks(values, ticks):
    """将真实值映射到 ticks 定义的均匀索引空间"""
    ticks = np.array(ticks)
    # 使用插值将 value 映射到 [0, 1, 2, ... len(ticks)-1]
    return np.interp(values, ticks, np.arange(len(ticks)))

# ================= 图例绘制函数 (参考 provided code) =================

def create_legend_box(fig, handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.95), frameon=True, orientation='horizontal'):
    """
    使用 offsetbox 构建单行居中图例
    """
    def create_legend_item(handle, label):
        da = DrawingArea(width=22, height=10, xdescent=0, ydescent=0)
        
        if isinstance(handle, Line2D):
            line = Line2D([0, 11, 22], [5, 5, 5],
                          color=handle.get_color(),
                          linewidth=handle.get_linewidth(),
                          linestyle=handle.get_linestyle(),
                          marker=handle.get_marker(),
                          markersize=handle.get_markersize(),
                          markeredgecolor=handle.get_markeredgecolor(),
                          markeredgewidth=handle.get_markeredgewidth(),
                          markevery=None) # 图例中显示marker通常更好看，或者设为None看style
            # 为了让图例更清晰，这里手动设置marker在中间
            line.set_markevery([1]) 
            da.add_artist(line)
        
        ta = TextArea(label, textprops=dict(color="black", size=8, family="DejaVu Sans", fontweight='bold'))
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
        anchored_box.patch.set_boxstyle("round,pad=0.2") # 减小 pad 让边框更贴近内容
        anchored_box.patch.set_linewidth(1.2) # 轻微边框可以适当减小，不过1.2也还行，保持一致吧
        anchored_box.patch.set_edgecolor('black')
        anchored_box.patch.set_facecolor('white')
        anchored_box.patch.set_alpha(0.1) # 背景透明度
    
    return anchored_box

# ================= 主绘图逻辑 =================

def plot_ax(ax, x_data, y_data, style_config, y_ticks, y_label):
    """绘制单个子图"""
    # 1. 转换 Y 数据到均匀刻度空间
    y_transformed = map_values_to_linear_ticks(y_data, y_ticks)
    
    # 2. 绘制折线
    # 注意: X轴这里保持原始 Time 值 (线性坐标)
    line, = ax.plot(x_data, y_transformed,
            color=style_config["color"],
            marker=style_config["marker"],
            linestyle=style_config["linestyle"],
            linewidth=style_config["linewidth"],
            markersize=8,
            markeredgecolor='white',
            markeredgewidth=1.0,
            label=style_config["label"])
            
    # 3. 设置 Y 轴刻度显示
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_yticklabels([str(y) for y in y_ticks])
    ax.set_ylim(-0.1, len(y_ticks) - 0.9)
    
    # 4. 设置 Label 和 Grid
    if y_label == "Bandwidth":
        ax.set_ylabel(y_label, fontweight='bold', fontsize=14, labelpad=6)
    else:
        ax.set_ylabel(y_label, fontweight='bold', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 去除上右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return line

def main():
    # 创建画布: 2行1列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
    
    # ================= 绘制上图 (Bandwidth) =================
    name = "Bandwidth"
    t, v = load_csv(FILE_PATHS[name])
    if len(t) > 0:
        l1 = plot_ax(ax1, t, v, LINE_STYLES[name], Y_TICKS_TOP, "Bandwidth")
        handles_top = [l1]
        labels_top = [LINE_STYLES[name]["label"]]
    else:
        handles_top = []
        labels_top = []

    # ================= 绘制下图 (Latency) =================
    handles_bottom = []
    labels_bottom = []
    
    # 依次读取并绘制三组 Latency
    for key in ["Latency3", "Latency2", "Latency1"]:
        t, v = load_csv(FILE_PATHS[key])
        if len(t) > 0:
            # 注意：在同一个 ax2 上绘制
            # 为了让 plot_ax 支持叠加，我们需要稍微调整一下逻辑，或者直接在这里调用
            # 由于 plot_ax 会重设 ticks，我们在循环外设置 ticks 比较好，或者确保 ticks 一致
            # 这里简单起见，我们把 transform 逻辑拿出来
            
            y_trans = map_values_to_linear_ticks(v, Y_TICKS_BOTTOM)
            line, = ax2.plot(t, y_trans,
                     color=LINE_STYLES[key]["color"],
                     marker=LINE_STYLES[key]["marker"],
                     linestyle=LINE_STYLES[key]["linestyle"],
                     linewidth=LINE_STYLES[key]["linewidth"],
                     markersize=8,
                     markeredgecolor='white',
                     markeredgewidth=1.0,
                     label=LINE_STYLES[key]["label"])
            handles_bottom.append(line)
            labels_bottom.append(LINE_STYLES[key]["label"])

    # 设置下图 Y 轴刻度 (因为是叠加绘制，只需要设置一次)
    ax2.set_yticks(np.arange(len(Y_TICKS_BOTTOM)))
    ax2.set_yticklabels([str(y) for y in Y_TICKS_BOTTOM])
    ax2.set_ylim(-0.1, len(Y_TICKS_BOTTOM) - 0.9)
    ax2.set_ylabel("Latency", fontweight='bold', fontsize=14)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ================= 设置 X 轴 =================
    ax2.set_xlabel("Time (s)", fontweight='bold', fontsize=16)
    
    # ================= 添加图例 =================
    # 分开图例：Bandwidth 在上图，Latency 方法在下图
    
    if handles_top:
        legend_top = create_legend_box(fig, handles_top, labels_top, 
                                     bbox_to_anchor=(0.3, 0.8), frameon=True) 
        fig.add_artist(legend_top)

    if handles_bottom:
        legend_bottom = create_legend_box(fig, handles_bottom, labels_bottom, 
                                     bbox_to_anchor=(0.3, 0.4), frameon=True, orientation='vertical') 
        fig.add_artist(legend_bottom)

    # ================= 添加阴影区域 (Fluctuation) =================
    fluc_start, fluc_end = 20, 40
    shadow_color = '#e41a1c'
    shadow_alpha = 0.1
    
    # 上图阴影
    ax1.axvspan(fluc_start, fluc_end, color=shadow_color, alpha=shadow_alpha, zorder=0)
    ax1.axvline(x=fluc_start, color='#868686', linestyle='--', linewidth=1.2)
    ax1.axvline(x=fluc_end, color='#868686', linestyle='--', linewidth=1.2)
    # 添加文字 "Fluctuation" (居中)
    # ax1 Y轴范围是 indices, 取中间位置 (len(Y_TICKS_TOP)-1)/2
    y_mid_idx = 2.5
    ax1.text((fluc_start + fluc_end) / 2, y_mid_idx, "Fluctuation", 
             ha='center', va='center', fontsize=12, fontweight='bold', color=shadow_color, zorder=10)
             
    # 下图阴影
    ax2.axvspan(fluc_start, fluc_end, color=shadow_color, alpha=shadow_alpha, zorder=0)
    ax2.axvline(x=fluc_start, color='#868686', linestyle='--', linewidth=1.2)
    ax2.axvline(x=fluc_end, color='#868686', linestyle='--', linewidth=1.2)

    # 调整布局
    plt.tight_layout()
    # 预留空间给顶部图例 (top=0.85 左右)
    plt.subplots_adjust(top=0.85, hspace=0.2)
    
    output_file = os.path.join(BASE_DIR, "bandwidth_latency_chart.pdf")
    plt.savefig(output_file)
    print(f"Plot generated: {output_file}")

if __name__ == "__main__":
    main()

