import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, DrawingArea
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

LOG_FILE_NAME = 'search_process.log'
INPUT_LOG_PATH = os.path.join(os.path.dirname(__file__), LOG_FILE_NAME)
OUTPUT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), f'{LOG_FILE_NAME.replace(".log", "")}_process.pdf')

# 自定义 Y 轴刻度列表 (左图)
# 刻度之间将以均匀间隔显示
CUSTOM_LEFT_Y_TICKS = [5, 6, 7, 8, 9, 12, 15, 20]

# 正则匹配模式
PATTERN_ITERATION_REMAINING = re.compile(r"--- BO Iteration (\d+)/\d+ \| Remaining: (\d+)/(\d+) ---")
PATTERN_PROPOSING = re.compile(r"Proposing Config \(ID:\d+\): CR=([\d\.]+)")
PATTERN_INFEASIBLE = re.compile(r"❌ INFEASIBLE")
PATTERN_FEASIBLE = re.compile(r"✅ Configuration is FEASIBLE")

# ================= USER CONFIGURATION =================
# 1. Colors for Legend Items
# Line 1: Process Trace
COLOR_PROCESS_TRACE = '#868686'  # Blue

# Line 2: Best CR Found
COLOR_BEST_CR = '#2CA02C'        # Green

# Point 1: Feasible
COLOR_FEASIBLE_POINT = '#2CA02C' # Green
MARKER_FEASIBLE = 'o'

# Point 2: Infeasible
COLOR_INFEASIBLE_POINT = '#e41a1c' # Red
MARKER_INFEASIBLE = 'x'

# Line Style
LINE_STYLE = '-'

FIGURE_SIZE = (10, 4)
FONT_SIZE = 16
TICK_FONT_SIZE = 14  # 刻度字号，与FONT_SIZE统一，可在文件开头修改
LINE_WIDTH = 3

def parse_log(log_path):
    """
    解析 Log 文件，返回两个列表：
    1. iterations_data: [{'iter': 1, 'cr': 20.9, 'is_feasible': False}, ...]
    2. remaining_data: [{'iter': 1, 'remaining': 3966}, ...]
    3. best_cr_final
    """
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return [], [], None

    iterations_data = []
    remaining_data = []
    
    current_iter = None
    current_remaining = None
    current_cr = None
    
    best_cr_final = 0.0

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 1. 匹配 Iteration 和 Remaining
            match_iter = PATTERN_ITERATION_REMAINING.search(line)
            if match_iter:
                current_iter = int(match_iter.group(1))
                current_remaining = int(match_iter.group(2))
                remaining_data.append({'iter': current_iter, 'remaining': current_remaining})
                continue

            # 2. 匹配 Proposing CR
            match_prop = PATTERN_PROPOSING.search(line)
            if match_prop:
                current_cr = float(match_prop.group(1))
                continue

            # 3. 匹配 Feasible / Infeasible
            if current_iter is not None and current_cr is not None:
                if PATTERN_FEASIBLE.search(line):
                    iterations_data.append({
                        'iter': current_iter,
                        'cr': current_cr,
                        'is_feasible': True
                    })
                    best_cr_final = max(best_cr_final, current_cr)
                    current_cr = None 
                elif PATTERN_INFEASIBLE.search(line):
                    iterations_data.append({
                        'iter': current_iter,
                        'cr': current_cr,
                        'is_feasible': False
                    })
                    current_cr = None

    return iterations_data, remaining_data, best_cr_final

def transform_to_equidistant(values, ticks):
    """
    将原始值映射到基于 ticks 的等间距坐标上。
    ticks 中的每个间隔将被映射为 1 的距离。
    例如 ticks=[5, 10, 20]
    5 -> 0
    10 -> 1
    20 -> 2
    7.5 (5和10中间) -> 0.5
    """
    if not ticks:
        return values
    
    transformed = []
    sorted_ticks = sorted(ticks)
    
    for v in values:
        # 找到 v 所在的区间
        if v <= sorted_ticks[0]:
            # 低于最小值，线性外推 (假设斜率与第一个区间相同)
            # slope = 1 / (sorted_ticks[1] - sorted_ticks[0])
            # mapped = 0 - (sorted_ticks[0] - v) * slope
            # 简单处理：直接截断或映射到0
            transformed.append(0) 
        elif v >= sorted_ticks[-1]:
            # 高于最大值
            transformed.append(len(sorted_ticks) - 1)
        else:
            # 在区间内
            for i in range(len(sorted_ticks) - 1):
                if sorted_ticks[i] <= v <= sorted_ticks[i+1]:
                    # 线性插值
                    ratio = (v - sorted_ticks[i]) / (sorted_ticks[i+1] - sorted_ticks[i])
                    transformed.append(i + ratio)
                    break
    return transformed

def create_custom_legend(fig):
    def create_item(label, draw_func, width=22):
        da = DrawingArea(width=width, height=10, xdescent=0, ydescent=0)
        draw_func(da)
        ta = TextArea(label, textprops=dict(color="black", size=12, family="DejaVu Sans", fontweight='bold'))
        return HPacker(children=[da, ta], align="center", pad=0, sep=5)

    # 1. Process Trace
    def draw_trace(da):
        line = Line2D([0, 22], [5, 5], color=COLOR_PROCESS_TRACE, linewidth=LINE_WIDTH, alpha=0.9)
        da.add_artist(line)
    
    # 2. Best CR Found
    def draw_best(da):
        line = Line2D([0, 22], [5, 5], color=COLOR_BEST_CR, linewidth=LINE_WIDTH, linestyle='--', alpha=0.9)
        da.add_artist(line)

    # 3. Feasible
    def draw_feasible(da):
        mk = Line2D([11], [5], marker=MARKER_FEASIBLE, color='w', 
                   markerfacecolor=COLOR_FEASIBLE_POINT, markersize=8)
        da.add_artist(mk)

    # 4. Infeasible
    def draw_infeasible(da):
        mk = Line2D(
            [11], [5], 
            marker=MARKER_INFEASIBLE, 
            color='w', 
            markerfacecolor=COLOR_INFEASIBLE_POINT, 
            markeredgecolor=COLOR_INFEASIBLE_POINT, 
            markersize=6,
            markeredgewidth=2.2  # 加粗边框
        )
        da.add_artist(mk)
        
    # 5. Phase 1
    def draw_p1(da):
        txt = Text(11, 5, "①", ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')
        da.add_artist(txt)
        
    # 6. Phase 2
    def draw_p2(da):
        txt = Text(11, 5, "②", ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')
        da.add_artist(txt)

    items = [
        create_item("Process Trace", draw_trace),
        create_item("Best CR Found", draw_best),
        create_item("Feasible", draw_feasible),
        create_item("Infeasible", draw_infeasible),
        create_item("Exploration", draw_p1),
        create_item("Exploitation", draw_p2),
    ]
    
    # Pack all horizontally
    hbox = HPacker(children=items, align="center", pad=0, sep=10)

    anchored_box = AnchoredOffsetbox(
        loc='lower center',
        child=hbox,
        pad=0,
        frameon=True,
        bbox_to_anchor=(0.5, 0.87),
        bbox_transform=fig.transFigure,
        borderpad=0.
    )
    
    anchored_box.patch.set_boxstyle("round,pad=0.4")
    anchored_box.patch.set_linewidth(1.2)
    anchored_box.patch.set_edgecolor('black')
    anchored_box.patch.set_facecolor('white')
    
    return anchored_box

def main():
    print(f"Reading log from: {INPUT_LOG_PATH}")
    iter_data, rem_data, best_cr = parse_log(INPUT_LOG_PATH)
    
    if not iter_data:
        print("No iteration data found.")
        return

    # 构建 Remaining Map 以便查找
    rem_map = {d['iter']: d['remaining'] for d in rem_data}

    # 准备绘图数据 (左图: CR)
    iters = [d['iter'] for d in iter_data]
    raw_crs = [d['cr'] for d in iter_data]
    
    # 应用坐标变换
    plot_crs = transform_to_equidistant(raw_crs, CUSTOM_LEFT_Y_TICKS)
    
    # 拆分 Feasible/Infeasible 并变换
    feasible_iters = [d['iter'] for d in iter_data if d['is_feasible']]
    raw_feasible_crs = [d['cr'] for d in iter_data if d['is_feasible']]
    plot_feasible_crs = transform_to_equidistant(raw_feasible_crs, CUSTOM_LEFT_Y_TICKS)
    
    infeasible_iters = [d['iter'] for d in iter_data if not d['is_feasible']]
    raw_infeasible_crs = [d['cr'] for d in iter_data if not d['is_feasible']]
    plot_infeasible_crs = transform_to_equidistant(raw_infeasible_crs, CUSTOM_LEFT_Y_TICKS)
    
    # 最佳 CR 变换
    plot_best_cr = transform_to_equidistant([best_cr], CUSTOM_LEFT_Y_TICKS)[0] if best_cr > 0 else -1

    # 准备绘图数据 (右图: Remaining)
    rem_iters = [d['iter'] for d in rem_data]
    rem_counts = [d['remaining'] for d in rem_data]
    
    feasible_rem_iters = []
    feasible_rem_counts = []
    infeasible_rem_iters = []
    infeasible_rem_counts = []
    
    for d in iter_data:
        it = d['iter']
        if it in rem_map:
            cnt = rem_map[it]
            if d['is_feasible']:
                feasible_rem_iters.append(it)
                feasible_rem_counts.append(cnt)
            else:
                infeasible_rem_iters.append(it)
                infeasible_rem_counts.append(cnt)

    # 开始绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    # --- 左图: Iteration vs CR (Transformed Y) ---
    ax1.plot(iters, plot_crs, color=COLOR_PROCESS_TRACE, linestyle=LINE_STYLE, alpha=0.9, zorder=1, linewidth=LINE_WIDTH)
    
    if feasible_iters:
        ax1.scatter(feasible_iters, plot_feasible_crs, color=COLOR_FEASIBLE_POINT, marker=MARKER_FEASIBLE, zorder=2)
    if infeasible_iters:
        ax1.scatter(infeasible_iters, plot_infeasible_crs, color=COLOR_INFEASIBLE_POINT, marker=MARKER_INFEASIBLE, zorder=2)
    
    if best_cr > 0:
        ax1.axhline(y=plot_best_cr, color=COLOR_BEST_CR, linestyle='--', alpha=0.9, zorder=1, linewidth=LINE_WIDTH)

    # 设置自定义 Y 轴刻度
    if CUSTOM_LEFT_Y_TICKS:
        ax1.set_yticks(range(len(CUSTOM_LEFT_Y_TICKS)))
        ax1.set_yticklabels(CUSTOM_LEFT_Y_TICKS, fontsize=TICK_FONT_SIZE)
        ax1.set_ylim(-0.5, len(CUSTOM_LEFT_Y_TICKS) - 0.5)
    
    # 设置x轴刻度字号
    ax1.tick_params(axis='x', labelsize=TICK_FONT_SIZE)

    ax1.set_ylabel('Compression Ratio (CR)', fontweight='bold', fontsize=FONT_SIZE)
    ax1.set_title('Prediction Process', fontweight='bold', fontsize=FONT_SIZE)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 右图: Iteration vs Remaining Space ---
    ax2.plot(rem_iters, rem_counts, color=COLOR_PROCESS_TRACE, linestyle=LINE_STYLE, alpha=0.9, zorder=1, linewidth=LINE_WIDTH)
    
    # 绘制 Feasible Points (Remaining)
    if feasible_rem_iters:
        ax2.scatter(feasible_rem_iters, feasible_rem_counts, color=COLOR_FEASIBLE_POINT, marker=MARKER_FEASIBLE, zorder=2)
    
    # 绘制 Infeasible Points (Remaining)
    if infeasible_rem_iters:
        ax2.scatter(infeasible_rem_iters, infeasible_rem_counts, color=COLOR_INFEASIBLE_POINT, marker=MARKER_INFEASIBLE, zorder=2)
    
    # 设置y轴使用科学计数法
    power = None
    if rem_counts:
        # 计算10的幂（使用最大值的数量级）
        max_val = max(rem_counts)
        if max_val > 0:
            # 计算数量级（10的幂）
            power = int(np.floor(np.log10(max_val)))
            # 自定义格式化器：只显示系数（除以10^power）
            def format_func(x, p):
                coeff = x / (10 ** power)
                if coeff == 0:
                    return '0'
                return f'{coeff:.0f}k'
            
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    
    # 将y轴刻度和标签移到右侧
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    
    # 设置刻度字号
    ax2.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
    ax2.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    
    ax2.set_ylabel('Remaining Search Space', fontweight='bold', fontsize=FONT_SIZE, rotation=270, labelpad=18)
    ax2.set_title('Pruning Process', fontweight='bold', fontsize=FONT_SIZE)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # --- 添加阶段分隔线和标记 (1-24: 探索, 25+: 剪枝) ---
    max_x = max(iters) if iters else 0
    split_x = 24.5
    min_x = 0.5  # Start boundary
    
    if max_x > split_x:
        # 统一标注的垂直位置 (Axes 坐标, 0-1)
        y_pos_arrow = 0.87
        
        for ax in (ax1, ax2):
            # 绘制竖虚线 (Phase 1 Start, Split, Phase 2 End)
            ax.axvline(x=min_x, color='#666666', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.axvline(x=split_x, color='#666666', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.axvline(x=max_x + 0.5, color='#666666', linestyle='--', linewidth=1.5, alpha=0.8)

            # Shading for Phase 1 and Phase 2
            # ax.axvspan(min_x, split_x, color='#868686', alpha=0.1, zorder=0)
            # ax.axvspan(split_x, max_x + 0.5, color='#e41a1c', alpha=0.1, zorder=0)
            
            # --- 阶段 1 标记 ---
            mid_1 = (min_x + split_x) / 2
            
            # 1. 绘制文字 (zorder=10 确保在上层, alpha=1.0 不透明背景遮挡箭头)
            ax.text(mid_1, y_pos_arrow, '①', transform=ax.get_xaxis_transform(),
                    ha='center', va='center', fontsize=20, fontweight='bold', color='#333333',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=4), zorder=10)
            
            # 2. 绘制双向箭头 (与文字同一高度)
            ax.annotate('', xy=(min_x, y_pos_arrow), xytext=(split_x, y_pos_arrow),
                        xycoords=ax.get_xaxis_transform(), textcoords=ax.get_xaxis_transform(),
                        arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5), zorder=9)

            # --- 阶段 2 标记 ---
            end_x = max_x + 0.5
            mid_2 = (split_x + end_x) / 2
            
            # 1. 绘制文字
            ax.text(mid_2, y_pos_arrow, '②', transform=ax.get_xaxis_transform(),
                    ha='center', va='center', fontsize=20, fontweight='bold', color='#333333',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=4), zorder=10)
            
            # 2. 绘制双向箭头
            ax.annotate('', xy=(split_x, y_pos_arrow), xytext=(end_x, y_pos_arrow),
                        xycoords=ax.get_xaxis_transform(), textcoords=ax.get_xaxis_transform(),
                        arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5), zorder=9)

    # --- 添加统一的横轴标签 ---
    fig.supxlabel('Evaluation Iteration', fontweight='bold', fontsize=FONT_SIZE, y=-0.02)

    # --- Legend ---
    # Create custom legend using offsetbox
    custom_legend = create_custom_legend(fig)
    fig.add_artist(custom_legend)

    # Adjust layout
    plt.tight_layout()
    # Reserve space at top for legend, reduce bottom space to bring label closer
    plt.subplots_adjust(top=0.78, bottom=0.12)
    
    # # 在右上角（图框外侧）添加10的幂标注（必须在tight_layout之后，以便获取正确的bbox）
    # if power is not None:
    #     # 使用figure坐标，放在axes右侧外侧
    #     # 获取ax2在figure中的位置
    #     bbox = ax2.get_position()
    #     # 在axes右侧外侧添加文本（x坐标在axes右侧，y坐标在axes顶部）
    #     fig.text(bbox.x1 - 0.02, bbox.y1 + 0.013, f'$\\times 10^{{{power}}}$', 
    #             fontsize=TICK_FONT_SIZE, 
    #             fontweight='bold',
    #             verticalalignment='bottom',
    #             horizontalalignment='left',
    #             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8),
    #             zorder=10)

    # Save
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"Plot saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    main()
