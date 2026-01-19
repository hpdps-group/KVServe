import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # 新增: 用于创建独立的 Colorbar
from matplotlib.ticker import MultipleLocator  # 用于设置刻度间隔
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf
import sys
import os

plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# ================= 配置区域 =================
JSON_FILE = 'merged_output.json'
OUTPUT_FILE = 'pareto_surface_plot_flexible.pdf'

# 定义轴对应的数据字段
KEY_X = 'cr'        # X轴: Compression Ratio
KEY_Y = 'accuracy'  # Y轴: Accuracy
KEY_Z = 'latency'   # Z轴: Latency

# 定义轴标签
LABEL_X = 'Compression Ratio'
LABEL_Y = 'Relative Accuracy (%)'
LABEL_Z = 'Latency (ms)'

# === 轴方向配置 ===
INVERT_X_AXIS = False 
INVERT_Y_AXIS = False 
INVERT_Z_AXIS = True 

# === 灵活裁剪与视野扩展配置 (6个变量) ===
# 定义在 Pareto 数据极值基础上，向各个方向扩展显示的比例
# 例如 0.05 表示向外延伸 5%

# X轴扩展 (CR)
EXTEND_X_MIN = 0.05   # 左侧扩展
EXTEND_X_MAX = 0.15   # 右侧扩展

# Y轴扩展 (Accuracy)
EXTEND_Y_MIN = 0.05   # 前侧扩展
EXTEND_Y_MAX = 0   # 后侧扩展

# Z轴扩展 (Latency)
EXTEND_Z_MIN = 0.25    # 底部扩展
EXTEND_Z_MAX = 0.05    # 顶部扩展 (之前设太大可能导致上方空旷，改回0.1)

# === 美化配置 ===
CMAP_NAME = 'viridis_r'
GRID_DENSITY = 2000
SURFACE_ALPHA = 0.65

# === 图片尺寸配置 ===
FIG_WIDTH = 12   # 图片宽度（英寸）
FIG_HEIGHT = 5   # 图片高度（英寸）
# 3D 坐标轴的宽高比 (x, y, z)，用于控制 3D 图的显示比例
# 调整这些值可以改变最终图片的宽高比效果
BOX_ASPECT = (1.0, 1.0, 0.5)  # (x, y, z) 比例，z 值越小，图片越扁平

# ================= 数据处理函数 =================

def load_data(filepath):
    """加载并清洗数据"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        points = []
        for item in data:
            if all(k in item for k in [KEY_X, KEY_Y, KEY_Z]):
                points.append([item[KEY_X], item[KEY_Y], item[KEY_Z]])
        return np.array(points)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def is_pareto_efficient(costs, return_mask=True):
    """寻找 Pareto 最优解"""
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def get_extended_range(values, ext_min_ratio, ext_max_ratio):
    """根据扩展比例计算视图范围"""
    v_min, v_max = min(values), max(values)
    v_range = v_max - v_min
    if v_range == 0: v_range = 1.0
    
    view_min = v_min - (v_range * ext_min_ratio)
    view_max = v_max + (v_range * ext_max_ratio)
    return view_min, view_max, v_range

def main():
    # 1. 准备数据
    data = load_data(JSON_FILE)
    if len(data) == 0: return

    x_all, y_all, z_all = data[:, 0], data[:, 1], data[:, 2]

    # 2. Pareto 筛选
    pareto_input = np.column_stack([-x_all, -y_all, z_all])
    pareto_mask = is_pareto_efficient(pareto_input)
    
    pareto_points = data[pareto_mask]
    x_pareto, y_pareto, z_pareto = pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2]

    print(f"Total points: {len(data)}, Pareto Frontier: {len(pareto_points)}")

    # 3. 计算视图范围 (View Limits) - 用户定义的最终相框大小
    x_view_min, x_view_max, x_range = get_extended_range(x_pareto, EXTEND_X_MIN, EXTEND_X_MAX)
    y_view_min, y_view_max, y_range = get_extended_range(y_pareto, EXTEND_Y_MIN, EXTEND_Y_MAX)
    z_view_min, z_view_max, z_range = get_extended_range(z_pareto, EXTEND_Z_MIN, EXTEND_Z_MAX)

    # 4. 生成平滑曲面数据
    XI, YI, ZI = None, None, None
    try:
        # A. 计算网格 (直接使用视图范围)
        # 我们生成稍微比视图大一点点的网格，防止边缘出现空白缝隙
        buffer = 0.01 
        xi = np.linspace(x_view_min, x_view_max, GRID_DENSITY)
        yi = np.linspace(y_view_min, y_view_max, GRID_DENSITY)
        XI, YI = np.meshgrid(xi, yi)
        
        # B. Rbf 插值
        rbf = Rbf(x_pareto, y_pareto, z_pareto, function='thin_plate', smooth=0)
        ZI = rbf(XI, YI)

        # C. 【关键修正】数据级硬裁剪
        # 显式地将超出 X, Y, Z 视图范围的数据点设为 NaN
        # 这样 Matplotlib 就绝对不会画出这些部分了
        
        # 裁剪 Z 轴 (上下)
        ZI[ZI < z_view_min] = np.nan
        ZI[ZI > z_view_max] = np.nan

        # 裁剪 X 轴 (虽然网格是基于X生成的，但Rbf可能会造成边界效应，这里双重保险)
        # 注意：这里其实不需要裁剪 X/Y，因为 XI/YI 本身就是按 view 范围生成的
        # 但为了保证万无一失，保留此逻辑结构
        
    except Exception as e:
        print(f"Interpolation warning: {e}")

    # 5. 绘图
    plt.style.use('default') 
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置 3D 坐标轴的宽高比，这会影响最终图片的显示比例
    # 注意：这个比例是相对于数据范围的，不是绝对的像素比例
    ax.set_box_aspect(BOX_ASPECT)

    # 绘制被支配点 (限制在视野内，避免画出太远的点干扰缩放)
    non_pareto_mask = ~pareto_mask
    mask_inside_view = (
        (x_all >= x_view_min) & (x_all <= x_view_max) &
        (y_all >= y_view_min) & (y_all <= y_view_max) &
        (z_all >= z_view_min) & (z_all <= z_view_max)
    )
    # 取交集：既是被支配点，又在视野内
    plot_bg_mask = non_pareto_mask & mask_inside_view
    
    ax.scatter(x_all[plot_bg_mask], y_all[plot_bg_mask], z_all[plot_bg_mask], 
               c='#000000', alpha=0.5, s=15, linewidth=0, marker='o', label='Design Space')

    if ZI is not None:
        # 绘制曲面
        surf = ax.plot_surface(XI, YI, ZI, cmap=CMAP_NAME, alpha=SURFACE_ALPHA,
                               rcount=100, ccount=100,
                               vmin=min(z_pareto), vmax=max(z_pareto),
                               edgecolor='none', antialiased=True, shade=True)
        # 线框
        ax.plot_wireframe(XI, YI, ZI, rstride=10, cstride=10, 
                          color='white', alpha=0.1, linewidth=0.5)

    # 绘制 Pareto 关键点
    p_scatter = ax.scatter(x_pareto, y_pareto, z_pareto, c=z_pareto, cmap=CMAP_NAME, 
                           s=50, edgecolors='black', linewidth=0.8, alpha=1.0, 
                           label='Pareto Optimal', zorder=10)

    # 6. 设置与美化
    ax.set_xlabel(LABEL_X, labelpad=10, fontsize=10, fontweight='bold')
    ax.set_ylabel(LABEL_Y, labelpad=10, fontsize=10, fontweight='bold')
    # ax.set_zlabel(LABEL_Z, labelpad=10, fontsize=10, fontweight='bold')
    # ax.set_title('Pareto Frontier Surface (Clipped View)', fontsize=14, pad=10, fontweight='bold')

    # 严格锁定视口范围
    ax.set_xlim(x_view_min, x_view_max)
    ax.set_ylim(y_view_min, y_view_max)
    ax.set_zlim(z_view_min, z_view_max)
    
    # 设置 z 轴刻度间隔为 10
    ax.zaxis.set_major_locator(MultipleLocator(10))

    # 背景微调
    ax.xaxis.pane.set_edgecolor('#f0f0f0')
    ax.yaxis.pane.set_edgecolor('#f0f0f0')
    ax.zaxis.pane.set_edgecolor('#f0f0f0')
    ax.xaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
    ax.yaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
    ax.zaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))

    # Colorbar
    norm = plt.Normalize(vmin=min(z_pareto), vmax=max(z_pareto))
    sm = cm.ScalarMappable(cmap=CMAP_NAME, norm=norm)
    sm.set_array([]) # 必须设置一个空数组
    
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.015, alpha=SURFACE_ALPHA)
    cbar.set_label(LABEL_Z, labelpad=18, fontsize=10, fontweight='bold', rotation=270)
    cbar.outline.set_visible(False)

    # 处理轴反转
    if INVERT_X_AXIS: ax.invert_xaxis()
    if INVERT_Y_AXIS: ax.invert_yaxis()
    if INVERT_Z_AXIS: ax.invert_zaxis(); cbar.ax.invert_yaxis()

    ax.view_init(elev=35, azim=45)
    plt.tight_layout()
    # 使用 bbox_inches='tight' 但设置 pad_inches 来控制边距
    # 或者使用固定边距来保持宽高比
    plt.savefig(OUTPUT_FILE, bbox_inches='tight', pad_inches=0.1)
    # 如果还是不够宽，可以尝试不使用 tight，改用固定边距：
    # plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches=None, pad_inches=0.2)
    print(f"Correctly clipped plot saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()