import numpy as np
import matplotlib.pyplot as plt

# ================= 配置参数 =================
plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
})

# 1. 数据配置
ORIGINAL_DATA_SIZE_GB = 1.25
TARGET_BANDWIDTHS = [50, 100, 200]  # <--- 修改为列表，横轴将遍历这些带宽

# 2. 压缩方案数据
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
        "name": "KVServe",
        "compressed_size_gb": 0.21654,
        "compression_time": 0.0032 + 0.0164,
        "decompression_time": 0.0019 + 0.0079,
    },
]

def calculate_transmission_time(data_size_gb, bandwidth_gbps):
    """计算传输时间 (seconds)"""
    bandwidth_gbps = max(bandwidth_gbps, 1e-9) 
    return (data_size_gb * 8.0) / bandwidth_gbps

def plot_grouped_latency_breakdown():
    # --- 1. 准备元数据 ---
    method_names = ["Original"] + [s["name"] for s in COMPRESSION_SCENARIOS]
    num_methods = len(method_names)
    num_bandwidths = len(TARGET_BANDWIDTHS)
    
    # --- 2. 绘图设置 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 布局参数
    group_width = 0.85  # 每个带宽组占用的总宽度比例 (0-1)
    bar_width = group_width / num_methods
    indices = np.arange(num_bandwidths)
    
    # 颜色与纹理样式
    # 颜色池：#e41a1c, #377eb8, #4daf4a, #984ea3, #ff7f00
    method_colors = ['#ff6f61', '#ffb74d', '#4fc3f7', '#4db6ac']
    color_map = {name: col for name, col in zip(method_names, method_colors)}
    
    # 纹理样式
    hatch_comp = '//'
    hatch_trans = '..'
    hatch_decomp = '\\\\'

    # --- 3. 循环绘制 ---
    # 外层循环：带宽（X轴主刻度）
    for i, bw in enumerate(TARGET_BANDWIDTHS):
        group_center = indices[i]
        
        # 内层循环：方法（组内柱子）
        for j, method in enumerate(method_names):
            # 计算当前柱子的中心 x 坐标
            x = group_center - (group_width / 2) + (j * bar_width) + (bar_width / 2)
            
            # 获取数据
            if method == "Original":
                t_c, t_d = 0.0, 0.0
                size = ORIGINAL_DATA_SIZE_GB
            else:
                scenario = next(s for s in COMPRESSION_SCENARIOS if s["name"] == method)
                t_c = scenario["compression_time"]
                t_d = scenario["decompression_time"]
                size = scenario["compressed_size_gb"]
            
            t_t = calculate_transmission_time(size, bw)
            
            # 转换为毫秒 (ms)
            t_c_ms = t_c * 1000
            t_t_ms = t_t * 1000
            t_d_ms = t_d * 1000
            
            # 获取当前方法的颜色
            c = color_map[method]
            
            # 绘制堆叠柱 (使用 bar_width * 0.9 让柱子之间有一点间隙)
            actual_bar_width = bar_width * 0.9
            
            # 1. Compression (底部)
            # 使用实心颜色 + 纹理
            ax.bar(x, t_c_ms, width=actual_bar_width, bottom=0, 
                   color=c, edgecolor='black', hatch=hatch_comp, linewidth=0.8, zorder=3)
            
            # 2. Transmission (中间)
            # 使用实心颜色 + 无纹理
            ax.bar(x, t_t_ms, width=actual_bar_width, bottom=t_c_ms, 
                   color=c, edgecolor='black', hatch=hatch_trans, linewidth=0.8, zorder=3)
            
            # 3. Decompression (顶部)
            # 使用实心颜色 + 纹理
            ax.bar(x, t_d_ms, width=actual_bar_width, bottom=t_c_ms + t_t_ms, 
                   color=c, edgecolor='black', hatch=hatch_decomp, linewidth=0.8, zorder=3)
            
            # 在柱子下方添加方法名 (旋转90度以节省空间) - 稍微调整位置
            # ax.text(x, -0.05, method, ha='center', va='top', 
            #         rotation=90, fontsize=10, transform=ax.get_xaxis_transform())

    # --- 4. 坐标轴与样式调整 ---
    ax.set_ylabel("Latency (ms)", fontweight='bold', fontsize=13)
    
    # 设置 X 轴主刻度为带宽
    ax.set_xticks(indices)
    ax.set_xticklabels([f"{bw} Gbps" for bw in TARGET_BANDWIDTHS], fontsize=12, fontweight='bold')
    
    # 将带宽标签向下移动
    # ax.tick_params(axis='x', pad=20) 
    
    # 添加辅助说明
    ax.set_xlabel("Network Bandwidth", fontweight='bold', fontsize=13, labelpad=10)

    # 网格线
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    # --- 图例设置 (两行) ---
    from matplotlib.patches import Patch
    
    # 1. Breakdown 图例 (上方)
    # 使用白色背景 + 黑色边框 + 对应纹理 来展示样式
    legend_breakdown = [
        Patch(facecolor='white', edgecolor='black', hatch=hatch_comp, label='Compression'),
        Patch(facecolor='white', edgecolor='black', hatch=hatch_trans, label='Transmission'),
        Patch(facecolor='white', edgecolor='black', hatch=hatch_decomp, label='Decompression')
    ]
    
    # 2. Methods 图例 (下方)
    legend_methods = [
        Patch(facecolor=color_map[m], edgecolor='black', label=m) for m in method_names
    ]
    
    # 添加 Methods 图例 (先添加下面的)
    leg_methods = ax.legend(handles=legend_methods, loc='lower center', 
                           bbox_to_anchor=(0.5, 1.02), ncol=4, 
                           frameon=False, fontsize=11, handletextpad=0.5, columnspacing=1.5)
    ax.add_artist(leg_methods)
    
    # 添加 Breakdown 图例 (再添加上面的)
    leg_breakdown = ax.legend(handles=legend_breakdown, loc='lower center', 
                             bbox_to_anchor=(0.5, 1.10), ncol=3, 
                             frameon=False, fontsize=11, handletextpad=0.5, columnspacing=1.5)

    # 去掉上方和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 自动调整 Y 轴范围
    max_latency_ms = 0
    for bw in TARGET_BANDWIDTHS:
        t = calculate_transmission_time(ORIGINAL_DATA_SIZE_GB, bw) * 1000
        if t > max_latency_ms: max_latency_ms = t
    
    # 给顶部留出更多空间给两行图例
    ax.set_ylim(0, max_latency_ms * 1.35)
    
    # 纵轴取整
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # --- 5. 保存 ---
    output_filename = "latency_breakdown_grouped.png"
    plt.savefig(output_filename, bbox_inches='tight') # bbox_inches='tight' 很重要，防止标签被切掉
    print(f"Plot generated: {output_filename}")

if __name__ == "__main__":
    plot_grouped_latency_breakdown()