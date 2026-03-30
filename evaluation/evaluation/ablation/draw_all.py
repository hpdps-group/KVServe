import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea, DrawingArea
import matplotlib.gridspec as gridspec
import os

# ================= GLOBAL CONFIGURATION =================
OUTPUT_PDF = "ablation_all_combined.pdf"
FIGURE_SIZE = (10, 4)

# Common Matplotlib rcParams
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.5,
    "grid.linestyle": "--",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# ================= ABLATION CONFIGURATION =================
ABLATION_STYLES = [
    # 1. w/o Exp
    {"bar_color": "#c7c7c7", "bar_edge": "black", "hatch": "..", "marker": "o", "markersize": 8, "alpha": 0.8},
    # 2. w/o Enc
    {"bar_color": "#c7c7c7", "bar_edge": "black", "hatch": "..", "marker": "o", "markersize": 8, "alpha": 0.8},
    # 3. w/o Prune
    {"bar_color": "#c7c7c7", "bar_edge": "black", "hatch": "..", "marker": "o", "markersize": 8, "alpha": 0.8},
    # 4. w/o Stop
    {"bar_color": "#c7c7c7", "bar_edge": "black", "hatch": "..", "marker": "o", "markersize": 8, "alpha": 0.8},
    # 5. Full (Highlights)
    {"bar_color": "#e41a1c", "bar_edge": "black", "hatch": "xx", "marker": "*", "markersize": 14, "alpha": 1.0},
]

ABLATION_LINE_COLOR = '#e41a1c'
ABLATION_GRID_ALPHA = 0.3
ABLATION_LINE_WIDTH = 2
ABLATION_CUSTOM_Y_TICKS_LEFT = [0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4]
ABLATION_CUSTOM_Y_TICKS_RIGHT = [150, 400]
ABLATION_CUSTOM_X_LABELS = ["w/o\nExp", "w/o\nEnc", "w/o\nPrune", "w/o\nStop", "KVServe"]

# ================= BANDWIDTH/LATENCY CONFIGURATION =================
BW_LINE_STYLES = {
    "Bandwidth": {"color": "#868686", "marker": "", "linestyle": "-", "linewidth": 3, "label": "Bandwidth"},
    "Latency3":  {"color": "#FFBB78", "marker": "", "linestyle": "--", "linewidth": 2, "label": "w/o Controller"},
    "Latency2":  {"color": "#C5B0D5", "marker": "", "linestyle": "--", "linewidth": 2, "label": "w/o Bandit"},
    "Latency1":  {"color": "#e41a1c", "marker": "", "linestyle": "-", "linewidth": 3, "label": "KVServe"},    
}
BW_Y_TICKS_TOP = [0, 20, 40, 60]
BW_Y_TICKS_BOTTOM = [0, 0.3, 0.6, 0.9]

# ================= HELPERS =================

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def map_values_to_linear_ticks(values, ticks):
    ticks = np.array(ticks)
    return np.interp(values, ticks, np.arange(len(ticks)))

# --- Ablation Helpers ---

def load_data_ablation(file_name):
    # Adjust path to look into offline_profile
    file_path = os.path.join(get_script_dir(), "offline_profile", file_name)
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None, None, None
    
    methods = []
    max_crs = []
    iters = []
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            methods.append(row['method'])
            max_crs.append(float(row['max_cr']))
            iters.append(int(row['iters']))
            
    return methods, np.array(max_crs), np.array(iters)

def create_legend_box_ablation(fig, handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.95), frameon=True, orientation='vertical', transform=None):
    """
    Modified create_legend_box for Ablation part (using width=15 etc.)
    """
    def create_legend_item(handle, label):
        da = DrawingArea(width=15, height=10, xdescent=0, ydescent=0)
        
        if isinstance(handle, Line2D):
            line = Line2D([0, 11, 15], [5, 5, 5],
                          color=handle.get_color(),
                          linewidth=handle.get_linewidth(),
                          linestyle=handle.get_linestyle(),
                          marker=handle.get_marker(),
                          markersize=handle.get_markersize(),
                          markeredgecolor=handle.get_markeredgecolor(),
                          markeredgewidth=handle.get_markeredgewidth(),
                          markevery=None)
            line.set_markevery([1]) 
            da.add_artist(line)
        elif isinstance(handle, (Patch, Rectangle)):
            fc = handle.get_facecolor()
            ec = handle.get_edgecolor()
            lw = handle.get_linewidth()
            alpha = handle.get_alpha()
            hatch = handle.get_hatch()
            
            r = Rectangle((0, 2), 15, 6, 
                          facecolor=fc,
                          edgecolor=ec,
                          hatch=hatch,
                          linewidth=lw if lw else 0,
                          alpha=alpha)
            da.add_artist(r)
        
        ta = TextArea(label, textprops=dict(color="black", size=10, family="DejaVu Sans", fontweight='bold'))
        return HPacker(children=[da, ta], align="center", pad=0, sep=5)

    items = [create_legend_item(h, l) for h, l in zip(handles, labels)]
    
    if orientation == 'vertical':
        packer = VPacker(children=items, align="left", pad=0, sep=1)
    else:
        packer = HPacker(children=items, align="center", pad=0, sep=10)
    
    anchored_box = AnchoredOffsetbox(
        loc=loc,
        child=packer,
        pad=0,
        frameon=frameon,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=transform if transform else fig.transFigure,
        borderpad=0
    )
    
    if frameon:
        anchored_box.patch.set_boxstyle("round,pad=0.2")
        anchored_box.patch.set_linewidth(1.2)
        anchored_box.patch.set_edgecolor('black')
        anchored_box.patch.set_facecolor('white')
        anchored_box.patch.set_alpha(0.1)
    
    return anchored_box

# --- Bandwidth Helpers ---

def load_csv_bw(file_name):
    # Adjust path to look into online_selection
    file_path = os.path.join(get_script_dir(), "online_selection", file_name)
    times = []
    values = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return np.array([]), np.array([])
        
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
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

def create_legend_box_bw(fig, handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.95), frameon=True, orientation='horizontal', transform=None):
    """
    create_legend_box from draw_bandwidth_latency.py
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
                          markevery=None)
            line.set_markevery([1]) 
            da.add_artist(line)
        
        ta = TextArea(label, textprops=dict(color="black", size=8, family="DejaVu Sans", fontweight='bold'))
        return HPacker(children=[da, ta], align="center", pad=0, sep=5)

    items = [create_legend_item(h, l) for h, l in zip(handles, labels)]
    
    if orientation == 'vertical':
        packer = VPacker(children=items, align="left", pad=0, sep=1)
    else:
        packer = HPacker(children=items, align="center", pad=0, sep=10)
    
    anchored_box = AnchoredOffsetbox(
        loc=loc,
        child=packer,
        pad=0,
        frameon=frameon,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=transform if transform else fig.transFigure,
        borderpad=0
    )
    
    if frameon:
        anchored_box.patch.set_boxstyle("round,pad=0.2")
        anchored_box.patch.set_linewidth(1.2)
        anchored_box.patch.set_edgecolor('black')
        anchored_box.patch.set_facecolor('white')
        anchored_box.patch.set_alpha(0.1)
    
    return anchored_box

def plot_bw_single_ax(ax, x_data, y_data, style_config, y_ticks, y_label):
    y_transformed = map_values_to_linear_ticks(y_data, y_ticks)
    line, = ax.plot(x_data, y_transformed,
            color=style_config["color"],
            marker=style_config["marker"],
            linestyle=style_config["linestyle"],
            linewidth=style_config["linewidth"],
            markersize=8,
            markeredgecolor='white',
            markeredgewidth=1.0,
            label=style_config["label"])
            
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_yticklabels([str(y) for y in y_ticks])
    ax.set_ylim(-0.1, len(y_ticks) - 0.9)
    
    if y_label == "Bandwidth":
        ax.set_ylabel("Bandwidth", fontweight='bold', fontsize=15, labelpad=6)
    else:
        ax.set_ylabel(y_label, fontweight='bold', fontsize=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return line

# ================= PLOTTING FUNCTIONS =================

def plot_ablation(fig, ax1):
    methods, max_crs, iters = load_data_ablation("data.csv")
    if methods is None:
        return

    max_crs_mapped = map_values_to_linear_ticks(max_crs, ABLATION_CUSTOM_Y_TICKS_LEFT)
    iters_mapped = map_values_to_linear_ticks(iters, ABLATION_CUSTOM_Y_TICKS_RIGHT)

    x_indices = np.arange(len(methods))
    bar_width = 0.6

    # --- Right Axis (Iterations) ---
    ax2 = ax1.twinx()
    
    bars = []
    for i in range(len(methods)):
        style = ABLATION_STYLES[i] if i < len(ABLATION_STYLES) else ABLATION_STYLES[-1]
        
        bar = ax2.bar(x_indices[i], iters_mapped[i], width=bar_width, 
                      color=style['bar_color'], 
                      edgecolor=style['bar_edge'],
                      hatch=style['hatch'],
                      alpha=style['alpha'],
                      zorder=1)
        bars.append(bar[0])

    for bar, val in zip(bars, iters):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

    # --- Left Axis (Max CR) ---
    ax1.plot(x_indices, max_crs_mapped, color=ABLATION_LINE_COLOR, linestyle='-', linewidth=2, zorder=10)
    
    for i in range(len(methods)):
        style = ABLATION_STYLES[i] if i < len(ABLATION_STYLES) else ABLATION_STYLES[-1]
        ax1.plot(x_indices[i], max_crs_mapped[i], 
                 color=ABLATION_LINE_COLOR, 
                 marker=style['marker'], 
                 markersize=style['markersize'], 
                 markeredgecolor='white',
                 markeredgewidth=0.5,
                 zorder=11)
    
    for x, y_mapped, y_real in zip(x_indices, max_crs_mapped, max_crs):
        ax1.text(x - 0.1, y_mapped + 0.3, f'{y_real:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=ABLATION_LINE_COLOR)

    # --- Axis Config ---
    # ax1.set_xlabel("Ablation Settings", fontweight='bold', fontsize=13)
    ax1.set_xticks(x_indices)
    
    if ABLATION_CUSTOM_X_LABELS and len(ABLATION_CUSTOM_X_LABELS) == len(methods):
        ax1.set_xticklabels(ABLATION_CUSTOM_X_LABELS, rotation=0, fontweight='bold', fontsize=13)
    else:
        ax1.set_xticklabels(methods, rotation=0)
        
    ax1.set_xlim(min(x_indices) - 0.6, max(x_indices) + 0.6)

    ax1.set_ylabel("Compression Ratio", fontweight='bold', fontsize=15)
    ax1.set_yticks(np.arange(len(ABLATION_CUSTOM_Y_TICKS_LEFT)))
    ax1.set_yticklabels([f"{y:.1f}" for y in ABLATION_CUSTOM_Y_TICKS_LEFT], fontsize=11)
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, len(ABLATION_CUSTOM_Y_TICKS_LEFT) - 1)
    
    ax2.set_ylabel("Iterations", fontweight='bold', fontsize=15, rotation=270, labelpad=-3)
    ax2.set_yticks(np.arange(len(ABLATION_CUSTOM_Y_TICKS_RIGHT)))
    ax2.set_yticklabels([str(y) for y in ABLATION_CUSTOM_Y_TICKS_RIGHT], fontsize=11)
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, len(ABLATION_CUSTOM_Y_TICKS_RIGHT) - 1)

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    ax1.grid(True, axis='y', linestyle='--', alpha=ABLATION_GRID_ALPHA)
    
    # --- Legend ---
    legend_handles = []
    legend_labels = []
    
    proxy_line = Line2D([0], [0], color=ABLATION_LINE_COLOR, markersize=8, linestyle='-')
    legend_handles.append(proxy_line)
    legend_labels.append("Max CR")

    style_wo = ABLATION_STYLES[0]
    proxy_rect_wo = Rectangle((0, 0), 1, 1, 
                           facecolor=style_wo['bar_color'], 
                           edgecolor=style_wo['bar_edge'],
                           hatch=style_wo['hatch'],
                           alpha=style_wo['alpha'])
    legend_handles.append(proxy_rect_wo)
    legend_labels.append("Iters (w/o)")

    style_full = ABLATION_STYLES[-1]
    proxy_rect_full = Rectangle((0, 0), 1, 1, 
                           facecolor=style_full['bar_color'], 
                           edgecolor=style_full['bar_edge'],
                           hatch=style_full['hatch'],
                           alpha=style_full['alpha'])
    legend_handles.append(proxy_rect_full)
    legend_labels.append("Iters (KVServe)")
    
    # Use ax1.transAxes for positioning
    legend_box = create_legend_box_ablation(fig, legend_handles, legend_labels, 
                                   loc='upper center', bbox_to_anchor=(0.22, 0.98), frameon=True, transform=ax1.transAxes)
    ax1.add_artist(legend_box)

def plot_bandwidth_latency(fig, ax1, ax2):
    # Bandwidth
    name = "Bandwidth"
    t, v = load_csv_bw("bandwidth.csv")
    if len(t) > 0:
        l1 = plot_bw_single_ax(ax1, t, v, BW_LINE_STYLES[name], BW_Y_TICKS_TOP, "Bandwidth")
        handles_top = [l1]
        labels_top = [BW_LINE_STYLES[name]["label"]]
        
        legend_top = create_legend_box_bw(fig, handles_top, labels_top, 
                                     loc='upper left', bbox_to_anchor=(0.02, 0.95), frameon=True, transform=ax1.transAxes) 
        ax1.add_artist(legend_top)

    # Latency
    handles_bottom = []
    labels_bottom = []
    for key in ["Latency3", "Latency2", "Latency1"]:
        t, v = load_csv_bw(f"{key.lower()}.csv")
        if len(t) > 0:
            y_trans = map_values_to_linear_ticks(v, BW_Y_TICKS_BOTTOM)
            line, = ax2.plot(t, y_trans,
                     color=BW_LINE_STYLES[key]["color"],
                     marker=BW_LINE_STYLES[key]["marker"],
                     linestyle=BW_LINE_STYLES[key]["linestyle"],
                     linewidth=BW_LINE_STYLES[key]["linewidth"],
                     markersize=8,
                     markeredgecolor='white',
                     markeredgewidth=1.0,
                     label=BW_LINE_STYLES[key]["label"])
            handles_bottom.append(line)
            labels_bottom.append(BW_LINE_STYLES[key]["label"])

    ax2.set_yticks(np.arange(len(BW_Y_TICKS_BOTTOM)))
    ax2.set_yticklabels([str(y) for y in BW_Y_TICKS_BOTTOM])
    ax2.set_ylim(-0.1, len(BW_Y_TICKS_BOTTOM) - 0.9)
    ax2.set_ylabel("Latency (s)", fontweight='bold', fontsize=15)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel("Time (s)", fontweight='bold', fontsize=15)

    if handles_bottom:
        legend_bottom = create_legend_box_bw(fig, handles_bottom, labels_bottom, 
                                     loc='upper left', bbox_to_anchor=(0.02, 0.95), frameon=True, orientation='vertical', transform=ax2.transAxes) 
        ax2.add_artist(legend_bottom)

    # Fluctuation Shadows
    fluc_start, fluc_end = 20, 40
    shadow_color = '#e41a1c'
    shadow_alpha = 0.1
    
    for ax in [ax1, ax2]:
        ax.axvspan(fluc_start, fluc_end, color=shadow_color, alpha=shadow_alpha, zorder=0)
        ax.axvline(x=fluc_start, color='#868686', linestyle='--', linewidth=1.2)
        ax.axvline(x=fluc_end, color='#868686', linestyle='--', linewidth=1.2)
    
    y_mid_idx = 2.7
    ax1.text((fluc_start + fluc_end) / 2, y_mid_idx, "Fluctuation", 
             ha='center', va='center', fontsize=12, fontweight='bold', color=shadow_color, zorder=10)

def main():
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1]) # Left:Right = 1:1

    # Left: Ablation (spans both rows)
    ax_ablation = fig.add_subplot(gs[:, 0])
    
    # Right: Bandwidth (Top) and Latency (Bottom)
    ax_bw = fig.add_subplot(gs[0, 1])
    ax_lat = fig.add_subplot(gs[1, 1], sharex=ax_bw)

    # Plot
    plot_ablation(fig, ax_ablation)
    plot_bandwidth_latency(fig, ax_bw, ax_lat)

    plt.tight_layout()
    # Adjust layout to prevent overlap
    plt.subplots_adjust(wspace=0.3, top=0.9, bottom=0.12)
    
    output_path = os.path.join(get_script_dir(), OUTPUT_PDF)
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()

