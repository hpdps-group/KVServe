import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle

# ================= GLOBAL CONFIGURATION =================
OUTPUT_PDF = "acc_time_cr_combined.pdf"
FIGURE_SIZE = (10, 3.5) # Wider figure for two subplots
GRID_ALPHA = 0.3
LINE_WIDTH = 2

# Matplotlib rcParams (Shared Style)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "axes.linewidth": 1.5,
    "lines.linewidth": LINE_WIDTH,
    "grid.linestyle": "--",
    "grid.alpha": GRID_ALPHA,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# ================= LEFT PLOT CONFIG (Acc & Time) =================
LEFT_DATA_FILE = "acc-time.csv"
LEFT_X_LABEL = "Dataset Percentage (%)"
LEFT_Y_LABEL_LEFT = "Accuracy (%)"
LEFT_Y_LABEL_RIGHT = "Time (min)"
LEFT_BAR_COLOR = '#c7c7c7'
LEFT_LINE_COLOR = '#e41a1c'
LEFT_SHADE_COLOR = '#e41a1c'
LEFT_BAR_WIDTH = 0.65

# Custom Ticks for Left Plot
CUSTOM_Y_TICKS_LEFT = [0, 70, 75, 80, 82, 84, 86, 88, 90]
CUSTOM_Y_TICKS_RIGHT = [0, 10, 20, 30, 40, 50, 60]

# ================= RIGHT PLOT CONFIG (CR) =================
RIGHT_TARGET_FILES = [
    "13.json",
    "15.json",
    "27.json",
    "43.json",
]
RIGHT_X_LABEL = "Request ID"
RIGHT_Y_LABEL = "Compression Ratio"
RIGHT_LEGEND_LABELS = {
    "13.json": "MixHQ 1",
    "15.json": "MixHQ 2",
    "27.json": "MixHQ 3",
    "43.json": "MixHQ 4",
}
RIGHT_COLORS = ['#1F77B4', '#9467BD', '#FF7F0E', '#2CA02C']

# Zoom Config for Right Plot
ENABLE_ZOOM = True
ZOOM_CONFIG = {
    "x_min": 0,      "x_max": 6,
    "y_min": 8.3,    "y_max": 8.5,
    "loc": "upper center",
    "width": "40%",
    "height": "30%",
    "bbox_to_anchor": (-0.1, 0, 1, 1)
}

# ================= HELPERS =================

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def load_csv_data(file_name):
    file_path = os.path.join(get_script_dir(), file_name)
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None, None, None
    x_vals, time_vals, acc_vals = [], [], []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            try:
                x_vals.append(float(row[0]))
                time_vals.append(float(row[1]))
                acc_vals.append(float(row[2]))
            except ValueError: continue
    return np.array(x_vals), np.array(time_vals), np.array(acc_vals)

def load_json_data(file_name):
    file_path = os.path.join(get_script_dir(), file_name)
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping.")
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

def map_values_to_linear_ticks(values, ticks):
    ticks = np.array(ticks)
    return np.interp(values, ticks, np.arange(len(ticks)))

# ================= PLOTTING FUNCTIONS =================

def plot_left_acc_time(ax1):
    x_real, time_sec, acc_real = load_csv_data(LEFT_DATA_FILE)
    if x_real is None: return

    time_min_real = time_sec / 60.0
    ax2 = ax1.twinx() # Create second y-axis

    # Mappings
    x_indices = np.arange(len(x_real))
    acc_mapped = map_values_to_linear_ticks(acc_real, CUSTOM_Y_TICKS_LEFT)
    time_mapped = map_values_to_linear_ticks(time_min_real, CUSTOM_Y_TICKS_RIGHT)

    # --- Bars (Right Axis) ---
    bars = ax2.bar(x_indices, time_mapped, width=LEFT_BAR_WIDTH, color=LEFT_BAR_COLOR, alpha=1, label="Time", zorder=1, edgecolor='black', linewidth=1.5, hatch='..')
    for bar, real_val in zip(bars, time_min_real):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{int(real_val)}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    # --- Line (Left Axis) ---
    line, = ax1.plot(x_indices, acc_mapped, color=LEFT_LINE_COLOR, marker='o', markersize=6, label="Accuracy", zorder=10)
    
    # Shade
    last_acc = acc_real[-1]
    upper_mapped = map_values_to_linear_ticks(last_acc + 1.0, CUSTOM_Y_TICKS_LEFT)
    lower_mapped = map_values_to_linear_ticks(last_acc - 1.0, CUSTOM_Y_TICKS_LEFT)
    
    ax1.axhline(y=upper_mapped, color=LEFT_SHADE_COLOR, linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=lower_mapped, color=LEFT_SHADE_COLOR, linestyle='--', linewidth=1, alpha=0.5)
    ax1.fill_between([min(x_indices)-1, max(x_indices)+1], lower_mapped, upper_mapped, color=LEFT_SHADE_COLOR, alpha=0.1, zorder=0)

    # --- Annotation for Best Trade-off (30%) ---
    target_percent = 30
    target_idx = np.where(x_real == target_percent)[0]
    
    if len(target_idx) > 0:
        idx = target_idx[0]
        x_pos = x_indices[idx]
        y_pos = acc_mapped[idx]
        
        # Using the style referenced from KVServe/evaluation/test.py
        ax1.annotate(
            'Best Trade-off', 
            xy=(x_pos + 0.1, y_pos + 0.1), 
            xytext=(x_pos + 1.5, y_pos + 2), # Offset text position
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='#cc0000', lw=2),
            fontsize=10, color='#cc0000', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cc0000", alpha=0.9),
            zorder=20
        )

    # Styles
    ax1.set_xlabel(LEFT_X_LABEL, fontweight='bold', fontsize=15)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f"{int(val)}" if val.is_integer() else str(val) for val in x_real], fontsize=12)
    ax1.set_xlim(min(x_indices) - 0.6, max(x_indices) + 0.6)

    ax1.set_ylabel(LEFT_Y_LABEL_LEFT, fontweight='bold', fontsize=15)
    ax1.set_yticks(np.arange(len(CUSTOM_Y_TICKS_LEFT)))
    ax1.set_yticklabels([str(y) for y in CUSTOM_Y_TICKS_LEFT], fontsize=12)
    ax1.set_ylim(0, len(CUSTOM_Y_TICKS_LEFT) - 1)
    ax1.spines['top'].set_visible(False)

    ax2.set_ylabel(LEFT_Y_LABEL_RIGHT, fontweight='bold', fontsize=15, rotation=270, labelpad=18)
    ax2.set_yticks(np.arange(len(CUSTOM_Y_TICKS_RIGHT)))
    ax2.set_yticklabels([str(y) for y in CUSTOM_Y_TICKS_RIGHT], fontsize=12)
    ax2.set_ylim(0, len(CUSTOM_Y_TICKS_RIGHT) - 1)
    ax2.spines['top'].set_visible(False)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax1.grid(True, axis='y', linestyle='--', alpha=GRID_ALPHA)

    # Add Legend (Combined from both axes)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Order: Line (Acc) then Bar (Time) as requested, or just combined
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', ncol=2, frameon=False, prop={'weight': 'bold', 'size': 10}, columnspacing=0.7)

def plot_right_cr(ax):
    datasets = []
    for idx, file_name in enumerate(RIGHT_TARGET_FILES):
        data = load_json_data(file_name)
        if data is None: continue
        datasets.append((
            RIGHT_LEGEND_LABELS.get(file_name, file_name.replace(".json", "")),
            np.arange(len(data)),
            np.array(data),
            RIGHT_COLORS[idx % len(RIGHT_COLORS)]
        ))
    
    if not datasets: return

    def draw_lines_helper(target_ax, legend=False):
        for label, x, y, color in datasets:
            target_ax.plot(x, y, label=label, color=color, alpha=0.9, linewidth=LINE_WIDTH)
        if legend:
            target_ax.legend(frameon=False, prop={'weight': 'bold', 'size': 10})

    # Main lines
    draw_lines_helper(ax, legend=True)
    
    ax.set_xlabel(RIGHT_X_LABEL, fontweight='bold', fontsize=15)
    ax.set_ylabel(RIGHT_Y_LABEL, fontweight='bold', fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Zoom Inset
    if ENABLE_ZOOM:
        axins = inset_axes(ax, width=ZOOM_CONFIG["width"], height=ZOOM_CONFIG["height"],
                           loc=ZOOM_CONFIG["loc"], bbox_to_anchor=ZOOM_CONFIG["bbox_to_anchor"],
                           bbox_transform=ax.transAxes)
        
        draw_lines_helper(axins, legend=False)
        axins.set_xlim(ZOOM_CONFIG["x_min"], ZOOM_CONFIG["x_max"])
        axins.set_ylim(ZOOM_CONFIG["y_min"], ZOOM_CONFIG["y_max"])
        
        # Hide ticks
        axins.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                          labelbottom=False, labelleft=False)
        axins.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)

        # Connectors
        rect = Rectangle((ZOOM_CONFIG["x_min"], ZOOM_CONFIG["y_min"]), 
                         width=(ZOOM_CONFIG["x_max"] - ZOOM_CONFIG["x_min"]), 
                         height=(ZOOM_CONFIG["y_max"] - ZOOM_CONFIG["y_min"]),
                         linewidth=1.0, edgecolor='k', facecolor='none', linestyle='--', alpha=0.8)
        ax.add_patch(rect)
        
        con1 = ConnectionPatch(xyA=(ZOOM_CONFIG["x_max"], ZOOM_CONFIG["y_max"]), coordsA=ax.transData,
                               xyB=(0, 1), coordsB=axins.transAxes,
                               axesA=ax, axesB=axins, arrowstyle="-", linestyle="--", linewidth=1.0, color="k", alpha=0.8)
        con2 = ConnectionPatch(xyA=(ZOOM_CONFIG["x_max"], ZOOM_CONFIG["y_min"]), coordsA=ax.transData,
                               xyB=(0, 0), coordsB=axins.transAxes,
                               axesA=ax, axesB=axins, arrowstyle="-", linestyle="--", linewidth=1.0, color="k", alpha=0.8)
        ax.add_artist(con1)
        ax.add_artist(con2)

def main():
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    # Plot Left
    plot_left_acc_time(ax_left)
    
    # Plot Right
    plot_right_cr(ax_right)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    pdf_path = os.path.join(get_script_dir(), OUTPUT_PDF)
    plt.savefig(pdf_path)
    print(f"Combined plot saved to: {pdf_path}")

if __name__ == "__main__":
    main()
