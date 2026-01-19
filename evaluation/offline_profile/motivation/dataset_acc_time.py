import csv
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= CONFIGURATION =================
DATA_FILE = "acc-time.csv"

# Labels
X_LABEL = "Dataset Percentage (%)"
Y_LABEL_LEFT = "Accuracy (%)"
Y_LABEL_RIGHT = "Time (min)"

# Custom Ticks Configuration
# Define the exact ticks you want to see on the axes.
# The code will map data values linearly relative to these ticks.
CUSTOM_Y_TICKS_LEFT = [0, 70, 75, 80, 82, 84, 86, 88, 90]
CUSTOM_Y_TICKS_RIGHT = [0, 10, 20, 30, 40, 50, 60]

# Output Settings
OUTPUT_PDF = "dataset_acc_time.pdf"

# Plot Aesthetics
FIGURE_SIZE = (8, 5)
LINE_WIDTH = 2
BAR_COLOR = '#1f77b4'      # Blue for bars
LINE_COLOR = '#d62728'     # Red for line
SHADE_COLOR = '#d62728'    # Red shade
GRID_ALPHA = 0.3
BAR_WIDTH = 0.5           # Adjusted for index-based x-axis (0, 1, 2...)

# Matplotlib rcParams (Matching request_cr.py style)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "lines.linewidth": LINE_WIDTH,
    "grid.linestyle": "--",
    "grid.alpha": GRID_ALPHA,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})
# =================================================

def load_data(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None, None, None

    x_vals = []
    time_vals = [] # Seconds
    acc_vals = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            try:
                x_vals.append(float(row[0]))
                time_vals.append(float(row[1]))
                acc_vals.append(float(row[2]))
            except ValueError:
                continue
                
    return np.array(x_vals), np.array(time_vals), np.array(acc_vals)

def map_values_to_linear_ticks(values, ticks):
    """
    Maps real data values to a linear index space [0, len(ticks)-1]
    based on the provided tick intervals.
    Example: If ticks are [0, 10, 100], value 5 maps to 0.5, value 55 maps to 1.5.
    """
    ticks = np.array(ticks)
    # np.interp expects increasing x. 
    return np.interp(values, ticks, np.arange(len(ticks)))

def main():
    x_real, time_sec, acc_real = load_data(DATA_FILE)
    if x_real is None:
        return

    # Convert time to minutes
    time_min_real = time_sec / 60.0

    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE)
    ax2 = ax1.twinx()

    # --- X Axis Handling: Equal Spacing ---
    # We use indices 0, 1, 2... for plotting to force equal spacing
    x_indices = np.arange(len(x_real))
    
    # --- Y Axis Mapping (Left - Accuracy) ---
    # Map real accuracy values to the custom tick indices
    acc_mapped = map_values_to_linear_ticks(acc_real, CUSTOM_Y_TICKS_LEFT)
    
    # --- Y Axis Mapping (Right - Time) ---
    # Map real time values to the custom tick indices
    time_mapped = map_values_to_linear_ticks(time_min_real, CUSTOM_Y_TICKS_RIGHT)

    # ================= PLOTTING =================

    # --- Right Axis: Bar Chart (Time) ---
    bars = ax2.bar(x_indices, time_mapped, width=BAR_WIDTH, color=BAR_COLOR, alpha=0.5, label="Time", zorder=1)
    
    # Add numbers on top of bars (Real values)
    for bar, real_val in zip(bars, time_min_real):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05, # small offset in mapped space
                 f'{int(real_val)}',
                 ha='center', va='bottom', fontsize=9, color='black')

    # --- Left Axis: Line Chart (Accuracy) ---
    line, = ax1.plot(x_indices, acc_mapped, color=LINE_COLOR, marker='o', markersize=6, label="Accuracy", zorder=10)
    
    # Shaded region logic
    last_acc_real = acc_real[-1]
    upper_bound_real = last_acc_real + 1.0
    lower_bound_real = last_acc_real - 1.0
    
    # Map bounds to the plotted space
    upper_bound_mapped = map_values_to_linear_ticks(upper_bound_real, CUSTOM_Y_TICKS_LEFT)
    lower_bound_mapped = map_values_to_linear_ticks(lower_bound_real, CUSTOM_Y_TICKS_LEFT)
    last_point_mapped = map_values_to_linear_ticks(last_acc_real, CUSTOM_Y_TICKS_LEFT)

    # Draw dashed lines
    ax1.axhline(y=upper_bound_mapped, color=SHADE_COLOR, linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=lower_bound_mapped, color=SHADE_COLOR, linestyle='--', linewidth=1, alpha=0.5)
    
    # Fill between
    ax1.fill_between([min(x_indices)-1, max(x_indices)+1], 
                     lower_bound_mapped, upper_bound_mapped, 
                     color=SHADE_COLOR, alpha=0.1, zorder=0)

    # ================= AXIS CONFIGURATION =================

    # --- X Axis ---
    ax1.set_xlabel(X_LABEL, fontweight='bold', fontsize=12)
    ax1.set_xticks(x_indices)
    # Format labels: 5.0 -> 5, 10.0 -> 10
    ax1.set_xticklabels([f"{int(val)}" if val.is_integer() else str(val) for val in x_real])
    ax1.set_xlim(min(x_indices) - 0.6, max(x_indices) + 0.6)

    # --- Left Y Axis (Accuracy) ---
    ax1.set_ylabel(Y_LABEL_LEFT, fontweight='bold', fontsize=12)
    ax1.set_yticks(np.arange(len(CUSTOM_Y_TICKS_LEFT)))
    ax1.set_yticklabels([str(y) for y in CUSTOM_Y_TICKS_LEFT])
    ax1.set_ylim(0, len(CUSTOM_Y_TICKS_LEFT) - 1)
    ax1.spines['top'].set_visible(False)
    
    # --- Right Y Axis (Time) ---
    ax2.set_ylabel(Y_LABEL_RIGHT, fontweight='bold', fontsize=12, rotation=270, labelpad=15)
    ax2.set_yticks(np.arange(len(CUSTOM_Y_TICKS_RIGHT)))
    ax2.set_yticklabels([str(y) for y in CUSTOM_Y_TICKS_RIGHT])
    ax2.set_ylim(0, len(CUSTOM_Y_TICKS_RIGHT) - 1)
    ax2.spines['top'].set_visible(False)

    # --- Z-Order & Grid ---
    # Bring ax1 to front so line is over bars
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    
    # Grid (on ax1, matching the custom ticks)
    ax1.grid(True, axis='y', linestyle='--', alpha=GRID_ALPHA)

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, OUTPUT_PDF)
    plt.savefig(pdf_path)
    print(f"Plot saved to: {pdf_path}")

if __name__ == "__main__":
    main()
