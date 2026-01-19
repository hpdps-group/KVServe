import json
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle

# ================= CONFIGURATION =================
# Files to read (filenames located in the same directory as this script)
# You can add more files to this list
TARGET_FILES = [
    "13.json",
    "15.json",
    "27.json",
    "43.json",
]

# Labels and Text
X_LABEL = "Request ID"
Y_LABEL = "Compression Ratio"
LEGEND_LABELS = {
    "13.json": "MixHQ 1",  # Custom legend mapping
    "15.json": "MixHQ 2",
    "27.json": "MixHQ 3",
    "43.json": "MixHQ 4",
}
DEFAULT_LEGEND_LABEL = "Data"

# Zoom Configuration (Magnifying Glass)
ENABLE_ZOOM = True
ZOOM_CONFIG = {
    "x_min": 0,      # X start for zoom
    "x_max": 6,      # X end for zoom
    "y_min": 8.3,     # Y start for zoom
    "y_max": 8.5,    # Y end for zoom
    "loc": "upper center", # Location of the inset axes
    "width": "40%",   # Width of inset axes relative to parent
    "height": "30%",  # Height of inset axes relative to parent
    # bbox_to_anchor controls the position more precisely. 
    # (x, y, width, height) in normalized axes coordinates.
    # Adjust x, y to move the box.
    "bbox_to_anchor": (0, 0, 1, 1) 
}

# Output Settings
OUTPUT_PDF = "request_cr_motivation.pdf"

# Plot Aesthetics (Academic Style)
FIGURE_SIZE = (6, 4)     # Width, Height in inches
LINE_WIDTH = 1.5
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd'] # Standard academic qualitative colors
GRID_ALPHA = 0.3

# Matplotlib rcParams for Publication Quality
plt.rcParams.update({
    "font.family": "DejaVu Sans",             # Use serif/Times like fonts
    # "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,              # Thicker axes lines
    "lines.linewidth": LINE_WIDTH,
    "grid.linestyle": "--",
    "grid.alpha": GRID_ALPHA,
    "xtick.direction": "in",            # Ticks pointing inwards
    "ytick.direction": "in",
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})
# =================================================

def load_data(file_name):
    """Loads JSON data from the same directory as the script."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping.")
        return None

    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def draw_lines(ax, datasets, show_legend=False):
    """
    Helper function to draw lines on a given axes.
    datasets: list of tuples (label, x, y, color)
    """
    for label, x, y, color in datasets:
        ax.plot(x, y, label=label, color=color, alpha=0.9, linewidth=LINE_WIDTH)
    
    if show_legend:
        ax.legend(frameon=False) # No box around legend is cleaner

def main():
    # Pre-load all data
    datasets = []
    for idx, file_name in enumerate(TARGET_FILES):
        data = load_data(file_name)
        if data is None:
            continue
        
        x = np.arange(len(data))
        y = np.array(data)
        label = LEGEND_LABELS.get(file_name, file_name.replace(".json", ""))
        color = COLORS[idx % len(COLORS)]
        
        datasets.append((label, x, y, color))

    if not datasets:
        print("No datasets loaded.")
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # 1. Draw Main Plot
    draw_lines(ax, datasets, show_legend=True)

    # Styling axes
    ax.set_xlabel(X_LABEL, fontweight='bold', fontsize=12)
    ax.set_ylabel(Y_LABEL, fontweight='bold', fontsize=12)
    ax.grid(True)
    
    # Remove top and right spines for a cleaner academic look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 2. Draw Zoom Inset (Magnifying Glass)
    if ENABLE_ZOOM:
        # Create inset axes
        axins = inset_axes(ax, 
                           width=ZOOM_CONFIG["width"], 
                           height=ZOOM_CONFIG["height"], 
                           loc=ZOOM_CONFIG["loc"],
                           bbox_to_anchor=ZOOM_CONFIG["bbox_to_anchor"],
                           bbox_transform=ax.transAxes)
        
        # Plot data on inset axes
        draw_lines(axins, datasets, show_legend=False)
        
        # Set limits for zoom
        axins.set_xlim(ZOOM_CONFIG["x_min"], ZOOM_CONFIG["x_max"])
        axins.set_ylim(ZOOM_CONFIG["y_min"], ZOOM_CONFIG["y_max"])
        
        # Style inset axes
        axins.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                          labelbottom=False, labelleft=False) 
        axins.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
        
        # Draw Rectangle on Main Plot
        rect = Rectangle((ZOOM_CONFIG["x_min"], ZOOM_CONFIG["y_min"]), 
                         width=(ZOOM_CONFIG["x_max"] - ZOOM_CONFIG["x_min"]), 
                         height=(ZOOM_CONFIG["y_max"] - ZOOM_CONFIG["y_min"]),
                         linewidth=1.0, edgecolor='k', facecolor='none', linestyle='--', alpha=0.8)
        ax.add_patch(rect)
        
        # Add connection lines
        # Connect Box Top-Right to Inset Left (somewhere)
        con1 = ConnectionPatch(xyA=(ZOOM_CONFIG["x_max"], ZOOM_CONFIG["y_max"]), coordsA=ax.transData,
                               xyB=(0, 1), coordsB=axins.transAxes,
                               axesA=ax, axesB=axins,
                               arrowstyle="-", linestyle="--", linewidth=1.0, color="k", alpha=0.8)
        
        con2 = ConnectionPatch(xyA=(ZOOM_CONFIG["x_max"], ZOOM_CONFIG["y_min"]), coordsA=ax.transData,
                               xyB=(0, 0), coordsB=axins.transAxes,
                               axesA=ax, axesB=axins,
                               arrowstyle="-", linestyle="--", linewidth=1.0, color="k", alpha=0.8)
        
        ax.add_artist(con1)
        ax.add_artist(con2)


    # Save outputs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, OUTPUT_PDF)
    
    plt.savefig(pdf_path)
    print(f"Plot saved to:\n  - {pdf_path}")
    # plt.show() 

if __name__ == "__main__":
    main()
