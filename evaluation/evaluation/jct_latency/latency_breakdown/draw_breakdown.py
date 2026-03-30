import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea, DrawingArea
import matplotlib.ticker as mtick

# ================= USER CONFIGURATION START =================

# Output filename
OUTPUT_FILENAME = "jct_breakdown.pdf"

# Stage Colors (Mapping stage name to hex color)
STAGE_COLORS = {
    'Prefill': '#FFBB78',      # Grey
    'Compress': '#C5B0D5',     # Light Orange
    'Transmition': '#e41a1c',  # Sky Blue
    'Decompress': '#C5B0D5',   # Medium Purple
    'Decode': '#FFBB78',       # Emerald Green
}
# Fallback color for unknown stages
DEFAULT_COLOR = '#333333'

# Text Colors for Labels (Mapping stage name to text color)
# Controls the color of the percentage text inside the bars
STAGE_TEXT_COLOR_MAP = {
    'Prefill': 'black',
    'Compress': 'black',
    'Transmition': 'white',
    'Decompress': 'black',
    'Decode': 'black',
}
DEFAULT_TEXT_COLOR = 'black'

# Stage Hatches (Mapping stage name to hatch pattern)
STAGE_HATCHES = {
    'Prefill': '/',
    'Compress': '///',
    'Transmition': '..',  # Matches CSV column name
    'Decompress': '\\\\\\',
    'Decode': '\\',
}

# Display labels for legend (Map CSV column to Display Name)
STAGE_LABELS = {
    'Prefill': 'Prefill',
    'Compress': 'Compression',
    'Transmition': 'Communication',
    'Decompress': 'Decompression',
    'Decode': 'Decode',
}

# Ordered list of stages to stack (Must match CSV columns)
STAGES_ORDER = ['Prefill', 'Compress', 'Transmition', 'Decompress', 'Decode']

# Minimum width for any non-zero segment (0.05 = 5%)
MIN_DISPLAY_WIDTH = 0.03

# Plot styling
FIGURE_SIZE = (10, 3)
BAR_HEIGHT = 0.4
Y_AXIS_SPACING = 0.55 # Control vertical spacing between bars
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 12

# ================= USER CONFIGURATION END =================

# Global plot settings
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": FONT_FAMILY,
    "axes.linewidth": 1.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
})

# ================= Legend Construction =================

def create_combined_legend_box(fig, handles_row2, labels_row2):
    """
    Constructs a single-row legend for stages using offsetbox.
    """
    
    def create_legend_item(handle, label):
        # Icon area
        da = DrawingArea(width=22, height=10, xdescent=0, ydescent=0)
        
        if isinstance(handle, Patch):
            rect = Rectangle((0, 0), width=22, height=10,
                             facecolor=handle.get_facecolor(),
                             edgecolor=handle.get_edgecolor(),
                             hatch=handle.get_hatch(),
                             linewidth=handle.get_linewidth())
            da.add_artist(rect)
        elif isinstance(handle, Line2D):
            line = Line2D([0, 11, 22], [5, 5, 5],
                          color=handle.get_color(),
                          linewidth=handle.get_linewidth(),
                          linestyle=handle.get_linestyle(),
                          marker=handle.get_marker(),
                          markersize=handle.get_markersize(),
                          markeredgecolor=handle.get_markeredgecolor(),
                          markeredgewidth=handle.get_markeredgewidth(),
                          markevery=[1])
            da.add_artist(line)
        
        # Text area
        ta = TextArea(label, textprops=dict(color="black", size=11, family="DejaVu Sans", fontweight='bold'))
        
        return HPacker(children=[da, ta], align="center", pad=0, sep=5)

    # Row 2 Items (Stages)
    row2_items = [create_legend_item(h, l) for h, l in zip(handles_row2, labels_row2)]
    row2_packer = HPacker(children=row2_items, align="center", pad=0, sep=15)
    
    # Box container
    anchored_box = AnchoredOffsetbox(
        loc='lower center',
        child=row2_packer,
        pad=0,
        frameon=True,
        bbox_to_anchor=(0.55, 0.9), # Position relative to figure
        bbox_transform=fig.transFigure,
        borderpad=0.
    )
    
    anchored_box.patch.set_boxstyle("round,pad=0.3")
    anchored_box.patch.set_linewidth(1.2)
    anchored_box.patch.set_edgecolor('black')
    anchored_box.patch.set_facecolor('white')
    
    return anchored_box

def adjust_ratios_for_min_width(ratios, min_width):
    """
    Adjusts a list of ratios (that sum to 1.0) so that any non-zero element
    has at least `min_width`. The deficit is subtracted from 'Prefill' (index 0).
    """
    # Identify indices that need expansion (0 < x < min)
    small_mask = (ratios > 0) & (ratios < min_width)
    if not np.any(small_mask):
        return ratios

    new_ratios = ratios.copy()
    
    # Calculate total needed amount to bring small segments up to min_width
    current_small_vals = ratios[small_mask]
    needed_total = np.sum(min_width - current_small_vals)
    
    # Set small items to min_width
    new_ratios[small_mask] = min_width
    
    # Subtract from Prefill (index 0)
    # We assume index 0 is Prefill based on STAGES_ORDER
    if small_mask[0]:
        # If Prefill is small, we can't subtract from it. 
        pass
    
    # Subtract from Prefill
    new_ratios[0] -= needed_total
    
    # Safety check: ensure Prefill doesn't go negative
    if new_ratios[0] < 0:
        new_ratios[0] = 0
        
    return new_ratios

def plot_single_breakdown(ax, data, title, show_ylabel=True):
    """
    Plots a horizontal stacked bar chart with percentage-based X-axis.
    Implements minimum width constraint for visibility.
    Adds text labels showing original percentage.
    """
    # Reverse data so first item in CSV appears at the top
    data = data.iloc[::-1].reset_index(drop=True)
    
    methods = data['Method'].tolist()
    # Adjust y positions to be closer
    y_pos = np.arange(len(methods)) * Y_AXIS_SPACING
    
    # 1. Calculate raw ratios
    # Get values for all ordered stages
    raw_values_df = data[STAGES_ORDER]
    total_latency = raw_values_df.sum(axis=1).values
    
    # Calculate initial ratios matrix (Rows: Methods, Cols: Stages)
    ratios_matrix = raw_values_df.div(total_latency, axis=0).values
    
    # 2. Adjust ratios row by row to enforce MIN_DISPLAY_WIDTH
    adjusted_ratios_list = []
    for row_ratios in ratios_matrix:
        adj_row = adjust_ratios_for_min_width(row_ratios, MIN_DISPLAY_WIDTH)
        adjusted_ratios_list.append(adj_row)
    
    adjusted_ratios_matrix = np.array(adjusted_ratios_list)
    
    # 3. Plot using adjusted ratios
    # We iterate by STAGE (column) to plot stacked bars
    
    # Initialize offsets for stacking
    current_offsets = np.zeros(len(methods))
    
    for col_idx, stage in enumerate(STAGES_ORDER):
        # Get adjusted ratios for this stage across all methods
        stage_ratios = adjusted_ratios_matrix[:, col_idx]
        original_ratios = ratios_matrix[:, col_idx]
        
        # Determine color for this stage
        stage_color = STAGE_COLORS.get(stage, DEFAULT_COLOR)
        
        # Plot bars
        ax.barh(y_pos, stage_ratios, left=current_offsets, height=BAR_HEIGHT,
                color=stage_color, edgecolor='black', hatch=STAGE_HATCHES.get(stage, ''),
                linewidth=1.2, zorder=3, alpha=1)
        
        # Add text labels
        for i, (ratio, orig_ratio, offset, method) in enumerate(zip(stage_ratios, original_ratios, current_offsets, methods)):
            # Only show label if original ratio is significant enough to care about, 
            # or maybe always show if it fits?
            # User said "show proportion situation", so we show the number.
            # We use the adjusted width to determine if we have space to draw it, 
            # but the number displayed is the ORIGINAL ratio.
            
            # Use a small threshold for displaying text to avoid clutter on 0% items
            if ratio > 0.005 and stage == 'Transmition': 
                # Calculate center of the bar
                x_center = offset + ratio / 2
                y_center = y_pos[i]
                
                # Get text color
                text_color = STAGE_TEXT_COLOR_MAP.get(stage, DEFAULT_TEXT_COLOR)
                stage_color = STAGE_COLORS.get(stage, DEFAULT_COLOR)
                
                # Format percentage
                pct_text = f"{orig_ratio*100:.0f}"
                if pct_text == "0" and orig_ratio > 0:
                     pct_text = "1"
                
                # Add text with bbox to hide hatch lines behind text
                ax.text(x_center, y_center, pct_text, 
                        ha='center', va='center', 
                        fontsize=10, fontweight='bold', 
                        color=text_color, 
                        bbox=dict(facecolor=stage_color, edgecolor='white', boxstyle='round,pad=0.2', alpha=1.0, linewidth=1.5),
                        zorder=10)

        # Update offsets
        current_offsets += stage_ratios

    # Configure Y Axis
    ax.set_yticks(y_pos)
    if show_ylabel:
        ax.set_yticklabels(methods, fontsize=14, fontweight='bold')
    else:
        ax.set_yticklabels([]) # Hide labels
        
    # Set Y Lim to fit tight
    # Default is roughly -0.5 to max+0.5
    # We want -margin to max+margin
    margin = BAR_HEIGHT
    ax.set_ylim(min(y_pos) - 0.3, max(y_pos) + 0.3)
        
    # Configure X Axis (Percentage)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{int(x*100)}'))
    ax.set_xticks(np.arange(0.25, 1.1, 0.25)) # 0, 0.2, 0.4, ... 1.0
    
    # Title with custom xy position
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10, loc='center', y=0.95, x=0.5)
    
    # Grid and Spines
    ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def main():
    # Load data
    csv_path = "latency.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Fallback for relative path
        import os
        csv_path = os.path.join(os.path.dirname(__file__), "latency.csv")
        df = pd.read_csv(csv_path)

    # Filter data for each dataset
    df_wiki = df[df['Dataset'] == '2WikiMQA'].copy()
    df_hotpot = df[df['Dataset'] == 'HotpotQA'].copy()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    # Plot Left (2WikiMQA) - Show Y Labels
    plot_single_breakdown(ax1, df_wiki, "2WikiMQA", show_ylabel=True)
    
    # Plot Right (HotpotQA) - Hide Y Labels (Shared)
    plot_single_breakdown(ax2, df_hotpot, "HotpotQA", show_ylabel=False)
    
    # Shared X Axis Label
    # Use fig.supxlabel to place one label for both subplots
    fig.supxlabel("Percentage of Total Latency (%)", fontweight='bold', fontsize=16, y=-0.02, x=0.55)
    
    # Prepare Legend Handles
    
    # Row 2: Stages (Colors + Hatches)
    row2_handles = []
    row2_labels = []
    for stage in STAGES_ORDER:
        p = Patch(facecolor=STAGE_COLORS.get(stage, 'white'), edgecolor='black', hatch=STAGE_HATCHES.get(stage, ''), linewidth=1.2)
        row2_handles.append(p)
        row2_labels.append(STAGE_LABELS.get(stage, stage))
        
    # Create and add legend (Only Row 2)
    custom_legend = create_combined_legend_box(fig, row2_handles, row2_labels)
    fig.add_artist(custom_legend)
    
    # Layout adjustment
    plt.tight_layout()
    # Adjust margins: top for legend, bottom for supxlabel, wspace for shared y-axis look
    plt.subplots_adjust(top=0.78, bottom=0.18, wspace=0.07)
    
    # Save
    plt.savefig(OUTPUT_FILENAME, bbox_inches='tight')
    print(f"Plot generated: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()
