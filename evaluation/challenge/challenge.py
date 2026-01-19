import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from scipy.interpolate import PchipInterpolator

# ==========================================
# CONFIGURATION
# ==========================================
plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# Colors
COLOR_PARETO_LINE = '#e41a1c'      # Red
COLOR_BUDGET_LINE = '#868686'      # Grey
COLOR_ALL_POINTS = '#868686'       # Grey for scatter points
COLOR_TEXT_GREY = '#868686'        # Dark Grey for text
COLOR_TEXT_RED = '#e41a1c'         # Darker Red for text readability

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON_PATH = os.path.join(current_dir, "merged_output.json")
OUTPUT_IMAGE_PATH = os.path.join(current_dir, "challenge_combined.pdf")

# ==========================================
# LEFT PLOT: SEARCH SPACE
# ==========================================

def plot_search_space(ax):
    # 1. Define Control Points
    stages = [
        "No\nCompression", 
        "Pipeline\nSelection", 
        "Moudle\nChoice", 
        "Coarse\nSetting", 
        "Fine\nTuning" 
    ]
    x_indices = np.arange(len(stages))
    
    # Requirement: 
    # Stages 0, 1, 2: < 100, flat growth
    # Stage 3: Distinct growth
    # Stage 4: Explosive (~8000)
    y_points = np.array([1, 3, 50, 600, 8000])

    # ---------------------------------------------------------
    # Custom Axis Transformation Logic (Gradually Increasing Spacing)
    # ---------------------------------------------------------
    def transform_y(y):
        y = np.asarray(y)
        # Ensure y >= 1 since we start at 10^0
        y = np.maximum(y, 1.0)
        
        log_y = np.log10(y)
        
        # Power factor k > 1 creates expanding intervals
        k = 1.7
        return np.power(log_y, k)

    # 2. Interpolation in Transformed Space (Visual Smoothness)
    y_points_trans = transform_y(y_points)
    
    spline = PchipInterpolator(x_indices, y_points_trans)
    
    x_smooth = np.linspace(0, 4, 300)
    y_smooth_trans = spline(x_smooth)

    # 3. Plotting
    # Add Background Regions (Stages)
    ax.axvspan(0, 2, color=COLOR_BUDGET_LINE, alpha=0.1, lw=0)
    ax.axvspan(2, 4, color=COLOR_PARETO_LINE, alpha=0.1, lw=0)
    
    # Add Text Labels for Regions
    text_y_val = 15000
    text_y_trans = transform_y(text_y_val)
    
    ax.text(1.0, text_y_trans, "Pipeline/Module Choices", 
            ha='center', fontsize=11, color=COLOR_TEXT_GREY, fontweight='bold')
             
    ax.text(3, text_y_trans, "Hybrid Parameter Tuning", 
            ha='center', fontsize=11, color=COLOR_TEXT_RED, fontweight='bold')

    # Draw Smooth Curve
    ax.plot(x_smooth, y_smooth_trans, color=COLOR_PARETO_LINE, linestyle='-', linewidth=2, 
            label='Search Space Size', zorder=5)
    
    # Draw Markers
    ax.scatter(x_indices, y_points_trans, color=COLOR_PARETO_LINE, marker='o', s=50, zorder=6)

    # 4. Budget Line
    budget_y_trans = transform_y(200)
    ax.axhline(y=budget_y_trans, color=COLOR_BUDGET_LINE, linestyle='--', linewidth=1.5, 
               zorder=3)
    
    ax.text(1, budget_y_trans + 0.2, 'Profile Budget', color=COLOR_BUDGET_LINE, 
            fontsize=9, verticalalignment='bottom', fontweight='bold', ha='center')

    # 5. Axes Configuration
    ax.set_xticks(x_indices)
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_xlabel('Configuration Granularity', fontweight='bold', fontsize=12, labelpad=10)
    
    # Y Axis - Manual Ticks
    tick_values = [1, 10, 100, 1000, 10000]
    tick_locs = transform_y(tick_values)
    tick_labels = [r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$']
    
    ax.set_yticks(tick_locs)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_ylabel('Search Space Size', fontweight='bold', fontsize=12)
    
    ax.set_ylim(-0.5, transform_y(40000))
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title("Search Space Growth", fontweight='bold', fontsize=12)


# ==========================================
# RIGHT PLOT: PARETO FRONTIER
# ==========================================

def get_pareto_frontier(points):
    """
    points: list of dict, e.g. [{'x': 90, 'y': 5, 'id': 1}, ...]
    Maximize X (Accuracy), Minimize Y (Latency)
    """
    # Sort: X desc, Y asc
    sorted_points = sorted(points, key=lambda p: (p['x'], -p['y']), reverse=True)
    
    pareto_front = []
    current_min_y = float('inf')
    
    for p in sorted_points:
        if p['y'] < current_min_y:
            pareto_front.append(p)
            current_min_y = p['y']
    
    return sorted(pareto_front, key=lambda p: p['x'])

def plot_pareto_frontier(ax):
    # 1. Read Data
    if not os.path.exists(INPUT_JSON_PATH):
        # Fallback / Mock data if file missing for demonstration
        print(f"Warning: {INPUT_JSON_PATH} not found. Using mock data.")
        # Mocking some data that looks like a Pareto front
        data = []
        import random
        for _ in range(50):
            acc = random.uniform(80, 100)
            lat = random.uniform(10, 100)
            data.append({'accuracy': acc, 'latency': lat, 'config_id': int(acc*lat)})
    else:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # 2. Extract Data
    KEY_ACCURACY = 'accuracy'
    KEY_LATENCY = 'latency'
    KEY_ID = 'config_id'
    
    points = []
    for entry in data:
        if KEY_ACCURACY in entry and KEY_LATENCY in entry:
            points.append({
                'x': entry[KEY_ACCURACY],
                'y': entry[KEY_LATENCY],
                'id': entry.get(KEY_ID, 'N/A')
            })
    
    if not points:
        return

    # 3. Calculate Pareto
    pareto_points = get_pareto_frontier(points)
    
    pareto_ids = set(p['id'] for p in pareto_points)
    non_pareto_points = [p for p in points if p['id'] not in pareto_ids]
    
    non_pareto_x = [p['x'] for p in non_pareto_points]
    non_pareto_y = [p['y'] for p in non_pareto_points]
    
    pareto_x = [p['x'] for p in pareto_points]
    pareto_y = [p['y'] for p in pareto_points]
    
    # 4. Plot
    # Pareto Line + Points
    ax.plot(pareto_x, pareto_y, c=COLOR_PARETO_LINE, linestyle='--', linewidth=2, 
             marker='o', markersize=7, label='Pareto Frontier', zorder=5)

    # Non-Pareto Points
    if non_pareto_x:
        ax.scatter(non_pareto_x, non_pareto_y, facecolors='none', edgecolors=COLOR_ALL_POINTS, 
                   linewidths=1.5, label='Trials')

    # Decoration
    ax.set_title("Pareto Frontier", fontweight='bold', fontsize=12)
    ax.set_xlabel('Relative Accuracy (%)', fontweight='bold', fontsize=12, labelpad=10)
    ax.set_ylabel('Latency (ms)', fontweight='bold', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.legend(prop={'weight': 'bold'}, loc='upper right')
    
    # Invert Y axis (Latency: lower is better -> higher visual position preferred in this context? 
    # Original code inverted Y: plt.gca().invert_yaxis(). 
    # Typically inverted Y means 0 is at top. 
    # Wait, original code: "反转 Y 轴，使得数值越大越靠下 (Latency: 越小越好 -> 越靠上越好)"
    # Standard plot: 0 at bottom. Invert: 0 at top.
    # If we want "better (smaller)" to be "higher", we invert.
    ax.invert_yaxis()


# ==========================================
# MAIN
# ==========================================

def main():
    # Create 1x2 Subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.25)
    
    # Plot Left
    plot_search_space(axes[0])
    
    # Plot Right
    plot_pareto_frontier(axes[1])
    
    # Save
    plt.tight_layout()
    
    # Align x-axis labels (must be done after tight_layout)
    fig.align_xlabels(axes)
    
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"Combined plot saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    main()

