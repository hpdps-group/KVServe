import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
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
COLOR_TEXT_GREY = '#868686'        # Dark Grey for text
COLOR_TEXT_RED = '#e41a1c'         # Darker Red for text readability

OUTPUT_IMAGE_PATH = "search_space_growth.pdf"

def main():
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
    # Goal: Spacing between 10^k and 10^(k+1) increases with k.
    #       Gap(0-1) < Gap(1-2) < Gap(2-3) < Gap(3-4)
    #       Use Power Function on Log Scale: T(y) = (log10(y))^k
    # ---------------------------------------------------------
    
    def transform_y(y):
        y = np.asarray(y)
        # Ensure y >= 1 since we start at 10^0
        y = np.maximum(y, 1.0)
        
        log_y = np.log10(y)
        
        # Power factor k > 1 creates expanding intervals
        # k=1.7 roughly gives the 30% / 70% split for 10^0-10^2 vs 10^2-10^4
        k = 1.7
        return np.power(log_y, k)

    # 2. Interpolation in Transformed Space (Visual Smoothness)
    # Transform control points
    y_points_trans = transform_y(y_points)
    
    # PchipInterpolator for smooth, monotonic curve in visual space
    spline = PchipInterpolator(x_indices, y_points_trans)
    
    x_smooth = np.linspace(0, 4, 300)
    y_smooth_trans = spline(x_smooth)

    # 3. Plotting
    plt.figure(figsize=(8, 5))
    
    # Add Background Regions (Stages)
    # Region 1: Discrete Choices (Indices 0, 1, 2 -> x: -0.5 to 2.5)
    plt.axvspan(-0.5, 2.5, color=COLOR_BUDGET_LINE, alpha=0.1, lw=0)
    
    # Region 2: Continuous/Hybrid Tuning (Indices 3, 4 -> x: 2.5 to 4.5)
    plt.axvspan(2.5, 4.5, color=COLOR_PARETO_LINE, alpha=0.1, lw=0)
    
    # Add Text Labels for Regions (at top of graph)
    # Place text around Y=15000 (transformed)
    text_y_val = 15000
    text_y_trans = transform_y(text_y_val)
    
    plt.text(1.0, text_y_trans, "Pipeline/Module Choices", 
             ha='center', fontsize=11, color=COLOR_TEXT_GREY, fontweight='bold')
             
    plt.text(3.5, text_y_trans, "Hybrid Parameter Tuning", 
             ha='center', fontsize=11, color=COLOR_TEXT_RED, fontweight='bold')

    # Draw Smooth Curve (in transformed Y coordinates)
    plt.plot(x_smooth, y_smooth_trans, color=COLOR_PARETO_LINE, linestyle='-', linewidth=2, 
             label='Search Space Size', zorder=5)
    
    # Draw Markers
    plt.scatter(x_indices, y_points_trans, color=COLOR_PARETO_LINE, marker='o', s=50, zorder=6)

    # 4. Budget Line
    # Budget at 200
    # Transform the budget value
    budget_y_trans = transform_y(200)
    plt.axhline(y=budget_y_trans, color=COLOR_BUDGET_LINE, linestyle='--', linewidth=1.5, 
                label='Profile Budget', zorder=3)
    
    plt.text(1, budget_y_trans + 0.2, 'Profile Budget', color=COLOR_BUDGET_LINE, 
             fontsize=9, verticalalignment='bottom', fontweight='bold', ha='center')

    # 5. Axes Configuration
    # X Axis
    plt.xticks(x_indices, stages, fontsize=10)
    plt.xlabel('Configuration Granularity', fontweight='bold', fontsize=12, labelpad=10)
    
    # Y Axis - Manual Ticks
    # We are plotting in 'transformed' space, so we need to put ticks at transformed locations
    # and label them with original values.
    tick_values = [1, 10, 100, 1000, 10000]
    tick_locs = transform_y(tick_values)
    tick_labels = [r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$']
    
    plt.yticks(tick_locs, tick_labels, fontsize=10)
    plt.ylabel('Search Space Size', fontweight='bold', fontsize=12)
    
    # Set Y limits (in transformed space)
    # Start slightly below 0 (transformed 10^0) to give padding
    plt.ylim(-0.5, transform_y(40000))
    
    # Grid
    plt.grid(True, linestyle=':', alpha=0.6)

    # 6. Decoration
    plt.title("Search Space Growth", fontweight='bold', fontsize=12)
    # Legend location updated to not overlap with text
    # plt.legend(prop={'weight': 'bold'}, loc='upper left', bbox_to_anchor=(0, 0.9))

    # 7. Save
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"Plot saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    main()
