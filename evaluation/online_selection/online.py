import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, DrawingArea

# ================= USER CONFIGURATION =================
# Colors for the 4 profiles (p1, p2, p3, p4)
PROFILE_COLORS = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD']

# Colors for Envelopes
LOWER_ENVELOPE_COLOR = '#868686'
OBSERVED_ENVELOPE_COLOR = '#e41a1c'

# Line Widths
LINE_WIDTH_PROFILE = 3
LINE_WIDTH_ENVELOPE = 4

# ================= STYLE SETTINGS =================
plt.rcParams.update({
    "font.size": 14,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "mathtext.default": "regular",
    "axes.unicode_minus": False
})

# ================= HELPER FUNCTIONS =================
def create_legend_box(ax, handles, labels):
    """
    Constructs a centered horizontal legend using offsetbox, placed above the plot.
    """
    def create_legend_item(handle, label):
        da = DrawingArea(width=22, height=10, xdescent=0, ydescent=0)
        
        if isinstance(handle, Line2D):
            line = Line2D([0, 11, 22], [5, 5, 5],
                          color=handle.get_color(),
                          linewidth=handle.get_linewidth(),
                          linestyle=handle.get_linestyle(),
                          marker=handle.get_marker(),
                          markersize=handle.get_markersize())
            da.add_artist(line)
        
        ta = TextArea(label, textprops=dict(color="black", size=14, family="DejaVu Sans", fontweight='bold'))
        return HPacker(children=[da, ta], align="center", pad=0, sep=5)

    # Build items
    items = [create_legend_item(h, l) for h, l in zip(handles, labels)]
    packer = HPacker(children=items, align="center", pad=0, sep=10)
    
    # Put into AnchoredOffsetbox
    anchored_box = AnchoredOffsetbox(
        loc='lower center',
        child=packer,
        pad=0,
        frameon=True,
        bbox_to_anchor=(0.5, 1.02), # Centered above the axes
        bbox_transform=ax.transAxes,
        borderpad=0.4
    )
    
    # Box style
    anchored_box.patch.set_boxstyle("round,pad=0.4")
    anchored_box.patch.set_linewidth(1.2)
    anchored_box.patch.set_edgecolor('black')
    anchored_box.patch.set_facecolor('white')
    
    return anchored_box

# ================= MAIN LOGIC =================
# 1. Data generation logic (unchanged)
def p1_func(x): return 2.5 * x + 0.30
def p2_func(x): return 1.3 * x + 0.70 
def p3_func(x): return 0.75 * x + 0.98
def p4_func(x): return 0.35 * x + 1.28

x = np.linspace(0, 1.05, 600)

y1 = p1_func(x)
y2 = p2_func(x)
y3 = p3_func(x)
y4 = p4_func(x)

y_envelope = np.minimum(np.minimum(y1, y2), np.minimum(y3, y4))

y1_drift = y1
y2_drift = y2 - 0.06 
y3_drift = y3 + 0.04 
y_obs_envelope = np.minimum(np.minimum(y1_drift, y2_drift), y3_drift)

# Calculate intersections
x1 = (0.70 - 0.30) / (2.5 - 1.3)
x2 = (0.98 - 0.70) / (1.3 - 0.75)

# Blue line logic
y_red_at_x1 = p1_func(x1)
slope_p3 = 0.75
y_blue_at_x2 = y_red_at_x1 + slope_p3 * (x2 - x1)
idx_x2 = np.abs(x - x2).argmin()
offset_x2 = y_blue_at_x2 - y_obs_envelope[idx_x2]

y_blue = np.where(
    (x >= x1) & (x <= x2),
    y_red_at_x1 + slope_p3 * (x - x1),
    y_obs_envelope + offset_x2,
)

# Calculate x3
mask_after_x2 = x >= x2
diff_blue_p3 = y_blue - y3
i_after = np.where(mask_after_x2)[0]
sign_changes = np.diff(np.sign(diff_blue_p3[i_after]))
cross_idx = np.where(sign_changes != 0)[0]
if len(cross_idx) > 0:
    i0 = i_after[cross_idx[0]]
    x3 = x[i0] - diff_blue_p3[i0] * (x[i0+1] - x[i0]) / (diff_blue_p3[i0+1] - diff_blue_p3[i0])
else:
    x3 = 0.8

# 3. Plotting
fig, ax = plt.subplots(figsize=(10, 4.5))

# --- Plot Profiles (dashed) ---
dashed_style = {'linestyle': '--', 'linewidth': LINE_WIDTH_PROFILE, 'alpha': 0.8}
ax.plot(x, y1, color=PROFILE_COLORS[0], label='$P_1$', **dashed_style)
ax.plot(x, y2, color=PROFILE_COLORS[1], label='$P_2$', **dashed_style)
ax.plot(x, y3, color=PROFILE_COLORS[2], label='$P_3$', **dashed_style)
ax.plot(x, y4, color=PROFILE_COLORS[3], label='$P_4$', **dashed_style)

# --- Plot Envelopes ---
# Lower Envelope
ax.plot(x, y_envelope, color=LOWER_ENVELOPE_COLOR, linewidth=LINE_WIDTH_ENVELOPE, label='Lower Envelope', alpha=0.9, zorder=10)

# Observed Lower Envelope
mask_blue = x >= x1
ax.plot(x[mask_blue], y_blue[mask_blue], color=OBSERVED_ENVELOPE_COLOR, linewidth=LINE_WIDTH_ENVELOPE, label='Observed Lower Envelope', zorder=11)

# --- Shaded Areas ---
ax.axvspan(x1, x2, color='#e41a1c', alpha=0.1, zorder=0)
ax.text((x1 + x2) / 2, 2, "Selected", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#e41a1c', zorder=10)
# --- Vertical Lines ---
for xc in [x1, x2, x3]:
    ax.axvline(x=xc, color='#868686', linestyle='--', linewidth=1.2)

# --- Text Annotations with Range Arrows ---
# Reference style from KVServe/evaluation/offline_profile/search_process/draw_search_process.py
boundaries = [0.0, x1, x2, x3, 1.0]
labels = ['Best: $P_1$', 'Best: $P_2$', 'Best: $P_3$', 'Best: $P_4$']
y_pos_arrow = 0.25

for i in range(4):
    start, end = boundaries[i], boundaries[i+1]
    mid = (start + end) / 2
    label = labels[i]
    
    # 1. Text with white background (masks the arrow line)
    # alpha = 0.1 if i == 1 else 1.0 
    facecolor = '#fce8e8' if i == 1 else 'white'
    ax.text(mid, y_pos_arrow, label, 
            ha='center', va='center', fontsize=15, color='#333333', fontweight='bold',
            bbox=dict(facecolor=facecolor, edgecolor='none', alpha=1, pad=3), zorder=10)
    
    # 2. Bidirectional arrow
    # Ensure arrow fits
    if end > start + 0.05:
        ax.annotate('', xy=(start, y_pos_arrow), xytext=(end, y_pos_arrow),
                    arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.2), zorder=9)


# Candidate Set Box
ax.text(0.025, 1.8, 'Best set = {$P_2$}\nNeighbor set = {$P_1, P_3$}\nCandidate set = {$P_1, P_2, P_3$}', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.4', linewidth=1.0))

# --- Arrows and Comments ---
arrow_tip_x = x2 + 0.02
arrow_tip_y = y_blue[np.abs(x - arrow_tip_x).argmin()]

# Using the style referenced from KVServe/evaluation/offline_profile/motivation/acc_time_cr.py
ax.annotate('Runtime Drift',
            xy=(arrow_tip_x, arrow_tip_y - 0.05),
            xytext=(arrow_tip_x + 0.2, arrow_tip_y - 0.45),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", color=OBSERVED_ENVELOPE_COLOR, lw=2.5, relpos=(0, 0.5)),
            fontsize=14, color=OBSERVED_ENVELOPE_COLOR, fontweight='bold', ha='left',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OBSERVED_ENVELOPE_COLOR, alpha=0.9, linewidth=1.5),
            zorder=20)

# --- Axes Settings ---
ax.set_xlim(0, 1.02)
ax.set_ylim(0.1, 2.45)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=False, nbins=6))
ax.set_xlabel('Parameter $x = 1/Bandwidth$', fontsize=18, labelpad=8, fontweight='bold')
ax.set_ylabel('Latency (s)', fontsize=18, labelpad=8, fontweight='bold')
# No Title as requested

# Custom Ticks
xticks = [0.0, 0.2, x1, 0.4, x2, 0.6, x3, 0.8, 1.0]
xticklabels = ['0.0', '0.2', '$x_1$', '0.4', '$x_2$', '0.6', '$x_3$', '0.8', '1.0']
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=15)
ax.tick_params(axis='y', labelsize=15, direction='out', top=False, right=False, which='both', length=4, width=1)
ax.tick_params(axis='x', direction='out', top=False, right=False, which='both', length=4, width=1)

# --- Grid ---
ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='#bfbfbf', alpha=0.6)

# --- Spines ---
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('black')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Custom Legend ---
legend_handles = []
legend_labels = []

# Profiles
for i in range(4):
    legend_handles.append(Line2D([0], [0], color=PROFILE_COLORS[i], linestyle='--', linewidth=LINE_WIDTH_PROFILE))
    legend_labels.append(f'$P_{i+1}$')

# Envelopes
legend_handles.append(Line2D([0], [0], color=LOWER_ENVELOPE_COLOR, linewidth=LINE_WIDTH_ENVELOPE))
legend_labels.append('Theoretical Envelope')

legend_handles.append(Line2D([0], [0], color=OBSERVED_ENVELOPE_COLOR, linewidth=LINE_WIDTH_ENVELOPE))
legend_labels.append('Observed Envelope')

# Create and add legend
custom_legend = create_legend_box(ax, legend_handles, legend_labels)
ax.add_artist(custom_legend)

plt.tight_layout()
_out_dir = os.path.dirname(os.path.abspath(__file__))
# Reserve space for top legend
plt.subplots_adjust(top=0.85) 
plt.savefig(os.path.join(_out_dir, 'lower_envelope.pdf'), dpi=400, bbox_inches='tight')
print(f"Plot generated: {os.path.join(_out_dir, 'lower_envelope.pdf')}")
# plt.show()
