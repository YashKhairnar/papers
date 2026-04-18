import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyArrowPatch

# ── helpers ────────────────────────────────────────────────────────────────────

def positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """
    PE = np.zeros((max_len, d_model))
    positions = np.arange(max_len)[:, None]          # (max_len, 1)
    dims      = np.arange(d_model)[None, :]           # (1, d_model)
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    PE[:, 0::2] = np.sin(angles[:, 0::2])             # even dims → sin
    PE[:, 1::2] = np.cos(angles[:, 1::2])             # odd  dims → cos
    return PE

# ── parameters ─────────────────────────────────────────────────────────────────

MAX_LEN   = 100        # number of token positions shown in the heatmap
D_MODEL   = 512        # embedding dimension (same as the original paper)
WAVE_DIMS = [0, 2, 10, 50, 100, 200]   # which dimensions to show as 1-D waves

# ── compute ────────────────────────────────────────────────────────────────────

PE = positional_encoding(MAX_LEN, D_MODEL)

# ── figure layout ──────────────────────────────────────────────────────────────

plt.style.use("dark_background")

fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")

outer = gridspec.GridSpec(
    3, 1,
    figure=fig,
    hspace=0.55,
    height_ratios=[0.6, 1.0, 0.9],
    left=0.06, right=0.97, top=0.93, bottom=0.06,
)

# Row 0 → title / formula banner
ax_title = fig.add_subplot(outer[0])

# Row 1 → heatmap
ax_heat = fig.add_subplot(outer[1])

# Row 2 → three 1-D wave panels side by side
wave_gs = outer[2].subgridspec(1, 3, wspace=0.38)
ax_sin  = fig.add_subplot(wave_gs[0])
ax_cos  = fig.add_subplot(wave_gs[1])
ax_both = fig.add_subplot(wave_gs[2])

ACCENT = "#58a6ff"    # blue
GOLD   = "#ffa657"    # orange
FG     = "#e6edf3"    # near-white
DIM    = "#8b949e"    # grey

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 0 – Title / formula banner
# ══════════════════════════════════════════════════════════════════════════════
ax_title.set_facecolor("#161b22")
ax_title.set_xlim(0, 1)
ax_title.set_ylim(0, 1)
ax_title.axis("off")

ax_title.text(
    0.5, 0.82,
    "Positional Encoding  —  Attention Is All You Need (Vaswani et al., 2017)",
    ha="center", va="center", fontsize=15, fontweight="bold", color=FG,
    transform=ax_title.transAxes,
)
ax_title.text(
    0.5, 0.52,
    r"$PE_{(pos,\,2i)}   = \sin\!\left(\dfrac{pos}{10000^{2i/d_{model}}}\right)$"
    r"     $PE_{(pos,\,2i+1)} = \cos\!\left(\dfrac{pos}{10000^{2i/d_{model}}}\right)$",
    ha="center", va="center", fontsize=13, color=ACCENT,
    transform=ax_title.transAxes,
)
explanation = (
    "• Each token at position pos gets a unique vector of length $d_{model}$.\n"
    "• Even dimensions use  sin, odd dimensions use  cos.\n"
    "• Lower dimensions oscillate fast (short wavelength); higher dimensions oscillate very slowly.\n"
    "• The model can learn to attend to relative distances because "
    r"$PE_{pos+k}$ is a linear function of $PE_{pos}$."
)
ax_title.text(
    0.5, 0.14,
    explanation,
    ha="center", va="center", fontsize=9.5, color=DIM, linespacing=1.7,
    transform=ax_title.transAxes,
)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 – Heatmap
# ══════════════════════════════════════════════════════════════════════════════
im = ax_heat.imshow(
    PE, aspect="auto", cmap="viridis", origin="upper",
    vmin=-1, vmax=1, interpolation="bilinear",
)
cbar = fig.colorbar(im, ax=ax_heat, pad=0.01, fraction=0.018)
cbar.set_label("PE value  (−1 → +1)", color=FG, fontsize=9)
cbar.ax.yaxis.set_tick_params(color=FG)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG, fontsize=8)

ax_heat.set_facecolor("#161b22")
ax_heat.set_title(
    "Positional Encoding Matrix  [100 positions × 512 dimensions]",
    color=FG, fontsize=11, pad=8,
)
ax_heat.set_xlabel("Embedding Dimension  →  lower dims: fast waves, higher dims: slow waves",
                   color=DIM, fontsize=9)
ax_heat.set_ylabel("Token Position  (0 = first word)", color=DIM, fontsize=9)
ax_heat.tick_params(colors=FG, labelsize=8)
for spine in ax_heat.spines.values():
    spine.set_edgecolor("#30363d")

# Annotate the "fast" region
ax_heat.annotate(
    "Fast oscillation\n(dim 0–20)\nshort wavelength\n→ captures local order",
    xy=(10, 50), xytext=(40, 72),
    fontsize=8, color=GOLD,
    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="#161b22", ec=GOLD, lw=0.8),
)
# Annotate the "slow" region
ax_heat.annotate(
    "Slow oscillation\n(dim 400–512)\nlong wavelength\n→ captures global order",
    xy=(450, 50), xytext=(330, 72),
    fontsize=8, color="#79c0ff",
    arrowprops=dict(arrowstyle="->", color="#79c0ff", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="#161b22", ec="#79c0ff", lw=0.8),
)
# Annotate the colour scale meaning
ax_heat.text(
    0.5, -0.13,
    "Colour encodes the PE value: yellow ≈ +1 (peak), purple ≈ −1 (trough), teal ≈ 0",
    ha="center", fontsize=8.5, color=DIM, transform=ax_heat.transAxes,
)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2a – sin waves (even dims)
# ══════════════════════════════════════════════════════════════════════════════
positions = np.arange(MAX_LEN)
sin_dims  = [d for d in WAVE_DIMS if d < D_MODEL and d % 2 == 0]
cos_dims  = [d for d in WAVE_DIMS if d < D_MODEL and d % 2 == 1] or [1, 3, 11, 51, 101, 201]

cmap_lines = plt.cm.plasma(np.linspace(0.15, 0.9, max(len(sin_dims), len(cos_dims))))

ax_sin.set_facecolor("#161b22")
for idx, d in enumerate(sin_dims):
    ax_sin.plot(positions, PE[:, d], color=cmap_lines[idx], lw=1.6,
                label=f"dim {d}")
ax_sin.set_title("sin  (even dimensions)", color=FG, fontsize=10, pad=6)
ax_sin.set_xlabel("Token position", color=DIM, fontsize=8.5)
ax_sin.set_ylabel("PE value", color=DIM, fontsize=8.5)
ax_sin.tick_params(colors=FG, labelsize=7)
ax_sin.legend(fontsize=7.5, loc="upper right", framealpha=0.15, labelcolor=FG)
ax_sin.set_ylim(-1.15, 1.45)
ax_sin.axhline(0, color="#30363d", lw=0.7, zorder=0)
ax_sin.text(
    0.5, 1.01,
    "Higher dim → longer wavelength → slower wave",
    ha="center", fontsize=7.5, color=DIM, transform=ax_sin.transAxes,
)
for spine in ax_sin.spines.values():
    spine.set_edgecolor("#30363d")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2b – cos waves (odd dims)
# ══════════════════════════════════════════════════════════════════════════════
ax_cos.set_facecolor("#161b22")
for idx, d in enumerate(cos_dims):
    ax_cos.plot(positions, PE[:, d], color=cmap_lines[idx], lw=1.6,
                label=f"dim {d}")
ax_cos.set_title("cos  (odd dimensions)", color=FG, fontsize=10, pad=6)
ax_cos.set_xlabel("Token position", color=DIM, fontsize=8.5)
ax_cos.set_ylabel("PE value", color=DIM, fontsize=8.5)
ax_cos.tick_params(colors=FG, labelsize=7)
ax_cos.legend(fontsize=7.5, loc="upper right", framealpha=0.15, labelcolor=FG)
ax_cos.set_ylim(-1.15, 1.45)
ax_cos.axhline(0, color="#30363d", lw=0.7, zorder=0)
ax_cos.text(
    0.5, 1.01,
    "cos starts at 1; otherwise mirrors sin behaviour",
    ha="center", fontsize=7.5, color=DIM, transform=ax_cos.transAxes,
)
for spine in ax_cos.spines.values():
    spine.set_edgecolor("#30363d")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2c – Why unique? Show PE vectors for a few positions as bar charts
# ══════════════════════════════════════════════════════════════════════════════
sample_dims = np.arange(0, 64)    # first 64 dims for clarity
sample_positions = [0, 5, 20, 60]
line_colors = ["#58a6ff", "#ffa657", "#3fb950", "#f85149"]

ax_both.set_facecolor("#161b22")
for i, pos in enumerate(sample_positions):
    ax_both.plot(sample_dims, PE[pos, :64],
                 color=line_colors[i], lw=1.4, alpha=0.9,
                 label=f"pos {pos}")

ax_both.set_title("PE vector slice  (dims 0–63, 4 positions)", color=FG, fontsize=10, pad=6)
ax_both.set_xlabel("Dimension index  (0–63)", color=DIM, fontsize=8.5)
ax_both.set_ylabel("PE value", color=DIM, fontsize=8.5)
ax_both.tick_params(colors=FG, labelsize=7)
ax_both.legend(fontsize=7.5, loc="upper right", framealpha=0.15, labelcolor=FG)
ax_both.set_ylim(-1.3, 1.5)
ax_both.axhline(0, color="#30363d", lw=0.7, zorder=0)
ax_both.text(
    0.5, 1.01,
    "Each position gets a unique 'fingerprint' vector → no two positions are identical",
    ha="center", fontsize=7.5, color=DIM, transform=ax_both.transAxes,
)
for spine in ax_both.spines.values():
    spine.set_edgecolor("#30363d")

# ── global figure title ────────────────────────────────────────────────────────
fig.suptitle(
    "Positional Encoding  |  Transformer  (Attention Is All You Need)",
    color=FG, fontsize=16, fontweight="bold", y=0.975,
)

plt.savefig("positional_encoding_explained.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved → positional_encoding_explained.png")
plt.show()
