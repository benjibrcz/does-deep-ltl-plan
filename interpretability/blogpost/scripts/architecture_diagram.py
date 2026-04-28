"""Architecture diagram for the DeepLTL agent.

Boxes-and-arrows: env_obs → env_net → env_emb (64d); LTL_seq → LTL_net (GRU)
→ ltl_emb (32d); concat → 96-d combined → actor.enc[H1, H2, H3] (each 64d)
→ μ (2d). Saved as figures/00_architecture.png.
"""
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "figures"))


def box(ax, x, y, w, h, label, fc, ec="black", text_fontsize=10, lw=0.8):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=text_fontsize)


def arrow(ax, x0, y0, x1, y1, text=None, text_offset=(0, 0.07), color="black",
          lw=1.0):
    a = FancyArrowPatch((x0, y0), (x1, y1),
                        arrowstyle="-|>",
                        mutation_scale=12,
                        linewidth=lw,
                        color=color)
    ax.add_patch(a)
    if text:
        ax.text((x0 + x1) / 2 + text_offset[0],
                (y0 + y1) / 2 + text_offset[1],
                text, ha="center", va="bottom", fontsize=8.5,
                color="#444", style="italic")


def main():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colour palette
    INPUT = "#cfe9f7"
    NET = "#bee3bd"
    EMB = "#fff2c8"
    HID = "#fde2c4"
    OUT = "#f7d0d0"
    PLAN = "#e2d5f0"

    # -- Top row: environment branch (y = 4.3)
    y_env = 4.3
    box(ax, 0.2, y_env, 1.6, 0.9, "lidar +\nproprio.\n(80-d)", INPUT)
    box(ax, 2.4, y_env, 1.4, 0.9, "env_net\n(MLP)", NET)
    box(ax, 4.4, y_env, 1.6, 0.9, "env_emb\n(64-d)", EMB)
    arrow(ax, 1.8, y_env + 0.45, 2.4, y_env + 0.45)
    arrow(ax, 3.8, y_env + 0.45, 4.4, y_env + 0.45)

    # -- Bottom row: LTL branch (y = 1.5)
    y_ltl = 1.5
    box(ax, 0.2, y_ltl, 1.6, 0.9, "LTL formula\n→ automaton\nstates", INPUT)
    box(ax, 2.4, y_ltl, 1.4, 0.9, "LTL_net\n(DeepSets+GRU)", NET)
    box(ax, 4.4, y_ltl, 1.6, 0.9, "ltl_emb\n(32-d)", EMB)
    arrow(ax, 1.8, y_ltl + 0.45, 2.4, y_ltl + 0.45)
    arrow(ax, 3.8, y_ltl + 0.45, 4.4, y_ltl + 0.45)

    # -- Concat (centred between rows)
    y_mid = 2.95
    box(ax, 6.5, y_mid, 1.4, 0.9, "concat\n(96-d)", PLAN)
    arrow(ax, 6.0, y_env + 0.45, 6.5, y_mid + 0.7, lw=0.9)
    arrow(ax, 6.0, y_ltl + 0.45, 6.5, y_mid + 0.2, lw=0.9)

    # -- Actor MLP H1, H2, H3
    box(ax, 8.6, y_mid, 0.75, 0.9, "H1\n(64)", HID, text_fontsize=9)
    box(ax, 9.45, y_mid, 0.75, 0.9, "H2\n(64)", HID, text_fontsize=9)
    box(ax, 10.30, y_mid, 0.75, 0.9, "H3\n(64)", HID, text_fontsize=9)
    arrow(ax, 7.9, y_mid + 0.45, 8.6, y_mid + 0.45, lw=0.9)
    arrow(ax, 9.35, y_mid + 0.45, 9.45, y_mid + 0.45, lw=0.9)
    arrow(ax, 10.20, y_mid + 0.45, 10.30, y_mid + 0.45, lw=0.9)

    # -- Output
    box(ax, 11.4, y_mid, 1.1, 0.9, "μ (2-d)\naction", OUT, text_fontsize=9)
    arrow(ax, 11.05, y_mid + 0.45, 11.4, y_mid + 0.45, lw=0.9)

    # -- "IN" label *above* concat→H1 arrow (no bracket, no overlap)
    ax.text(8.25, y_mid + 0.62, "IN",
            ha="center", fontsize=9, color="#444", style="italic")

    # -- "actor.enc" label *below* the H1/H2/H3 boxes via a single bracket
    ax.annotate("", xy=(8.6, y_mid - 0.12), xytext=(11.05, y_mid - 0.12),
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#888"))
    ax.text(9.83, y_mid - 0.40, "actor.enc",
            ha="center", fontsize=9, color="#444", style="italic")

    # -- Title at top
    ax.text(6.5, 5.55, "DeepLTL agent",
            ha="center", fontsize=13, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "00_architecture.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
