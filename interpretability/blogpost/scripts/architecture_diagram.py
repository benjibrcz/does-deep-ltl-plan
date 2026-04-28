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
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colour palette
    INPUT = "#cfe9f7"
    NET = "#bee3bd"
    EMB = "#fff2c8"
    HID = "#fde2c4"
    OUT = "#f7d0d0"
    PLAN = "#e2d5f0"

    # -- Top row: environment branch
    box(ax, 0.2, 3.5, 1.6, 0.8, "lidar +\nproprio.\n(80-d)", INPUT)
    box(ax, 2.4, 3.5, 1.4, 0.8, "env_net\n(MLP)", NET)
    box(ax, 4.4, 3.5, 1.6, 0.8, "env_emb\n(64-d)", EMB)
    arrow(ax, 1.8, 3.9, 2.4, 3.9)
    arrow(ax, 3.8, 3.9, 4.4, 3.9)

    # -- Bottom row: LTL branch
    box(ax, 0.2, 1.7, 1.6, 0.8, "LTL formula\n→ automaton\nstates", INPUT)
    box(ax, 2.4, 1.7, 1.4, 0.8, "LTL_net\n(DeepSets+GRU)", NET)
    box(ax, 4.4, 1.7, 1.6, 0.8, "ltl_emb\n(32-d)", EMB)
    arrow(ax, 1.8, 2.1, 2.4, 2.1)
    arrow(ax, 3.8, 2.1, 4.4, 2.1)

    # planner brief annotation
    ax.text(3.1, 1.45, "(includes search over plans)",
            ha="center", fontsize=8, color="#666", style="italic")

    # -- Concat
    box(ax, 6.4, 2.6, 1.4, 0.8, "concat\n(96-d)", PLAN)
    arrow(ax, 6.0, 3.9, 6.4, 3.2, lw=0.9)
    arrow(ax, 6.0, 2.1, 6.4, 2.85, lw=0.9)

    # -- Actor MLP H1, H2, H3
    box(ax, 8.2, 2.6, 0.7, 0.8, "H1\n64", HID, text_fontsize=9)
    box(ax, 9.0, 2.6, 0.7, 0.8, "H2\n64", HID, text_fontsize=9)
    box(ax, 9.8, 2.6, 0.7, 0.8, "H3\n64", HID, text_fontsize=9)
    arrow(ax, 7.8, 3.0, 8.2, 3.0, lw=0.9)
    arrow(ax, 8.9, 3.0, 9.0, 3.0, lw=0.9)
    arrow(ax, 9.7, 3.0, 9.8, 3.0, lw=0.9)
    # bracket label "actor.enc"
    ax.annotate("", xy=(8.2, 2.45), xytext=(10.5, 2.45),
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#666"))
    ax.text(9.35, 2.25, "actor.enc (H1 → H2 → H3)",
            ha="center", fontsize=8.5, color="#444", style="italic")

    # -- Output
    box(ax, 10.8, 2.6, 1.0, 0.8, "μ (2-d)\naction", OUT, text_fontsize=9)
    arrow(ax, 10.5, 3.0, 10.8, 3.0, lw=0.9)

    # -- IN label
    ax.annotate("", xy=(7.0, 2.45), xytext=(7.6, 2.45),
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#666"))
    ax.text(7.3, 2.25, "IN", ha="center", fontsize=8.5, color="#444",
            style="italic")

    # -- Title
    ax.text(6.0, 4.7, "DeepLTL agent",
            ha="center", fontsize=12, fontweight="bold")

    # -- Footer note
    ax.text(6.0, 0.85,
            "The 'planner output' (`ltl_emb`) jumps at sub-goal switches; "
            "the actor concatenates it with `env_emb` and unrolls a 3-layer MLP "
            "to produce the action.",
            ha="center", fontsize=9, color="#444", wrap=True)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "00_architecture.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
