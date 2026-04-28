"""Regenerate figures 06, 07, 09 for the blogpost.

Numbers are taken from interpretability/LONG_REPORT.md (fresh repo) and
interpretability/REPORT.md (sibling goal-representation repo). Running this
script overwrites the corresponding files in ../figures.
"""

import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "figures"))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

GREEN = "#27ae60"
RED = "#e74c3c"
BLUE = "#2980b9"
GREY = "#95a5a6"

ZONE_BLUE = "#2196f3"
ZONE_GREEN = "#4caf50"
ZONE_YELLOW = "#fdd835"
ZONE_MAGENTA = "violet"
AGENT = "#ff9800"


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


def figure_01_setup():
    """Setup diagram for §1, using the repo's own zone-visualisation conventions.

    Same `draw_zones` / `draw_diamond` / `setup_axis` / `FancyAxes` styling as
    the optvar trajectory panels in figure 02, so the two figures look like
    they came from the same set of tools.
    """
    import sys as _sys
    _sys.path.insert(0, "src")
    from matplotlib import projections
    from visualize.zones import draw_zones, draw_diamond, setup_axis, FancyAxes
    projections.register_projection(FancyAxes)

    # Positions in the same coord system as fig 02 (xlim/ylim = (-3, 3))
    agent = (0.0, 1.4)
    myopic_blue = (1.7, 1.4)
    optimal_blue = (-1.8, 0.2)
    goal = (-1.4, -1.8)
    distractor_yellow = (2.1, -0.5)
    distractor_magenta = (0.4, 2.4)

    zones = {
        "blue_myopic":     myopic_blue,
        "blue_optimal":    optimal_blue,
        "green_goal":      goal,
        "yellow_distract": distractor_yellow,
        "magenta_distract": distractor_magenta,
    }

    fig = plt.figure(figsize=(6.2, 6.6))
    ax = fig.add_subplot(1, 1, 1, projection="fancy_box_axes",
                         edgecolor="gray", linewidth=0.5)
    setup_axis(ax)

    draw_zones(ax, zones)
    draw_diamond(ax, agent, color="orange")

    # Two annotated paths as polylines (not dashed grid lines, not the agent's
    # actual trajectory — the figure illustrates *task structure*).
    ax.plot([agent[0], optimal_blue[0], goal[0]],
            [agent[1], optimal_blue[1], goal[1]],
            linestyle="--", linewidth=2.2, color="#2e7d32",
            label="Optimal path", zorder=3)
    ax.plot([agent[0], myopic_blue[0], goal[0]],
            [agent[1], myopic_blue[1], goal[1]],
            linestyle=":", linewidth=2.2, color="#ef6c00",
            label="Myopic path", zorder=3)

    # Light text labels next to relevant zones
    ax.annotate("MYOPIC blue", xy=(myopic_blue[0] + 0.55, myopic_blue[1] + 0.15),
                fontsize=9, fontweight="bold", color="#333")
    ax.annotate("OPTIMAL blue", xy=(optimal_blue[0] - 0.6, optimal_blue[1] + 0.55),
                fontsize=9, fontweight="bold", color="#333")
    ax.annotate("GOAL", xy=(goal[0] + 0.5, goal[1] - 0.05),
                fontsize=9, fontweight="bold", color="#333")
    ax.annotate("agent", xy=(agent[0] - 0.3, agent[1] + 0.05),
                ha="right", va="center", fontsize=9, fontweight="bold",
                color="#333")

    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title(r"Task: $F\,\mathrm{blue}$ THEN $F\,\mathrm{green}$",
                 fontsize=11, pad=6)
    fig.tight_layout()
    save(fig, "01_setup_map.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_01_setup()
