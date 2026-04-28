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


def figure_06_value_delta():
    """Value function is blind to second-step difficulty when first is held fixed.

    From LONG_REPORT §3.3 Test C (fresh_baseline):
        mean ΔV (easy C)  = -0.1850
        mean ΔV (hard C)  = -0.1875
        difference         = 0.0026
    """
    labels = ["Easy second\nsub-goal", "Hard second\nsub-goal"]
    delta_v = [-0.1850, -0.1875]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, delta_v, color=[GREEN, RED],
                  edgecolor="black", linewidth=0.6, width=0.55)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(r"$\Delta V = V(s, [A, C]) - V(s, [A])$")
    ax.set_title("Marginal value of the second sub-goal, first step held constant\n"
                 r"$\Delta V$(easy) $-$ $\Delta V$(hard) $= 0.003$, within noise")

    for b, v in zip(bars, delta_v):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.006,
                f"{v:.4f}", ha="center", va="top", fontweight="bold", color="white")

    ax.set_ylim(-0.22, 0.02)
    fig.tight_layout()
    save(fig, "06_value_anticipation.png")
    plt.close(fig)


def figure_07_planning_incentives():
    """Training interventions raise probe R² without changing behaviour.

    From LONG_REPORT §4.2:
        Optimal choice (controlled-orientation): all ~50%
        Probe R² (chained distance): 0.315, 0.356, 0.361, 0.405
        Task success: 93%, 90%, 89%, 75%
    """
    variants = ["Baseline", "Aux loss\n(0.2)", "Trans loss\n(0.1)", "Combined"]
    optimal = [50, 50, 50, 50]
    probe_r2 = [0.315, 0.356, 0.361, 0.405]
    task_success = [93, 90, 89, 75]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

    ax = axes[0]
    bars = ax.bar(variants, optimal, color=RED, edgecolor="black", linewidth=0.6)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Optimal choice (%)")
    ax.set_title("Planning behaviour\n(on optvar)")
    ax.legend(loc="upper right", frameon=False)
    for b, v in zip(bars, optimal):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"~{v}%",
                ha="center", fontweight="bold")

    ax = axes[1]
    bars = ax.bar(variants, probe_r2, color=BLUE, edgecolor="black", linewidth=0.6)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel(r"Chained-distance probe R$^2$")
    ax.set_title("Representational content\n(intermediate → goal)")
    for b, v in zip(bars, probe_r2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontweight="bold")

    ax = axes[2]
    bars = ax.bar(variants, task_success, color=GREEN, edgecolor="black", linewidth=0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Task success (%)")
    ax.set_title("Task completion")
    for b, v in zip(bars, task_success):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%",
                ha="center", fontweight="bold")

    fig.suptitle("Auxiliary supervision improves the representation, "
                 "not the behaviour",
                 y=1.04, fontsize=13)
    fig.tight_layout()
    save(fig, "07_planning_incentives.png")
    plt.close(fig)


def figure_09_letter_world_direction():
    """Letter World directional bias.

    From deep-ltl/interpretability/REPORT.md §3.2 (fixed-map safety task):
        UP    65%
        DOWN  47%
        RIGHT 40%
        LEFT  12%
    Overall safe success 38%; chance among four directions is 25%.
    """
    directions = ["Up", "Down", "Right", "Left"]
    correct = [65, 47, 40, 12]

    # Colour each bar relative to chance
    colours = [GREEN, GREY, GREY, RED]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(directions, correct, color=colours,
                  edgecolor="black", linewidth=0.6, width=0.55)
    ax.axhline(25, color="gray", linestyle="--", linewidth=1, label="Chance (25%)")
    ax.set_ylim(0, 80)
    ax.set_ylabel("Correct-choice rate (%)")
    ax.set_xlabel("Direction of the unblocked goal, relative to the agent")
    ax.set_title("Letter World: success depends almost entirely on which way the goal lies")
    ax.legend(frameon=False, loc="upper right")

    for b, v in zip(bars, correct):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5,
                f"{v}%", ha="center", fontweight="bold")

    fig.tight_layout()
    save(fig, "09_letter_world_safety.png")
    plt.close(fig)


def figure_01_setup():
    """Clean setup diagram for the intro.

    Shows the task `F blue THEN F green` with two candidate blue zones, one
    myopic (closer to the agent) and one optimal (closer to the goal), plus
    two distractor zones. Dashed lines indicate the paths a planning and a
    myopic agent would each take. The figure does *not* claim the agent does
    either — it illustrates the task structure.
    """
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(7, 7))

    # Positions chosen so the two paths are visually distinct (not collinear)
    agent = (0.0, 1.4)
    myopic_blue = (1.8, 1.4)           # right of agent — very close
    optimal_blue = (-1.8, 0.1)         # far left, mid — longer first leg
    goal = (-1.5, -1.9)                # bottom left, close to optimal
    distractor_yellow = (2.2, -0.4)
    distractor_magenta = (0.2, 2.6)

    zone_radius = 0.35

    def zone(xy, colour, label=None, dashed=False, label_offset=(0, -0.6)):
        circ = mpatches.Circle(xy, zone_radius,
                               facecolor=colour,
                               edgecolor=colour,
                               alpha=0.35 if dashed else 0.9,
                               linestyle="--" if dashed else "-",
                               linewidth=1.8)
        ax.add_patch(circ)
        if label is not None:
            ax.annotate(label,
                        xy=(xy[0] + label_offset[0], xy[1] + label_offset[1]),
                        ha="center", va="top",
                        fontsize=10, fontweight="bold",
                        color="#333")

    zone(distractor_yellow, ZONE_YELLOW, dashed=True)
    zone(distractor_magenta, ZONE_MAGENTA, dashed=True)

    zone(myopic_blue, ZONE_BLUE,
         label="MYOPIC blue", label_offset=(0.55, 0.15))
    zone(optimal_blue, ZONE_BLUE,
         label="OPTIMAL blue", label_offset=(-0.7, 0.05))
    zone(goal, ZONE_GREEN,
         label="GOAL (green)", label_offset=(0.55, -0.05))

    ax.plot([agent[0], optimal_blue[0], goal[0]],
            [agent[1], optimal_blue[1], goal[1]],
            linestyle="--", linewidth=2.0, color="#2e7d32",
            label="Optimal path  (agent → optimal blue → goal)",
            zorder=3)
    ax.plot([agent[0], myopic_blue[0], goal[0]],
            [agent[1], myopic_blue[1], goal[1]],
            linestyle=":", linewidth=2.0, color="#ef6c00",
            label="Myopic path  (agent → closer blue → goal)",
            zorder=3)

    ax.plot(*agent, marker="D", markersize=14,
            markerfacecolor=AGENT, markeredgecolor="black", zorder=5)
    ax.annotate("agent", xy=(agent[0] - 0.35, agent[1] + 0.05),
                ha="right", va="center",
                fontsize=10, fontweight="bold", color="#333")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaa")
        spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title(r"Task: $F\,\mathrm{blue}$ THEN $F\,\mathrm{green}$"
                 "\nTwo candidate blue zones: the nearer one leaves a longer onward path",
                 fontsize=11)

    fig.tight_layout()
    save(fig, "01_setup_map.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_01_setup()
    figure_06_value_delta()
    figure_07_planning_incentives()
    figure_09_letter_world_direction()
