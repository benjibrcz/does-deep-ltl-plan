"""
Varied optimality test environment.

Generates random configurations with:
- 2 zones of intermediate color
- 1 zone of goal color
- 2 distractor zones (1 each of remaining colors)

Ensures optimal != myopic by construction.
Supports configurable colors via config dict.
"""

import numpy as np
from safety_gymnasium.tasks.ltl.ltl_base_task import LtlBaseTask
from safety_gymnasium.assets.geoms import Zones


ALL_COLORS = ['blue', 'green', 'yellow', 'magenta']


class LtlOptimalityVaried(LtlBaseTask):
    """
    Varied optimality test configurations.

    Each reset generates a new random layout where:
    - One intermediate zone is closer to agent (myopic choice)
    - One intermediate zone has shorter total path (optimal choice)
    - These are guaranteed to be different

    Config options:
    - layout_seed: Starting seed for layout generation
    - intermediate_color: Color for 2 intermediate zones (default: 'blue')
    - goal_color: Color for 1 goal zone (default: 'green')
    """

    def __init__(self, config) -> None:
        # Extract our custom config keys before passing to parent
        # (parent validates that all config keys exist as attributes)
        config = config or {}

        self._layout_seed = config.pop('layout_seed', 0)
        self._intermediate_color = config.pop('intermediate_color', 'blue')
        self._goal_color = config.pop('goal_color', 'green')
        self._bounds = 1.8

        # Distractors are the remaining colors
        used_colors = {self._intermediate_color, self._goal_color}
        self._distractor_colors = [c for c in ALL_COLORS if c not in used_colors]

        super().__init__(config=config, zone_size=0.4)

        self._setup_initial_layout()

    def _setup_initial_layout(self):
        """Set up initial zone layout."""
        cfg = self._generate_layout(self._layout_seed)

        # Intermediate color (2 zones)
        self._add_geoms(Zones(
            color=self._intermediate_color,
            size=self.zone_size,
            num=2,
            locations=[cfg['int_near'], cfg['int_far']],
            keepout=0
        ))

        # Goal color (1 zone)
        self._add_geoms(Zones(
            color=self._goal_color,
            size=self.zone_size,
            num=1,
            locations=[cfg['goal']],
            keepout=0
        ))

        # Distractors (1 each of remaining colors)
        if len(self._distractor_colors) >= 1:
            self._add_geoms(Zones(
                color=self._distractor_colors[0],
                size=self.zone_size,
                num=1,
                locations=[cfg['distractor1']],
                keepout=0
            ))
        if len(self._distractor_colors) >= 2:
            self._add_geoms(Zones(
                color=self._distractor_colors[1],
                size=self.zone_size,
                num=1,
                locations=[cfg['distractor2']],
                keepout=0
            ))

        self._set_agent_location(cfg['agent'])

    def _check_min_distances(self, positions, min_dist):
        """Check that all positions are at least min_dist apart."""
        for i, p1 in enumerate(positions):
            for j, p2 in enumerate(positions):
                if i < j:
                    if np.linalg.norm(np.array(p1) - np.array(p2)) < min_dist:
                        return False
        return True

    def _generate_layout(self, seed):
        """Generate a layout where optimal != myopic with proper zone separation."""
        rng = np.random.RandomState(seed)
        bounds = 2.2  # Larger bounds for more spread
        min_zone_dist = 1.0  # Minimum distance between zone centers
        min_agent_dist = 0.6  # Minimum distance from agent to zones

        for attempt in range(200):  # Max attempts
            # Agent position (center area)
            agent = rng.uniform(-0.3, 0.3, 2)

            # Goal position (away from agent, in a random direction)
            goal_angle = rng.uniform(0, 2 * np.pi)
            goal_dist = rng.uniform(1.8, 2.4)
            goal = agent + goal_dist * np.array([np.cos(goal_angle), np.sin(goal_angle)])
            goal = np.clip(goal, -bounds, bounds)

            # Intermediate far (optimal): positioned between agent and goal
            # but offset so it's not directly on the path
            t = rng.uniform(0.5, 0.7)
            int_far_base = agent + t * (goal - agent)
            # Perpendicular offset
            perp = np.array([-(goal - agent)[1], (goal - agent)[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-6)
            int_far = int_far_base + perp * rng.uniform(-0.4, 0.4)
            int_far = np.clip(int_far, -bounds, bounds)

            # Intermediate near (myopic): close to agent but in opposite direction from goal
            near_angle = goal_angle + np.pi + rng.uniform(-0.5, 0.5)
            near_dist = rng.uniform(0.8, 1.2)
            int_near = agent + near_dist * np.array([np.cos(near_angle), np.sin(near_angle)])
            int_near = np.clip(int_near, -bounds, bounds)

            # Check optimality condition
            d_near_agent = np.linalg.norm(int_near - agent)
            d_far_agent = np.linalg.norm(int_far - agent)
            d_near_goal = np.linalg.norm(goal - int_near)
            d_far_goal = np.linalg.norm(goal - int_far)

            total_near = d_near_agent + d_near_goal
            total_far = d_far_agent + d_far_goal

            # Must have: near closer to agent, but far has shorter total path
            if not (d_near_agent < d_far_agent and total_far < total_near):
                continue
            if not (total_near - total_far) > 0.3:
                continue

            # Generate distractors with minimum distance constraints
            main_positions = [int_near, int_far, goal]

            distractor1 = None
            for _ in range(50):
                d1 = rng.uniform(-bounds, bounds, 2)
                all_pos = main_positions + [d1]
                if self._check_min_distances(all_pos, min_zone_dist):
                    if np.linalg.norm(d1 - agent) > min_agent_dist:
                        distractor1 = d1
                        break

            if distractor1 is None:
                continue

            distractor2 = None
            for _ in range(50):
                d2 = rng.uniform(-bounds, bounds, 2)
                all_pos = main_positions + [distractor1, d2]
                if self._check_min_distances(all_pos, min_zone_dist):
                    if np.linalg.norm(d2 - agent) > min_agent_dist:
                        distractor2 = d2
                        break

            if distractor2 is None:
                continue

            # Final check: all zones well separated
            all_zones = [int_near, int_far, goal, distractor1, distractor2]
            if not self._check_min_distances(all_zones, min_zone_dist):
                continue

            # Check agent is not too close to any zone
            agent_ok = all(np.linalg.norm(agent - z) > min_agent_dist for z in all_zones)
            if not agent_ok:
                continue

            # Success!
            return {
                'agent': tuple(agent),
                'int_near': tuple(int_near),
                'int_far': tuple(int_far),
                'goal': tuple(goal),
                'distractor1': tuple(distractor1),
                'distractor2': tuple(distractor2),
            }

        # Fallback: simple layout if we can't find a good one
        return {
            'agent': (0, 0),
            'int_near': (-1.2, 0.8),
            'int_far': (0.8, 1.2),
            'goal': (1.5, 1.8),
            'distractor1': (-1.5, -1.2),
            'distractor2': (1.2, -1.5),
        }

    def set_layout_seed(self, seed):
        """Update layout for next reset."""
        self._layout_seed = seed

    def specific_reset(self):
        """Called on each reset - update zone positions."""
        cfg = self._generate_layout(self._layout_seed)

        # Update zone locations based on configured colors
        for geom in self._geoms.values():
            if hasattr(geom, 'name'):
                name_lower = geom.name.lower()
                if self._intermediate_color in name_lower:
                    geom.locations = [cfg['int_near'], cfg['int_far']]
                elif self._goal_color in name_lower:
                    geom.locations = [cfg['goal']]
                elif len(self._distractor_colors) >= 1 and self._distractor_colors[0] in name_lower:
                    geom.locations = [cfg['distractor1']]
                elif len(self._distractor_colors) >= 2 and self._distractor_colors[1] in name_lower:
                    geom.locations = [cfg['distractor2']]

        self._set_agent_location(cfg['agent'])
        self._layout_seed += 1
