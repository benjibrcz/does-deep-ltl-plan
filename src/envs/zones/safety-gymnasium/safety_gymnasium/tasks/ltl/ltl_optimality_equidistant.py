"""
Equidistant optimality test environment.

Generates random configurations with:
- 2 zones of intermediate color at EQUAL distance from agent
- 1 zone of goal color
- 2 distractor zones (1 each of remaining colors)

Both intermediates are equidistant from the agent, so the only way to
choose optimally is to consider the full path (agent -> intermediate -> goal).
"""

import numpy as np
from safety_gymnasium.tasks.ltl.ltl_base_task import LtlBaseTask
from safety_gymnasium.assets.geoms import Zones


ALL_COLORS = ['blue', 'green', 'yellow', 'magenta']


class LtlOptimalityEquidistant(LtlBaseTask):
    """
    Equidistant optimality test configurations.

    Each reset generates a new random layout where:
    - Both intermediate zones are at the SAME distance from the agent
    - One intermediate zone has shorter total path (optimal choice)
    - The only way to choose correctly is to consider the full path

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
            locations=[cfg['int_optimal'], cfg['int_suboptimal']],
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
        """Generate a layout where both intermediates are equidistant from agent."""
        rng = np.random.RandomState(seed)
        bounds = 2.2
        min_zone_dist = 1.0
        min_agent_dist = 0.6
        equidistant_tolerance = 0.05  # How close the distances need to be

        for attempt in range(500):  # More attempts needed for equidistant constraint
            # Agent position (center area)
            agent = rng.uniform(-0.3, 0.3, 2)

            # Goal position (away from agent)
            goal_angle = rng.uniform(0, 2 * np.pi)
            goal_dist = rng.uniform(1.8, 2.4)
            goal = agent + goal_dist * np.array([np.cos(goal_angle), np.sin(goal_angle)])
            goal = np.clip(goal, -bounds, bounds)

            # Both intermediates at the same distance from agent
            # but at different angles
            int_dist = rng.uniform(1.0, 1.5)  # Same distance for both

            # Optimal intermediate: closer to goal direction
            # Place it roughly toward the goal but offset
            optimal_angle = goal_angle + rng.uniform(-0.6, 0.6)
            int_optimal = agent + int_dist * np.array([np.cos(optimal_angle), np.sin(optimal_angle)])
            int_optimal = np.clip(int_optimal, -bounds, bounds)

            # Suboptimal intermediate: away from goal direction
            # Place it roughly opposite to goal
            suboptimal_angle = goal_angle + np.pi + rng.uniform(-0.8, 0.8)
            int_suboptimal = agent + int_dist * np.array([np.cos(suboptimal_angle), np.sin(suboptimal_angle)])
            int_suboptimal = np.clip(int_suboptimal, -bounds, bounds)

            # Verify equidistant constraint (clipping may have changed distances)
            d_optimal_agent = np.linalg.norm(int_optimal - agent)
            d_suboptimal_agent = np.linalg.norm(int_suboptimal - agent)

            if abs(d_optimal_agent - d_suboptimal_agent) > equidistant_tolerance:
                continue

            # Calculate total paths
            d_optimal_goal = np.linalg.norm(goal - int_optimal)
            d_suboptimal_goal = np.linalg.norm(goal - int_suboptimal)

            total_optimal = d_optimal_agent + d_optimal_goal
            total_suboptimal = d_suboptimal_agent + d_suboptimal_goal

            # Verify optimal is actually better (with margin)
            if not (total_optimal < total_suboptimal - 0.3):
                continue

            # Check zone separation
            main_positions = [int_optimal, int_suboptimal, goal]
            if not self._check_min_distances(main_positions, min_zone_dist):
                continue

            # Generate distractors
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

            # Final check
            all_zones = [int_optimal, int_suboptimal, goal, distractor1, distractor2]
            if not self._check_min_distances(all_zones, min_zone_dist):
                continue

            agent_ok = all(np.linalg.norm(agent - z) > min_agent_dist for z in all_zones)
            if not agent_ok:
                continue

            # Success!
            return {
                'agent': tuple(agent),
                'int_optimal': tuple(int_optimal),
                'int_suboptimal': tuple(int_suboptimal),
                'goal': tuple(goal),
                'distractor1': tuple(distractor1),
                'distractor2': tuple(distractor2),
            }

        # Fallback: manually construct equidistant layout
        # Agent at center, both intermediates at distance 1.2
        # Goal far away in +x direction
        return {
            'agent': (0, 0),
            'int_optimal': (1.0, 0.66),      # Toward goal
            'int_suboptimal': (-1.0, 0.66),  # Away from goal
            'goal': (2.0, 0.5),
            'distractor1': (-1.5, -1.2),
            'distractor2': (0.5, -1.5),
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
                    geom.locations = [cfg['int_optimal'], cfg['int_suboptimal']]
                elif self._goal_color in name_lower:
                    geom.locations = [cfg['goal']]
                elif len(self._distractor_colors) >= 1 and self._distractor_colors[0] in name_lower:
                    geom.locations = [cfg['distractor1']]
                elif len(self._distractor_colors) >= 2 and self._distractor_colors[1] in name_lower:
                    geom.locations = [cfg['distractor2']]

        self._set_agent_location(cfg['agent'])
        self._layout_seed += 1
