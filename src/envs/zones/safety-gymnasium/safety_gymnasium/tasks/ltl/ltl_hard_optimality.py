# Copyright 2025. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================
"""Hard optimality task configurations for training planning capabilities.

These configurations are designed to make myopic behavior fail badly,
forcing the agent to learn chained distance computation.

Each configuration has:
- 4 colors (green, yellow, blue, magenta) with 2 zones each
- Fixed positions that create "optimality traps" for certain color pairs
- The task sampler randomly selects which colors to use for intermediate/final goals
"""
import numpy as np
from safety_gymnasium.assets.geoms import Zones
from safety_gymnasium.tasks.ltl.ltl_base_task import LtlBaseTask


class LtlHardOptimality1(LtlBaseTask):
    """Extreme opposite: For each color pair, one zone is opposite from the goal.

    Layout designed so that for ANY choice of intermediate/final colors,
    one intermediate zone is closer to agent but farther from goal.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config, zone_size=0.4)

        # Agent at center
        self._set_agent_location((0.0, 0.0))

        # Each color has 2 zones: one in +x direction, one in -x direction
        # Goals are placed at far ends so "closer to agent" != "closer to goal"

        # Green: zones at opposite ends of y-axis
        self._add_geoms(Zones(color='green', size=self.zone_size, num=2,
                             locations=[(0.0, 1.2), (0.0, -1.2)], keepout=0))

        # Yellow: zones at opposite ends of x-axis
        self._add_geoms(Zones(color='yellow', size=self.zone_size, num=2,
                             locations=[(1.2, 0.0), (-1.2, 0.0)], keepout=0))

        # Blue: diagonal positions
        self._add_geoms(Zones(color='blue', size=self.zone_size, num=2,
                             locations=[(1.0, 1.0), (-1.0, -1.0)], keepout=0))

        # Magenta: opposite diagonal
        self._add_geoms(Zones(color='magenta', size=self.zone_size, num=2,
                             locations=[(-1.0, 1.0), (1.0, -1.0)], keepout=0))


class LtlHardOptimality2(LtlBaseTask):
    """Tempting trap: One zone of each color is very close to agent.

    Creates "temptation" - the close zone is always suboptimal for reaching
    zones of other colors.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config, zone_size=0.4)

        self._set_agent_location((0.0, 0.0))

        # Each color has one zone very close (0.5-0.6 away) and one far (1.5-2.0 away)
        # The far zones are clustered, so going to close zone = long path to any other color

        # Green: close in +x, far in -x,-y corner
        self._add_geoms(Zones(color='green', size=self.zone_size, num=2,
                             locations=[(0.5, 0.1), (-1.8, -1.8)], keepout=0))

        # Yellow: close in +y, far in -x,-y area
        self._add_geoms(Zones(color='yellow', size=self.zone_size, num=2,
                             locations=[(0.1, 0.5), (-1.5, -2.0)], keepout=0))

        # Blue: close in -x, far in -x,-y area
        self._add_geoms(Zones(color='blue', size=self.zone_size, num=2,
                             locations=[(-0.5, 0.1), (-2.0, -1.5)], keepout=0))

        # Magenta: close in -y, far in -x,-y area
        self._add_geoms(Zones(color='magenta', size=self.zone_size, num=2,
                             locations=[(0.1, -0.5), (-1.8, -1.5)], keepout=0))


class LtlHardOptimality3(LtlBaseTask):
    """Asymmetric layout: Creates clear optimal/suboptimal choices.

    One quadrant has all the "far" zones clustered together.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config, zone_size=0.4)

        self._set_agent_location((0.5, 0.5))

        # Near zones spread around agent, far zones clustered in -x,-y quadrant

        self._add_geoms(Zones(color='green', size=self.zone_size, num=2,
                             locations=[(1.2, 0.8), (-1.5, -1.2)], keepout=0))

        self._add_geoms(Zones(color='yellow', size=self.zone_size, num=2,
                             locations=[(0.8, 1.2), (-1.2, -1.5)], keepout=0))

        self._add_geoms(Zones(color='blue', size=self.zone_size, num=2,
                             locations=[(1.0, -0.2), (-1.8, -1.0)], keepout=0))

        self._add_geoms(Zones(color='magenta', size=self.zone_size, num=2,
                             locations=[(-0.2, 1.0), (-1.0, -1.8)], keepout=0))


class LtlHardOptimality4(LtlBaseTask):
    """Radial layout: Zones at different distances from center.

    Inner ring (closer to agent) vs outer ring (closer to other colors).
    """

    def __init__(self, config) -> None:
        super().__init__(config=config, zone_size=0.4)

        self._set_agent_location((0.0, 0.0))

        # Inner ring at radius ~0.8, outer ring at radius ~2.0
        # Outer ring zones are closer to each other

        import math
        r_inner = 0.8
        r_outer = 2.0

        # Green at 0° and 180°
        self._add_geoms(Zones(color='green', size=self.zone_size, num=2,
                             locations=[(r_inner, 0.0), (-r_outer, 0.0)], keepout=0))

        # Yellow at 90° and 270°
        self._add_geoms(Zones(color='yellow', size=self.zone_size, num=2,
                             locations=[(0.0, r_inner), (0.0, -r_outer)], keepout=0))

        # Blue at 45° and 225°
        d_inner = r_inner * math.sqrt(2) / 2
        d_outer = r_outer * math.sqrt(2) / 2
        self._add_geoms(Zones(color='blue', size=self.zone_size, num=2,
                             locations=[(d_inner, d_inner), (-d_outer, -d_outer)], keepout=0))

        # Magenta at 135° and 315°
        self._add_geoms(Zones(color='magenta', size=self.zone_size, num=2,
                             locations=[(-d_inner, d_inner), (d_outer, -d_outer)], keepout=0))


class LtlHardOptimalityMixed(LtlBaseTask):
    """Randomly selects one of the hard configurations each reset.

    This provides variety during training while ensuring all episodes
    require optimal planning to succeed efficiently.
    """

    # Configuration data for each layout
    CONFIGS = [
        # Config 1: Extreme opposite
        {
            'agent': (0.0, 0.0),
            'green': [(0.0, 1.2), (0.0, -1.2)],
            'yellow': [(1.2, 0.0), (-1.2, 0.0)],
            'blue': [(1.0, 1.0), (-1.0, -1.0)],
            'magenta': [(-1.0, 1.0), (1.0, -1.0)],
        },
        # Config 2: Tempting trap
        {
            'agent': (0.0, 0.0),
            'green': [(0.5, 0.1), (-1.8, -1.8)],
            'yellow': [(0.1, 0.5), (-1.5, -2.0)],
            'blue': [(-0.5, 0.1), (-2.0, -1.5)],
            'magenta': [(0.1, -0.5), (-1.8, -1.5)],
        },
        # Config 3: Asymmetric
        {
            'agent': (0.5, 0.5),
            'green': [(1.2, 0.8), (-1.5, -1.2)],
            'yellow': [(0.8, 1.2), (-1.2, -1.5)],
            'blue': [(1.0, -0.2), (-1.8, -1.0)],
            'magenta': [(-0.2, 1.0), (-1.0, -1.8)],
        },
        # Config 4: Radial
        {
            'agent': (0.0, 0.0),
            'green': [(0.8, 0.0), (-2.0, 0.0)],
            'yellow': [(0.0, 0.8), (0.0, -2.0)],
            'blue': [(0.566, 0.566), (-1.414, -1.414)],
            'magenta': [(-0.566, 0.566), (1.414, -1.414)],
        },
    ]

    def __init__(self, config) -> None:
        super().__init__(config=config, zone_size=0.4)

        # Pick a random configuration
        idx = np.random.randint(len(self.CONFIGS))
        cfg = self.CONFIGS[idx]

        self._set_agent_location(cfg['agent'])

        for color in ['green', 'yellow', 'blue', 'magenta']:
            self._add_geoms(Zones(color=color, size=self.zone_size, num=2,
                                 locations=cfg[color], keepout=0))
