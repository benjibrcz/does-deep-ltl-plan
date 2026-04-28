from typing import Any, SupportsFloat, Callable

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from ltl.automata import LDBASequence
from ltl.logic import Assignment


class SequenceWrapper(gymnasium.Wrapper):
    """
    Wrapper that adds a reach-avoid sequence of propositions to the observation space.

    Args:
        env: The base environment
        sample_sequence: Callable that returns an LDBASequence task
        partial_reward: If True, give intermediate rewards for reaching subgoals
        step_penalty: Penalty subtracted from reward each step (incentivizes shorter paths)
    """

    def __init__(self, env: gymnasium.Env, sample_sequence: Callable[[], LDBASequence],
                 partial_reward=False, step_penalty=0.0):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            'features': env.observation_space,
        })
        self.sample_sequence = sample_sequence
        self.goal_seq = None
        self.num_reached = 0
        self.propositions = set(env.get_propositions())
        self.partial_reward = partial_reward
        self.step_penalty = step_penalty
        self.obs = None
        self.info = None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if (action == LDBASequence.EPSILON).all():
            obs, _, terminated, truncated, info = self.apply_epsilon_action()
            reward = 0.
        else:
            assert not (action == LDBASequence.EPSILON).any()
            obs, reward, terminated, truncated, info = super().step(action)
        reach, avoid = self.goal_seq[self.num_reached]
        active_props = info['propositions']
        assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()
        if assignment in avoid:
            reward = -1.
            info['violation'] = True
            terminated = True
        elif reach != LDBASequence.EPSILON and assignment in reach:
            self.num_reached += 1
            terminated = self.num_reached >= len(self.goal_seq)
            if terminated:
                info['success'] = True
            if self.partial_reward:
                reward = 1. if terminated else 1 / (len(self.goal_seq) - self.num_reached + 1)
            else:
                reward = 1. if terminated else 0
        # Apply step penalty to incentivize shorter paths
        reward -= self.step_penalty
        self.obs = obs
        self.info = info
        # Compute chained distance for auxiliary prediction loss
        info['optimal_chained_distance'] = self.compute_optimal_chained_distance(info)
        obs = self.complete_observation(obs, info)
        return obs, reward, terminated, truncated, info

    def apply_epsilon_action(self):
        assert self.goal_seq[self.num_reached][0] == LDBASequence.EPSILON
        self.num_reached += 1
        return self.obs, 0.0, False, False, self.info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal_seq = self.sample_sequence()
        self.num_reached = 0
        obs = self.complete_observation(obs, info)
        self.obs = obs
        self.info = info
        return obs, info

    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any] = None) -> WrapperObsType:
        return {
            'features': obs,
            'goal': self.goal_seq[self.num_reached:],
            'initial_goal': self.goal_seq,
            'propositions': info['propositions'],
        }

    def compute_optimal_chained_distance(self, info: dict[str, Any]) -> float:
        """
        Compute the optimal chained distance for the remaining goals.

        For a sequence like F color1 THEN F color2:
        - Returns min over all color1 zones of: d(agent→color1) + d(color1→nearest_color2)

        Returns 0.0 if zone_positions or agent_pos not available in info.
        """
        if 'zone_positions' not in info or 'agent_pos' not in info:
            return 0.0

        remaining_goals = self.goal_seq[self.num_reached:]
        if len(remaining_goals) < 1:
            return 0.0

        agent_pos = np.array(info['agent_pos'][:2])
        zone_positions = info['zone_positions']

        # Extract the colors from the reach sets
        goal_colors = []
        for reach, avoid in remaining_goals:
            if reach == LDBASequence.EPSILON:
                continue
            # reach is a frozenset of FrozenAssignment objects
            # Each FrozenAssignment has get_true_propositions() method
            for assignment in reach:
                if hasattr(assignment, 'get_true_propositions'):
                    true_props = assignment.get_true_propositions()
                    for prop in true_props:
                        if prop not in goal_colors:
                            goal_colors.append(prop)
                            break
                    break

        if len(goal_colors) == 0:
            return 0.0

        # Get positions for first goal color
        first_color = goal_colors[0]
        first_positions = [
            pos for name, pos in zone_positions.items()
            if name.startswith(first_color)
        ]

        if len(first_positions) == 0:
            return 0.0

        if len(goal_colors) == 1:
            # Only one goal - just return distance to nearest zone
            distances = [np.linalg.norm(agent_pos - pos) for pos in first_positions]
            return float(min(distances))

        # Two or more goals - compute chained distance
        second_color = goal_colors[1]
        second_positions = [
            pos for name, pos in zone_positions.items()
            if name.startswith(second_color)
        ]

        if len(second_positions) == 0:
            distances = [np.linalg.norm(agent_pos - pos) for pos in first_positions]
            return float(min(distances))

        # Compute optimal chained distance (min over all first-color zones)
        best_total = float('inf')
        for first_pos in first_positions:
            d_to_first = np.linalg.norm(agent_pos - first_pos)
            # Distance from this first-color zone to nearest second-color zone
            d_to_second = min(np.linalg.norm(first_pos - s) for s in second_positions)
            total = d_to_first + d_to_second
            best_total = min(best_total, total)

        return float(best_total)
