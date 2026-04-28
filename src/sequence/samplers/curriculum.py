import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Callable, Optional

import numpy as np
import torch

from ltl.automata import LDBASequence
from sequence.samplers.flatworld_sequence_samplers import flatworld_all_reach_tasks, \
    flatworld_sample_reach_avoid, flatworld_sample_reach_stay, flatworld_sample_reach
from sequence.samplers.sequence_samplers import (
    sample_reach_avoid, all_reach_avoid_tasks, all_reach_tasks,
    all_reach_stay_tasks, sample_reach_stay,
    # Planning-required samplers
    sample_disjunctive_reach, sample_global_avoid, sample_disjunctive_reach_global_avoid,
    all_disjunctive_reach_tasks, all_reach_global_avoid_tasks, all_disjunctive_reach_global_avoid_tasks,
    # Hard optimality samplers
    sample_optimality_task, all_optimality_tasks,
)


@dataclass
class CurriculumStage(ABC):
    threshold: float | None
    threshold_type: Literal['mean', 'min'] | None

    @abstractmethod
    def sample(self, propositions: list[str]) -> LDBASequence:
        pass

    @abstractmethod
    def update_task_success(self, task_success: dict[LDBASequence, float]) -> None:
        pass


@dataclass
class ExplicitCurriculumStage(CurriculumStage):
    """A curriculum stage in which all tasks are explicitly listed, and sampled from according to previous success."""
    task_fn: Optional[Callable[[list[str]], list[LDBASequence]]]
    eps_task_fn: Optional[Callable[[list[str]], list[LDBASequence]]] = None
    temperature: float = 0.5
    _tasks: list[LDBASequence] | None = None
    _task_success: dict[LDBASequence, float] | None = None

    def sample(self, propositions: list[str]) -> LDBASequence:
        if self._tasks is None:
            self._tasks = []
            if self.task_fn is not None:
                self._tasks += self.task_fn(propositions)
            if self.eps_task_fn is not None:
                self._tasks += self.eps_task_fn(propositions)
        if self._task_success is None:
            return random.choice(self._tasks)
        probs = self.compute_sampling_prob()
        index = np.random.choice(np.arange(len(self._tasks)), p=probs).item()
        return self._tasks[index]

    def compute_sampling_prob(self) -> np.ndarray:
        if len(self._task_success) != len(self._tasks):
            raise ValueError('Task success must be available for all tasks')
        success = torch.tensor([self._task_success[t] for t in self._tasks])
        probs = torch.nn.functional.softmax(-success / self.temperature, dim=0)
        return probs.numpy()

    def update_task_success(self, task_success: dict[LDBASequence, float]) -> None:
        if self._task_success is None:
            self._task_success = {k: v for k, v in task_success.items() if k in self._tasks}
            for t in self._tasks:
                if t not in self._task_success:
                    self._task_success[t] = 0.0
        else:
            self._task_success.update(task_success)


@dataclass
class RandomCurriculumStage(CurriculumStage):
    """A curriculum stage in which tasks are sampled randomly."""
    sampler: Callable[[list[str]], LDBASequence]

    def sample(self, propositions: list[str]) -> LDBASequence:
        return self.sampler(propositions)

    def update_task_success(self, task_success: dict[LDBASequence, float]) -> None:
        pass


@dataclass
class MultiRandomStage(CurriculumStage):
    """A combination of multiple RandomCurriculumStages with associated sampling probabilities."""
    stages: list[RandomCurriculumStage]
    probs: list[float]

    def sample(self, propositions: list[str]) -> LDBASequence:
        stage = np.random.choice(self.stages, p=self.probs)
        return stage.sample(propositions)

    def update_task_success(self, task_success: dict[LDBASequence, float]) -> None:
        pass


class Curriculum:
    def __init__(self, stages: list[CurriculumStage]):
        self.stages = stages
        self.stage_index = 0
        self.num_updates = 0

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    @property
    def finished(self) -> bool:
        return self.stage_index >= len(self.stages)

    def sample(self, propositions: list[str]) -> LDBASequence:
        return self.current_stage.sample(propositions)

    def update_task_success(self, task_success: dict[LDBASequence, float], verbose=False) -> None:
        if self.current_stage.threshold is None:
            return
        self.num_updates += 1
        self.num_updates %= 100
        self.current_stage.update_task_success(task_success)
        aggr = np.mean if self.current_stage.threshold_type == 'mean' else np.min
        if aggr(list(task_success.values())) >= self.current_stage.threshold:
            if verbose:
                print('=' * 80)
                print(f"Stage {self.stage_index} completed.")
                print('=' * 80)
            self.stage_index += 1
        else:
            if verbose and self.num_updates % 100 == 0:
                print(f"Stage {self.stage_index} not completed.")
                print(f'MEAN: {np.mean(list(task_success.values()))}, THRESHOLD: {self.current_stage.threshold}')


LETTER_CURRICULUM = Curriculum([
    ExplicitCurriculumStage(
        task_fn=all_reach_avoid_tasks(1),
        temperature=0.1,
        threshold=0.95,
        threshold_type='mean',
    ),
    RandomCurriculumStage(
        sampler=sample_reach_avoid(1, (1, 2), (0, 2)),
        threshold=0.95,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=sample_reach_avoid(2, (1, 2), (1, 2)),
        threshold=0.95,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=sample_reach_avoid(3, (1, 2), (0, 3)),
        threshold=None,
        threshold_type=None
    ),
])

ZONES_CURRICULUM = Curriculum([
    ExplicitCurriculumStage(  # 0
        task_fn=all_reach_tasks(1),
        temperature=0.5,
        threshold=0.8,
        threshold_type='min',
    ),
    ExplicitCurriculumStage(  # 1
        task_fn=all_reach_tasks(2),
        threshold=0.95,
        threshold_type='mean'
    ),
    ExplicitCurriculumStage(  # 2
        task_fn=all_reach_avoid_tasks(1),
        threshold=0.95,
        threshold_type='mean'
    ),
    ExplicitCurriculumStage(  # 3
        task_fn=all_reach_avoid_tasks(2),
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 4
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(1, (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(30, (0, 1)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.4, 0.6],
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 5
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(2, (1, 2), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(60, (0, 1)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 6
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(3, (1, 2), (0, 3)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(60, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=None,
        threshold_type=None
    ),
])


# =============================================================================
# ZONES TWO-STEP CURRICULUM (Author's suggestion)
# =============================================================================
# Modified curriculum that STARTS with 2-step sequences to avoid biasing
# the agent toward "go to closest zone" heuristic.
#
# Key changes from ZONES_CURRICULUM:
# - Stage 0: Start with 2-step reach sequences (not 1-step)
# - Stage 1: 2-step reach-avoid sequences (skip 1-step reach-avoid)
# - Combined with lower discount factor (0.95) to amplify return differences
#
# Rationale (from DeepLTL author):
# - Single-step training biases agent to prefer closest zone
# - High discount (0.998) makes optimal/suboptimal returns nearly equal
# - Starting with 2-step may encourage planning from the beginning

ZONES_TWOSTEP_CURRICULUM = Curriculum([
    ExplicitCurriculumStage(  # 0: Start directly with 2-step reach
        task_fn=all_reach_tasks(2),
        temperature=0.5,
        threshold=0.8,
        threshold_type='min',
    ),
    ExplicitCurriculumStage(  # 1: 2-step reach-avoid (skip 1-step)
        task_fn=all_reach_avoid_tasks(2),
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 2: Mix of 2-step and 3-step reach-avoid
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(2, (1, 2), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_avoid(3, (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 3: Complex multi-step tasks
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(3, (1, 2), (0, 3)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(60, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=None,
        threshold_type=None
    ),
])


# =============================================================================
# ZONES MIXED CURRICULUM (V2: mixed2)
# =============================================================================
# Starts with 2-step sequences but includes 10-25% of 1-step for stability.
# This avoids the "nearest-zone lock-in" while maintaining training stability.
#
# Key design:
# - Stage 0: 75% 2-step reach + 25% 1-step reach (stabilized start)
# - Stage 1: 90% 2-step reach-avoid + 10% 1-step reach-avoid
# - Later stages: progressively longer sequences

ZONES_MIXED_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0: Mixed 1-step (25%) + 2-step (75%) reach
        stages=[
            ExplicitCurriculumStage(
                task_fn=all_reach_tasks(1),
                temperature=0.5,
                threshold=None,
                threshold_type=None,
            ),
            ExplicitCurriculumStage(
                task_fn=all_reach_tasks(2),
                temperature=0.5,
                threshold=None,
                threshold_type=None,
            ),
        ],
        probs=[0.25, 0.75],
        threshold=0.8,
        threshold_type='min',
    ),
    MultiRandomStage(  # 1: Mixed 1-step (10%) + 2-step (90%) reach-avoid
        stages=[
            ExplicitCurriculumStage(
                task_fn=all_reach_avoid_tasks(1),
                temperature=0.5,
                threshold=None,
                threshold_type=None,
            ),
            ExplicitCurriculumStage(
                task_fn=all_reach_avoid_tasks(2),
                temperature=0.5,
                threshold=None,
                threshold_type=None,
            ),
        ],
        probs=[0.10, 0.90],
        threshold=0.9,
        threshold_type='mean',
    ),
    MultiRandomStage(  # 2: Mix of 2-step and 3-step reach-avoid
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(2, (1, 2), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_avoid(3, (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 3: Complex multi-step tasks
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(3, (1, 2), (0, 3)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(60, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=None,
        threshold_type=None
    ),
])


# =============================================================================
# ZONES PLANNING CURRICULUM
# =============================================================================
# Extended curriculum that adds planning-required formula types:
# - Disjunction: (F A | F B) - agent must choose between goals
# - Global safety: G !avoid - permanent avoid constraint
# - Combined: (F A | F B) & G !C - paper's Figure 1c safety test
#
# Stages 0-6: Same as ZONES_CURRICULUM (foundational skills)
# Stages 7-11: New planning-required formulas

ZONES_PLANNING_CURRICULUM = Curriculum([
    # =========================================================================
    # PHASE 1: Original curriculum (stages 0-6) - same as ZONES_CURRICULUM
    # =========================================================================
    ExplicitCurriculumStage(  # 0: Simple reach
        task_fn=all_reach_tasks(1),
        temperature=0.5,
        threshold=0.8,
        threshold_type='min',
    ),
    ExplicitCurriculumStage(  # 1: 2-step reach sequences
        task_fn=all_reach_tasks(2),
        threshold=0.95,
        threshold_type='mean'
    ),
    ExplicitCurriculumStage(  # 2: Reach-avoid (1 step)
        task_fn=all_reach_avoid_tasks(1),
        threshold=0.95,
        threshold_type='mean'
    ),
    ExplicitCurriculumStage(  # 3: Reach-avoid (2 steps)
        task_fn=all_reach_avoid_tasks(2),
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 4: Mixed reach-avoid + reach-stay
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(1, (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(30, (0, 1)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.4, 0.6],
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 5: Deeper reach-avoid + reach-stay
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(2, (1, 2), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(60, (0, 1)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 6: Complex reach-avoid + reach-stay
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(3, (1, 2), (0, 3)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(60, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=0.85,  # Need threshold to advance to planning stages
        threshold_type='mean'
    ),

    # =========================================================================
    # PHASE 2: Planning-required formulas (stages 7-11)
    # =========================================================================

    # Stage 7: Disjunctive reach - (F A | F B)
    ExplicitCurriculumStage(
        task_fn=all_disjunctive_reach_tasks(1, num_disjuncts=2),
        temperature=0.5,
        threshold=0.9,
        threshold_type='mean'
    ),

    # Stage 8: Global safety - F goal & G !avoid
    ExplicitCurriculumStage(
        task_fn=all_reach_global_avoid_tasks(1),
        temperature=0.5,
        threshold=0.9,
        threshold_type='mean'
    ),

    # Stage 9: Combined - (F A | F B) & G !C (paper's Figure 1c)
    ExplicitCurriculumStage(
        task_fn=all_disjunctive_reach_global_avoid_tasks(num_disjuncts=2),
        temperature=0.5,
        threshold=0.85,
        threshold_type='mean'
    ),

    # Stage 10: Mixed planning tasks
    MultiRandomStage(
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(2, (1, 2), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_disjunctive_reach(1, (2, 3), (0, 1)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_global_avoid(1, 1, (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_disjunctive_reach_global_avoid(1, (2, 3), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.3, 0.2, 0.2, 0.3],
        threshold=0.85,
        threshold_type='mean'
    ),

    # Stage 11: Final stage - all formula types
    MultiRandomStage(
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(3, (1, 2), (0, 3)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_disjunctive_reach((1, 2), (2, 3), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_global_avoid((1, 2), 1, (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_disjunctive_reach_global_avoid(1, (2, 3), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(60, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.2, 0.2, 0.15, 0.3, 0.15],
        threshold=None,  # Final stage
        threshold_type=None
    ),
])


FLATWORLD_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_avoid((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=0.8,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=flatworld_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
        threshold=None,
        threshold_type=None
    ),
])

FLATWORLD_BIG_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_avoid((1, 2), 1, 0),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=0.8,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=sample_reach_avoid((1, 2), (1, 2), (0, 2)),
        threshold=None,
        threshold_type=None
    ),
])


# =============================================================================
# HARD OPTIMALITY CURRICULUM
# =============================================================================
# Curriculum for training on hard optimality maps where myopic behavior fails.
# Uses fixed zone positions (via PointLtl2-v0.hardmix environment) with
# the "F blue THEN F green" task.
#
# The goal is to force the agent to learn chained distance computation:
# choosing the blue zone that minimizes total path length, not just
# the one closest to the agent.

HARD_OPTIMALITY_CURRICULUM = Curriculum([
    # Single stage: Always the optimality task on hard maps
    # The environment (hardmix) provides the challenging zone configurations
    ExplicitCurriculumStage(
        task_fn=all_optimality_tasks(),
        temperature=0.5,
        threshold=None,  # No progression, always this task
        threshold_type=None
    ),
])
