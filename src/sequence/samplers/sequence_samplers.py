import random
from pprint import pprint
from typing import Callable

from ltl.automata import LDBASequence
from ltl.logic import Assignment


def sample_reach_avoid(
        depth: int | tuple[int, int],
        num_reach: int | tuple[int, int],
        num_avoid: int | tuple[int, int],
        not_reach_same_as_last: bool = False
) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        def sample_one(last_reach: set[str]):
            nr = random.randint(*num_reach) if isinstance(num_reach, tuple) else num_reach
            na = random.randint(*num_avoid) if isinstance(num_avoid, tuple) else num_avoid
            available = [p for p in propositions if p not in last_reach] if not_reach_same_as_last else propositions
            reach = random.sample(available, nr)
            available = [p for p in propositions if p not in reach and p not in last_reach]
            if len(available) < na:
                if isinstance(num_avoid, tuple):
                    na = random.randint(num_avoid[0], len(available)) if num_avoid[0] < len(available) else len(
                        available)
                else:
                    raise ValueError('Not enough propositions to sample from')
            avoid = random.sample(available, na)
            reach_assignments = frozenset([Assignment.single_proposition(p, propositions).to_frozen() for p in reach])
            avoid_assignments = frozenset([Assignment.single_proposition(p, propositions).to_frozen() for p in avoid])
            return reach_assignments, avoid_assignments, reach

        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        last_reach = set()
        seq = []
        for _ in range(d):
            reach, avoid, reach_props = sample_one(last_reach)
            seq.append((reach, avoid))
            last_reach = reach_props
        return LDBASequence(seq)

    return wrapper


def all_reach_avoid_tasks(depth: int) -> Callable[[list[str]], list[LDBASequence]]:
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        reach_avoids = [(frozenset([Assignment.single_proposition(p, propositions).to_frozen()]),
                         frozenset([Assignment.single_proposition(q, propositions).to_frozen()]))
                        for p in propositions for q in propositions if p != q]

        def rec(depth: int):
            if depth == 0:
                return []
            if depth == 1:
                return [[ra] for ra in reach_avoids]
            rec_res = rec(depth - 1)
            result = []
            for task in rec_res:
                next_reach, next_avoid = task[0]
                for p, q in reach_avoids:
                    if p == next_reach or p == next_avoid:
                        continue
                    result.append([(p, q)] + task)
            return result

        return [LDBASequence(task) for task in rec(depth)]

    return wrapper


def all_reach_tasks(depth: int) -> Callable[[list[str]], list[LDBASequence]]:
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        reachs = [(frozenset([Assignment.single_proposition(p, propositions).to_frozen()]),
                   frozenset())
                  for p in propositions]

        def rec(depth: int):
            if depth == 0:
                return []
            if depth == 1:
                return [[r] for r in reachs]
            rec_res = rec(depth - 1)
            result = []
            for task in rec_res:
                next_reach = task[0][0]
                for p, _ in reachs:
                    if p == next_reach:
                        continue
                    result.append([(p, frozenset())] + task)
            return result

        return [LDBASequence(task) for task in rec(depth)]

    return wrapper


def all_reach_stay_tasks(num_stay: int) -> Callable[[list[str]], list[LDBASequence]]:
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        tasks = []
        for p in propositions:
            reach = frozenset([Assignment.single_proposition(p, propositions).to_frozen()])
            avoid = frozenset([
                Assignment.zero_propositions(propositions).to_frozen(),
                *[Assignment.single_proposition(q, propositions).to_frozen() for q in propositions if q != p]
            ])
            task = [(LDBASequence.EPSILON, frozenset()), (reach, avoid)]
            tasks.append(LDBASequence(task, repeat_last=num_stay))
        return tasks

    return wrapper


def sample_reach_stay(num_stay: int, num_avoid: tuple[int, int]) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        p = random.choice(propositions)
        reach = frozenset([Assignment.single_proposition(p, propositions).to_frozen()])
        na = random.randint(*num_avoid)
        available = [q for q in propositions if q != p]
        avoid = random.sample(available, na)
        avoid = frozenset([Assignment.single_proposition(q, propositions).to_frozen() for q in avoid])
        second_avoid = frozenset([
            Assignment.zero_propositions(propositions).to_frozen(),
            *[Assignment.single_proposition(q, propositions).to_frozen() for q in propositions if q != p]
        ])
        task = [(LDBASequence.EPSILON, avoid), (reach, second_avoid)]
        return LDBASequence(task, repeat_last=num_stay)

    return wrapper


def fixed(sequence: LDBASequence) -> Callable[[list[str]], Callable[[], LDBASequence]]:
    def wrapper(propositions: list[str]) -> Callable[[], LDBASequence]:
        return lambda: sequence

    return wrapper


# =============================================================================
# PLANNING-REQUIRED FORMULA SAMPLERS
# =============================================================================
# These formula types require planning/lookahead and cannot be solved with
# simple distance-based heuristics:
# - Disjunction: (F A | F B) - agent must choose between goals
# - Global safety: G !avoid - permanent avoid constraint
# - Combined: (F A | F B) & G !C - the paper's Figure 1c formula

def sample_disjunctive_reach(
        depth: int | tuple[int, int],
        num_disjuncts: int | tuple[int, int],
        num_avoid: int | tuple[int, int],
) -> Callable[[list[str]], LDBASequence]:
    """
    Sample tasks with disjunctive goals: (F A | F B) or sequences thereof.
    The agent can satisfy each step by reaching ANY of the disjunctive goals.
    """
    def wrapper(propositions: list[str]) -> LDBASequence:
        def sample_one(last_reach: set[str]):
            nd = random.randint(*num_disjuncts) if isinstance(num_disjuncts, tuple) else num_disjuncts
            na = random.randint(*num_avoid) if isinstance(num_avoid, tuple) else num_avoid

            available = [p for p in propositions if p not in last_reach]
            nd = min(nd, len(available))
            reach_props = random.sample(available, nd)

            available_avoid = [p for p in propositions if p not in reach_props and p not in last_reach]
            na = min(na, len(available_avoid))
            avoid_props = random.sample(available_avoid, na) if na > 0 else []

            reach_assignments = frozenset([
                Assignment.single_proposition(p, propositions).to_frozen()
                for p in reach_props
            ])
            avoid_assignments = frozenset([
                Assignment.single_proposition(p, propositions).to_frozen()
                for p in avoid_props
            ])
            return reach_assignments, avoid_assignments, set(reach_props)

        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        last_reach = set()
        seq = []
        for _ in range(d):
            reach, avoid, reach_props = sample_one(last_reach)
            seq.append((reach, avoid))
            last_reach = reach_props
        return LDBASequence(seq)

    return wrapper


def sample_global_avoid(
        depth: int | tuple[int, int],
        num_reach: int | tuple[int, int],
        num_global_avoid: int | tuple[int, int],
) -> Callable[[list[str]], LDBASequence]:
    """
    Sample tasks with global (permanent) avoid constraints: F goal & G !avoid
    Unlike reach-avoid (!avoid U reach), global avoid persists even after
    reaching intermediate goals.
    """
    def wrapper(propositions: list[str]) -> LDBASequence:
        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        nr = random.randint(*num_reach) if isinstance(num_reach, tuple) else num_reach
        nga = random.randint(*num_global_avoid) if isinstance(num_global_avoid, tuple) else num_global_avoid

        nga = min(nga, len(propositions) - 1)
        global_avoid_props = random.sample(propositions, nga)
        global_avoid = frozenset([
            Assignment.single_proposition(p, propositions).to_frozen()
            for p in global_avoid_props
        ])

        available = [p for p in propositions if p not in global_avoid_props]
        seq = []
        last_reach = set()

        for _ in range(d):
            reach_available = [p for p in available if p not in last_reach]
            if len(reach_available) == 0:
                reach_available = available
            nr_step = min(nr, len(reach_available))
            reach_props = random.sample(reach_available, nr_step)

            reach = frozenset([
                Assignment.single_proposition(p, propositions).to_frozen()
                for p in reach_props
            ])
            seq.append((reach, global_avoid))
            last_reach = set(reach_props)

        return LDBASequence(seq, repeat_last=100)

    return wrapper


def sample_disjunctive_reach_global_avoid(
        depth: int | tuple[int, int],
        num_disjuncts: int | tuple[int, int],
        num_global_avoid: int | tuple[int, int],
) -> Callable[[list[str]], LDBASequence]:
    """
    Sample tasks combining disjunction AND global safety: (F A | F B) & G !C
    This is the most planning-demanding formula type (paper's Figure 1c).
    """
    def wrapper(propositions: list[str]) -> LDBASequence:
        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        nd = random.randint(*num_disjuncts) if isinstance(num_disjuncts, tuple) else num_disjuncts
        nga = random.randint(*num_global_avoid) if isinstance(num_global_avoid, tuple) else num_global_avoid

        nga = min(nga, len(propositions) - 2)
        global_avoid_props = random.sample(propositions, nga)
        global_avoid = frozenset([
            Assignment.single_proposition(p, propositions).to_frozen()
            for p in global_avoid_props
        ])

        available = [p for p in propositions if p not in global_avoid_props]
        seq = []
        last_reach = set()

        for _ in range(d):
            reach_available = [p for p in available if p not in last_reach]
            if len(reach_available) < 2:
                reach_available = available
            nd_step = min(nd, len(reach_available))
            reach_props = random.sample(reach_available, nd_step)

            reach = frozenset([
                Assignment.single_proposition(p, propositions).to_frozen()
                for p in reach_props
            ])
            seq.append((reach, global_avoid))
            last_reach = set(reach_props)

        return LDBASequence(seq, repeat_last=100)

    return wrapper


def all_disjunctive_reach_tasks(depth: int, num_disjuncts: int = 2) -> Callable[[list[str]], list[LDBASequence]]:
    """Generate all possible disjunctive reach tasks."""
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        from itertools import combinations

        disjunctive_options = list(combinations(propositions, num_disjuncts))
        reach_options = []
        for props in disjunctive_options:
            reach = frozenset([
                Assignment.single_proposition(p, propositions).to_frozen()
                for p in props
            ])
            reach_options.append((reach, frozenset()))

        def rec(d: int):
            if d == 0:
                return []
            if d == 1:
                return [[opt] for opt in reach_options]
            rec_res = rec(d - 1)
            result = []
            for task in rec_res:
                prev_reach = task[0][0]
                for reach, avoid in reach_options:
                    if reach == prev_reach:
                        continue
                    result.append([(reach, avoid)] + task)
            return result

        return [LDBASequence(task) for task in rec(depth)]

    return wrapper


def all_reach_global_avoid_tasks(depth: int) -> Callable[[list[str]], list[LDBASequence]]:
    """Generate all reach + global avoid tasks: F goal & G !avoid"""
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        tasks = []
        for reach_prop in propositions:
            for avoid_prop in propositions:
                if reach_prop == avoid_prop:
                    continue

                reach = frozenset([Assignment.single_proposition(reach_prop, propositions).to_frozen()])
                avoid = frozenset([Assignment.single_proposition(avoid_prop, propositions).to_frozen()])

                if depth == 1:
                    task = [(reach, avoid)]
                else:
                    available = [p for p in propositions if p != avoid_prop and p != reach_prop]
                    task = [(reach, avoid)]
                    for _ in range(depth - 1):
                        if available:
                            next_reach_prop = random.choice(available)
                            next_reach = frozenset([
                                Assignment.single_proposition(next_reach_prop, propositions).to_frozen()
                            ])
                            task.append((next_reach, avoid))

                tasks.append(LDBASequence(task, repeat_last=100))

        return tasks

    return wrapper


def all_disjunctive_reach_global_avoid_tasks(
        num_disjuncts: int = 2
) -> Callable[[list[str]], list[LDBASequence]]:
    """Generate all (F A | F B) & G !C tasks - the key planning formula."""
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        from itertools import combinations

        tasks = []
        for goal_combo in combinations(propositions, num_disjuncts):
            for avoid_prop in propositions:
                if avoid_prop in goal_combo:
                    continue

                reach = frozenset([
                    Assignment.single_proposition(p, propositions).to_frozen()
                    for p in goal_combo
                ])
                avoid = frozenset([Assignment.single_proposition(avoid_prop, propositions).to_frozen()])

                task = [(reach, avoid)]
                tasks.append(LDBASequence(task, repeat_last=100))

        return tasks

    return wrapper


# =============================================================================
# HARD OPTIMALITY SAMPLERS
# =============================================================================
# These samplers create "F color1 THEN F color2" tasks for hard optimality training.
# The task requires choosing between two zones of color1 to minimize total path length.

def sample_optimality_task() -> Callable[[list[str]], LDBASequence]:
    """
    Create a random optimality task: F color1 THEN F color2 (sequential).
    Agent must reach any zone of color1 first, then reach color2.

    Randomly selects two different colors from available propositions.
    This tests whether the agent chooses the optimal intermediate zone
    (the one closer to final goal) vs the myopic choice (closer to agent).
    """
    def wrapper(propositions: list[str]) -> LDBASequence:
        # Randomly select two different colors
        colors = random.sample(propositions, 2)
        intermediate_color, final_color = colors

        # Step 1: Reach intermediate color (any zone of that color)
        intermediate_reach = frozenset([
            Assignment.single_proposition(intermediate_color, propositions).to_frozen()
        ])

        # Step 2: Reach final color
        final_reach = frozenset([
            Assignment.single_proposition(final_color, propositions).to_frozen()
        ])

        # No avoid constraints
        no_avoid = frozenset()

        # Sequential: intermediate THEN final
        return LDBASequence([(intermediate_reach, no_avoid), (final_reach, no_avoid)])

    return wrapper


def all_optimality_tasks() -> Callable[[list[str]], list[LDBASequence]]:
    """
    Generate all possible optimality tasks (all color pair permutations).
    For ExplicitCurriculumStage - samples based on success rate.
    """
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        from itertools import permutations

        tasks = []
        # Generate all ordered pairs of colors
        for color1, color2 in permutations(propositions, 2):
            reach1 = frozenset([Assignment.single_proposition(color1, propositions).to_frozen()])
            reach2 = frozenset([Assignment.single_proposition(color2, propositions).to_frozen()])
            no_avoid = frozenset()
            task = LDBASequence([(reach1, no_avoid), (reach2, no_avoid)])
            tasks.append(task)

        return tasks

    return wrapper


if __name__ == '__main__':
    print(sample_reach_stay(100, (0, 2))(['a', 'b', 'c']))
