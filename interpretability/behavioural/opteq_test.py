#!/usr/bin/env python3
"""
Equidistant Optimality Test.

Tests whether the agent can choose the optimal path when both intermediate
zones are at the SAME distance from the agent. The only way to choose
correctly is to consider the full path (agent -> intermediate -> goal).
"""

import os
import sys
sys.path.insert(0, 'src')

import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import projections
from tqdm import tqdm

import preprocessing
from model.model import build_model
from config import model_configs
from utils.model_store import ModelStore
from sequence.samplers import CurriculumSampler, curricula
from ltl.automata import LDBASequence
from ltl.logic import Assignment

# Direct imports for creating custom env
import safety_gymnasium
from gymnasium.wrappers import FlattenObservation, TimeLimit
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from envs.seq_wrapper import SequenceWrapper
from envs.remove_trunc_wrapper import RemoveTruncWrapper


COLORS = ['blue', 'green', 'yellow', 'magenta']

# All possible color pairs (intermediate, goal)
COLOR_PAIRS = [
    (a, b) for a in COLORS for b in COLORS if a != b
]

# Import built-in visualization
from visualize.zones import draw_zones, draw_path, draw_diamond, setup_axis, FancyAxes


class Agent:
    def __init__(self, model, propositions):
        self.model = model
        self.propositions = propositions

    def get_action(self, obs, deterministic=True):
        if not isinstance(obs, list):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs, self.propositions)
        with torch.no_grad():
            dist, value = self.model(preprocessed)
            action = dist.mode if deterministic else dist.sample()
        return action.detach().numpy().flatten()


def get_unwrapped(env):
    """Get the base environment."""
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    return unwrapped


def create_opteq_env(int_color, goal_color, layout_seed, propositions):
    """Create an equidistant environment with specified colors and layout seed."""
    from safety_gymnasium.utils.registration import make

    config = {
        'agent_name': 'Point',
        'intermediate_color': int_color,
        'goal_color': goal_color,
        'layout_seed': layout_seed,
    }

    # Create base env with config
    env = make(
        'PointLtl2-v0.opteq',
        config=config,
    )
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)

    # Create task for this color pair
    int_reach = frozenset([Assignment.single_proposition(int_color, propositions).to_frozen()])
    goal_reach = frozenset([Assignment.single_proposition(goal_color, propositions).to_frozen()])
    task = LDBASequence([(int_reach, frozenset()), (goal_reach, frozenset())])

    def sampler(props):
        return lambda: task

    env = SequenceWrapper(env, sampler(propositions), step_penalty=0.0)
    env = TimeLimit(env, max_episode_steps=500)
    env = RemoveTruncWrapper(env)

    return env


def run_episode(env, agent, int_color, goal_color, max_steps=500):
    """Run one episode and collect trajectory."""
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    unwrapped = get_unwrapped(env)

    # Get initial positions
    start_pos = unwrapped.agent_pos[:2].copy()
    zone_positions = {k: np.array(v[:2]) for k, v in unwrapped.zone_positions.items()}

    # Collect trajectory
    trajectory = [start_pos.copy()]

    # Find intermediate zones and goal zones
    int_zones = [(k, v) for k, v in zone_positions.items() if int_color in k]
    goal_zones = [(k, v) for k, v in zone_positions.items() if goal_color in k]

    if len(int_zones) < 2 or len(goal_zones) < 1:
        return None

    # Compute which intermediate zone is optimal (shorter total path)
    int_analysis = []
    for name, pos in int_zones:
        d_agent = np.linalg.norm(pos - start_pos)
        d_goal = min(np.linalg.norm(gpos - pos) for _, gpos in goal_zones)
        int_analysis.append({
            'name': name,
            'pos': pos,
            'd_agent': d_agent,
            'd_goal': d_goal,
            'total': d_agent + d_goal,
        })

    # Optimal = shortest total path
    optimal_zone = min(int_analysis, key=lambda x: x['total'])
    suboptimal_zone = max(int_analysis, key=lambda x: x['total'])

    # Check if truly equidistant
    d_diff = abs(int_analysis[0]['d_agent'] - int_analysis[1]['d_agent'])
    is_equidistant = d_diff < 0.1

    # Path length difference
    path_diff = suboptimal_zone['total'] - optimal_zone['total']

    # Run episode
    choice = None
    reached_intermediate = False
    reached_goal = False

    for step in range(max_steps):
        action = agent.get_action(obs, deterministic=True)
        if len(action) == 1:
            action = action[0]

        obs, reward, done, info = env.step(action)
        pos = unwrapped.agent_pos[:2].copy()
        trajectory.append(pos)

        # Check if we reached an intermediate zone first
        if not reached_intermediate:
            for name, zone_pos in int_zones:
                if np.linalg.norm(pos - zone_pos) < 0.4:
                    reached_intermediate = True
                    if name == optimal_zone['name']:
                        choice = 'optimal'
                    else:
                        choice = 'suboptimal'
                    break

        if done:
            reached_goal = info.get('success', False)
            break

    return {
        'trajectory': np.array(trajectory),
        'start_pos': start_pos,
        'zone_positions': zone_positions,
        'int_zones': int_zones,
        'goal_zones': goal_zones,
        'optimal_zone': optimal_zone,
        'suboptimal_zone': suboptimal_zone,
        'choice': choice or 'none',
        'reached_goal': reached_goal,
        'is_equidistant': is_equidistant,
        'd_diff': d_diff,
        'path_diff': path_diff,
        'int_color': int_color,
        'goal_color': goal_color,
    }


def plot_episode(ax, result, episode_num=0):
    """Plot a single episode using built-in visualization style."""
    # Use built-in axis setup (grid, clean look)
    setup_axis(ax)

    # Draw all zones using built-in function
    draw_zones(ax, result['zone_positions'])

    # Draw start position (orange diamond)
    draw_diamond(ax, result['start_pos'], color='orange')

    # Draw equidistant circle
    avg_d = (result['optimal_zone']['d_agent'] + result['suboptimal_zone']['d_agent']) / 2
    circle = plt.Circle(result['start_pos'], avg_d, fill=False, color='gray',
                       linestyle='--', linewidth=1, alpha=0.5)
    ax.add_patch(circle)

    # Draw trajectory with color based on choice
    traj = result['trajectory']
    if len(traj) > 1:
        if result['choice'] == 'optimal':
            line_color = '#4caf50'  # Green
        elif result['choice'] == 'suboptimal':
            line_color = '#f44336'  # Red
        else:
            line_color = 'gray'

        draw_path(ax, traj.tolist(), color=line_color, linewidth=3)

    # Title with choice and path difference
    choice_text = result['choice'].upper()
    choice_color = '#4caf50' if result['choice'] == 'optimal' else '#f44336' if result['choice'] == 'suboptimal' else 'gray'

    ax.set_title(f"{result['int_color']}→{result['goal_color']} | {choice_text}\n"
                f"Δpath={result['path_diff']:.2f}",
                fontsize=9, color=choice_color, fontweight='bold')


def load_model(exp_name, env_name='PointLtl2-v0'):
    """Load a trained model."""
    from envs import make_env

    temp_sampler = CurriculumSampler.partial(curricula[env_name])
    temp_env = make_env(env_name, temp_sampler, sequence=True)

    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp_name, seed=0)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()

    model = build_model(temp_env, training_status, config)
    model.eval()
    props = list(temp_env.get_propositions())
    temp_env.close()

    return model, props


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='fresh_baseline')
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("EQUIDISTANT OPTIMALITY TEST")
    print("=" * 70)
    print(f"Model: {args.exp}")
    print(f"Environment: PointLtl2-v0.opteq (equidistant)")
    print(f"Episodes: {args.num_episodes}")
    print()
    print("Both intermediate zones are at the SAME distance from agent.")
    print("The only way to choose optimally is to consider the full path.")
    print("=" * 70)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    print("\nLoading model...")
    model, propositions = load_model(args.exp)
    agent = Agent(model, set(propositions))

    # Run episodes with varied colors and layouts
    results = []
    equidistant_results = []

    print("\nRunning episodes...")
    for i in tqdm(range(args.num_episodes)):
        # Pick random color pair
        int_color, goal_color = random.choice(COLOR_PAIRS)

        # Create environment with this color pair and unique layout seed
        layout_seed = args.seed + i * 7

        try:
            env = create_opteq_env(int_color, goal_color, layout_seed, propositions)
            result = run_episode(env, agent, int_color, goal_color)
            env.close()
        except Exception as e:
            print(f"\nError in episode {i}: {e}")
            continue

        if result is None:
            continue

        results.append(result)
        if result['is_equidistant']:
            equidistant_results.append(result)

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nTotal episodes: {len(results)}")
    print(f"Truly equidistant (Δd < 0.1): {len(equidistant_results)}")

    optimal_count = 0
    suboptimal_count = 0
    success_count = 0

    if equidistant_results:
        optimal_count = sum(1 for r in equidistant_results if r['choice'] == 'optimal')
        suboptimal_count = sum(1 for r in equidistant_results if r['choice'] == 'suboptimal')
        other_count = len(equidistant_results) - optimal_count - suboptimal_count
        success_count = sum(1 for r in equidistant_results if r['reached_goal'])

        print(f"\nAmong equidistant cases:")
        print(f"  Optimal:    {optimal_count}/{len(equidistant_results)} ({100*optimal_count/len(equidistant_results):.1f}%)")
        print(f"  Suboptimal: {suboptimal_count}/{len(equidistant_results)} ({100*suboptimal_count/len(equidistant_results):.1f}%)")
        print(f"  Other:      {other_count}/{len(equidistant_results)}")
        print(f"\n  Success rate: {success_count}/{len(equidistant_results)} ({100*success_count/len(equidistant_results):.1f}%)")

        # Breakdown by color pair
        print("\nBy color pair:")
        pair_stats = {}
        for r in equidistant_results:
            pair = (r['int_color'], r['goal_color'])
            if pair not in pair_stats:
                pair_stats[pair] = {'optimal': 0, 'suboptimal': 0, 'total': 0}
            pair_stats[pair]['total'] += 1
            if r['choice'] == 'optimal':
                pair_stats[pair]['optimal'] += 1
            elif r['choice'] == 'suboptimal':
                pair_stats[pair]['suboptimal'] += 1

        for pair, stats in sorted(pair_stats.items()):
            if stats['total'] > 0:
                opt_pct = 100 * stats['optimal'] / stats['total']
                print(f"  {pair[0]}→{pair[1]}: {stats['optimal']}/{stats['total']} optimal ({opt_pct:.0f}%)")

        # Analyze by path difference magnitude
        print("\nBy path length difference:")
        small_diff = [r for r in equidistant_results if r['path_diff'] < 0.5]
        medium_diff = [r for r in equidistant_results if 0.5 <= r['path_diff'] < 1.0]
        large_diff = [r for r in equidistant_results if r['path_diff'] >= 1.0]

        for label, subset in [('Small (<0.5)', small_diff), ('Medium (0.5-1.0)', medium_diff), ('Large (≥1.0)', large_diff)]:
            if subset:
                opt = sum(1 for r in subset if r['choice'] == 'optimal')
                print(f"  {label}: {opt}/{len(subset)} optimal ({100*opt/len(subset):.0f}%)")

    # Create visualization
    output_dir = 'interpretability/results/optimality_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Register FancyAxes projection
    projections.register_projection(FancyAxes)

    # Plot equidistant cases
    n_show = min(20, len(equidistant_results))
    if n_show > 0:
        cols = 5
        rows = (n_show + cols - 1) // cols

        fig = plt.figure(figsize=(4 * cols, 4.5 * rows))

        for i in range(n_show):
            ax = fig.add_subplot(rows, cols, i + 1,
                               projection='fancy_box_axes',
                               edgecolor='gray', linewidth=0.5)
            plot_episode(ax, equidistant_results[i], i)

        opt_pct = 100 * optimal_count / len(equidistant_results) if equidistant_results else 0
        subopt_pct = 100 * suboptimal_count / len(equidistant_results) if equidistant_results else 0

        fig.suptitle(f"EQUIDISTANT Optimality Test: {args.exp}\n"
                    f"Both intermediates SAME distance from agent (gray circle)\n"
                    f"Optimal: {optimal_count}/{len(equidistant_results)} ({opt_pct:.0f}%) | "
                    f"Suboptimal: {suboptimal_count}/{len(equidistant_results)} ({subopt_pct:.0f}%)",
                    fontsize=14, fontweight='bold')

        plt.tight_layout(pad=3)
        output_file = f"{output_dir}/optimality_test_equidistant_{args.exp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\nSaved visualization: {output_file}")

    print("\n" + "=" * 70)

    return {
        'optimal_count': optimal_count,
        'suboptimal_count': suboptimal_count,
        'equidistant_count': len(equidistant_results),
        'total_count': len(results),
    }


if __name__ == '__main__':
    main()
