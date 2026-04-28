#!/usr/bin/env python3
"""
Clean Optimality Test with proper trajectory visualization.

Tests optimality behavior with varied:
- Zone positions (different layouts each episode)
- Color pairs (different intermediate/goal colors)
- Agent starting positions
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


def create_optvar_env(int_color, goal_color, layout_seed, propositions):
    """Create an optvar environment with specified colors and layout seed."""
    from safety_gymnasium.utils.registration import make

    # Custom config must include agent_name since we're overriding the registered config
    config = {
        'agent_name': 'Point',
        'intermediate_color': int_color,
        'goal_color': goal_color,
        'layout_seed': layout_seed,
    }

    # Create base env with config
    env = make(
        'PointLtl2-v0.optvar',
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

    # Compute which intermediate zone is myopic (closer to agent) vs optimal (shorter total path)
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

    # Myopic = closest to agent
    myopic_zone = min(int_analysis, key=lambda x: x['d_agent'])

    # Optimal = shortest total path
    optimal_zone = min(int_analysis, key=lambda x: x['total'])

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
                    elif name == myopic_zone['name']:
                        choice = 'myopic'
                    else:
                        choice = 'other'
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
        'myopic_zone': myopic_zone,
        'optimal_zone': optimal_zone,
        'choice': choice or 'none',
        'reached_goal': reached_goal,
        'is_contested': myopic_zone['name'] != optimal_zone['name'],
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

    # Draw trajectory with color based on choice
    traj = result['trajectory']
    if len(traj) > 1:
        if result['choice'] == 'optimal':
            line_color = '#4caf50'  # Green (matches paper)
        elif result['choice'] == 'myopic':
            line_color = '#ff9800'  # Orange (matches paper's "myopic" color)
        else:
            line_color = 'gray'

        draw_path(ax, traj.tolist(), color=line_color, linewidth=3)

    # Title with choice and colors
    choice_text = result['choice'].upper()
    choice_color = '#4caf50' if result['choice'] == 'optimal' else '#ff9800' if result['choice'] == 'myopic' else 'gray'

    ax.set_title(f"{result['int_color']}→{result['goal_color']} | {choice_text}",
                fontsize=10, color=choice_color, fontweight='bold')


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
    print("CLEAN OPTIMALITY TEST (Varied Maps & Colors)")
    print("=" * 70)
    print(f"Model: {args.exp}")
    print(f"Environment: PointLtl2-v0.optvar (varied)")
    print(f"Episodes: {args.num_episodes}")
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
    contested_results = []

    print("\nRunning episodes...")
    for i in tqdm(range(args.num_episodes)):
        # Pick random color pair
        int_color, goal_color = random.choice(COLOR_PAIRS)

        # Create environment with this color pair and unique layout seed
        layout_seed = args.seed + i * 7  # Multiply to get more variation

        try:
            env = create_optvar_env(int_color, goal_color, layout_seed, propositions)
            result = run_episode(env, agent, int_color, goal_color)
            env.close()
        except Exception as e:
            print(f"\nError in episode {i}: {e}")
            continue

        if result is None:
            continue

        results.append(result)
        if result['is_contested']:
            contested_results.append(result)

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nTotal episodes: {len(results)}")
    print(f"Contested (optimal != myopic): {len(contested_results)}")
    print(f"Non-contested (optimal == myopic): {len(results) - len(contested_results)}")

    optimal_count = 0
    myopic_count = 0
    success_count = 0

    if contested_results:
        optimal_count = sum(1 for r in contested_results if r['choice'] == 'optimal')
        myopic_count = sum(1 for r in contested_results if r['choice'] == 'myopic')
        other_count = len(contested_results) - optimal_count - myopic_count
        success_count = sum(1 for r in contested_results if r['reached_goal'])

        print(f"\nAmong contested cases:")
        print(f"  Optimal: {optimal_count}/{len(contested_results)} ({100*optimal_count/len(contested_results):.1f}%)")
        print(f"  Myopic:  {myopic_count}/{len(contested_results)} ({100*myopic_count/len(contested_results):.1f}%)")
        print(f"  Other:   {other_count}/{len(contested_results)}")
        print(f"\n  Success rate: {success_count}/{len(contested_results)} ({100*success_count/len(contested_results):.1f}%)")

        # Breakdown by color pair
        print("\nBy color pair:")
        pair_stats = {}
        for r in contested_results:
            pair = (r['int_color'], r['goal_color'])
            if pair not in pair_stats:
                pair_stats[pair] = {'optimal': 0, 'myopic': 0, 'total': 0}
            pair_stats[pair]['total'] += 1
            if r['choice'] == 'optimal':
                pair_stats[pair]['optimal'] += 1
            elif r['choice'] == 'myopic':
                pair_stats[pair]['myopic'] += 1

        for pair, stats in sorted(pair_stats.items()):
            if stats['total'] > 0:
                opt_pct = 100 * stats['optimal'] / stats['total']
                print(f"  {pair[0]}→{pair[1]}: {stats['optimal']}/{stats['total']} optimal ({opt_pct:.0f}%)")

    # Create visualization
    output_dir = 'interpretability/results/optimality_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Register FancyAxes projection
    projections.register_projection(FancyAxes)

    # Plot contested cases
    n_show = min(20, len(contested_results))
    if n_show > 0:
        cols = 5
        rows = (n_show + cols - 1) // cols

        fig = plt.figure(figsize=(4 * cols, 4 * rows))

        for i in range(n_show):
            ax = fig.add_subplot(rows, cols, i + 1,
                               projection='fancy_box_axes',
                               edgecolor='gray', linewidth=0.5)
            plot_episode(ax, contested_results[i], i)

        opt_pct = 100 * optimal_count / len(contested_results) if contested_results else 0
        myo_pct = 100 * myopic_count / len(contested_results) if contested_results else 0

        fig.suptitle(f"Optimality Test: {args.exp}\n"
                    f"Contested Cases Only | Optimal: {optimal_count}/{len(contested_results)} ({opt_pct:.0f}%) | "
                    f"Myopic: {myopic_count}/{len(contested_results)} ({myo_pct:.0f}%)",
                    fontsize=14, fontweight='bold')

        plt.tight_layout(pad=3)
        output_file = f"{output_dir}/optimality_test_{args.exp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\nSaved visualization: {output_file}")

    print("\n" + "=" * 70)

    return {
        'optimal_count': optimal_count,
        'myopic_count': myopic_count,
        'contested_count': len(contested_results),
        'total_count': len(results),
    }


if __name__ == '__main__':
    main()
