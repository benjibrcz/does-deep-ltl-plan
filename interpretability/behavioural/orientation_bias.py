#!/usr/bin/env python
"""
Analyze whether "spatial bias" is actually "orientation bias"

Hypothesis: The agent doesn't have a LEFT/RIGHT preference in absolute coordinates.
Instead, it has a "forward motion" preference - it goes in the direction it's initially facing.

This script:
1. Runs OPTEQ episodes and logs agent's initial orientation
2. Computes whether chosen zone is "forward" or "backward" relative to heading
3. Tests if forward/backward explains choices better than absolute LEFT/RIGHT
"""
import sys
sys.path.insert(0, 'src/')

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from gymnasium.wrappers import TimeLimit, FlattenObservation

from envs import make_env
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from envs.seq_wrapper import SequenceWrapper
from envs.remove_trunc_wrapper import RemoveTruncWrapper
from model.model import build_model
from config import model_configs
from utils.model_store import ModelStore
from sequence.samplers import CurriculumSampler, curricula
from ltl.automata import LDBASequence
from ltl.logic import Assignment
import preprocessing
from safety_gymnasium.utils.registration import make


def get_unwrapped(env):
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    return unwrapped


def angle_to_zone(agent_pos, zone_pos):
    """Calculate angle from agent to zone (in radians)."""
    dx = zone_pos[0] - agent_pos[0]
    dy = zone_pos[1] - agent_pos[1]
    return np.arctan2(dy, dx)


def angle_difference(angle1, angle2):
    """Compute smallest angle difference (handling wraparound)."""
    diff = angle1 - angle2
    # Normalize to [-pi, pi]
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff


def is_forward(agent_rot, agent_pos, zone_pos, threshold_deg=90):
    """Check if zone is in the "forward" direction of agent heading."""
    zone_angle = angle_to_zone(agent_pos, zone_pos)
    diff = abs(angle_difference(zone_angle, agent_rot))
    return diff < np.deg2rad(threshold_deg)


def run_orientation_analysis(model_name='fresh_baseline', n_episodes=100, seed=42):
    """Run OPTEQ episodes and analyze orientation vs choice."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load model
    env_name = 'PointLtl2-v0'
    temp_sampler = CurriculumSampler.partial(curricula[env_name])
    temp_env = make_env(env_name, temp_sampler, sequence=True)

    config = model_configs[env_name]
    model_store = ModelStore(env_name, model_name, seed=0)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()

    model = build_model(temp_env, training_status, config)
    model.eval()
    propositions = list(temp_env.get_propositions())

    results = []

    for ep in tqdm(range(n_episodes), desc=f"Running {model_name}"):
        # Create OPTEQ environment
        env_config = {
            'agent_name': 'Point',
            'intermediate_color': 'blue',
            'goal_color': 'green',
            'layout_seed': seed + ep,
        }

        try:
            base_env = make('PointLtl2-v0.opteq', config=env_config)
        except Exception as e:
            continue

        # Wrap the environment properly
        env = SafetyGymWrapper(base_env)
        env = FlattenObservation(env)

        # Create task
        int_reach = frozenset([Assignment.single_proposition('blue', propositions).to_frozen()])
        goal_reach = frozenset([Assignment.single_proposition('green', propositions).to_frozen()])
        task = LDBASequence([(int_reach, frozenset()), (goal_reach, frozenset())])

        def sampler(props):
            return lambda: task

        env = SequenceWrapper(env, sampler(propositions), step_penalty=0.0)
        env = TimeLimit(env, max_episode_steps=500)
        env = RemoveTruncWrapper(env)

        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        unwrapped = get_unwrapped(env)

        # Get initial state
        agent_pos = unwrapped.agent_pos[:2].copy()
        agent_rot = unwrapped.agent_rot
        zone_positions = {k: np.array(v[:2]) for k, v in unwrapped.zone_positions.items()}

        # Get intermediate zone positions
        int_zones = [k for k in zone_positions if 'blue' in k]
        int_positions = {k: zone_positions[k] for k in int_zones}

        # Run episode
        trajectory = [agent_pos.copy()]
        chosen_zone = None

        for step in range(500):
            obs_tensor = preprocessing.preprocess_obss([obs], propositions)
            with torch.no_grad():
                dist, _ = model(obs_tensor)
            action = dist.sample().numpy().flatten()

            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            current_pos = get_unwrapped(env).agent_pos[:2].copy()
            trajectory.append(current_pos.copy())

            # Check if reached an intermediate zone
            for zone_name, zone_pos in int_positions.items():
                if np.linalg.norm(current_pos - zone_pos) < 0.3:
                    chosen_zone = zone_name
                    break

            if chosen_zone or done:
                break

        if chosen_zone is None:
            env.close()
            continue

        chosen_pos = int_positions[chosen_zone]
        other_zone = [z for z in int_zones if z != chosen_zone][0]
        other_pos = int_positions[other_zone]

        # Compute metrics
        # 1. Absolute position (LEFT/RIGHT)
        chosen_x = chosen_pos[0]
        chose_left = chosen_x < 0

        # 2. Forward/backward relative to initial heading
        chosen_forward = is_forward(agent_rot, agent_pos, chosen_pos)
        other_forward = is_forward(agent_rot, agent_pos, other_pos)

        # 3. Angle difference from heading
        chosen_angle = angle_to_zone(agent_pos, chosen_pos)
        other_angle = angle_to_zone(agent_pos, other_pos)
        chosen_angle_diff = abs(angle_difference(chosen_angle, agent_rot))
        other_angle_diff = abs(angle_difference(other_angle, agent_rot))

        # 4. Which zone is "more forward"?
        chose_more_forward = chosen_angle_diff < other_angle_diff

        # 5. Goal direction (optimal choice)
        goal_pos = zone_positions['green_zone0']  # Assuming green_zone0 is the goal
        d_chosen_to_goal = np.linalg.norm(chosen_pos - goal_pos)
        d_other_to_goal = np.linalg.norm(other_pos - goal_pos)
        chose_optimal = d_chosen_to_goal < d_other_to_goal

        results.append({
            'episode': ep,
            'agent_rot': agent_rot,
            'agent_rot_deg': np.rad2deg(agent_rot),
            'chosen_zone': chosen_zone,
            'chosen_x': chosen_x,
            'chose_left': chose_left,
            'chosen_forward': chosen_forward,
            'other_forward': other_forward,
            'chosen_angle_diff': np.rad2deg(chosen_angle_diff),
            'other_angle_diff': np.rad2deg(other_angle_diff),
            'chose_more_forward': chose_more_forward,
            'chose_optimal': chose_optimal,
        })

        env.close()

    return results


def analyze_results(results, model_name):
    """Analyze and visualize orientation bias results."""

    import pandas as pd
    df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print(f"Orientation Bias Analysis: {model_name}")
    print(f"{'='*60}")
    print(f"N episodes: {len(df)}")

    # 1. Absolute LEFT/RIGHT bias
    left_rate = df['chose_left'].mean()
    print(f"\n1. Absolute spatial bias:")
    print(f"   Chose LEFT (x<0): {left_rate:.1%}")
    print(f"   Chose RIGHT (x>0): {1-left_rate:.1%}")

    # 2. Forward preference
    forward_rate = df['chose_more_forward'].mean()
    print(f"\n2. Forward preference (chose zone closer to heading direction):")
    print(f"   Chose MORE FORWARD: {forward_rate:.1%}")
    print(f"   Chose LESS FORWARD: {1-forward_rate:.1%}")

    # Statistical test
    n = len(df)
    k = df['chose_more_forward'].sum()
    binom_result = stats.binomtest(k, n, 0.5, alternative='two-sided')
    binom_p = binom_result.pvalue
    print(f"   Binomial test (vs 50%): p = {binom_p:.4f}")

    # 3. Both zones forward vs one forward
    both_forward = (df['chosen_forward'] & df['other_forward']).sum()
    neither_forward = (~df['chosen_forward'] & ~df['other_forward']).sum()
    one_forward = len(df) - both_forward - neither_forward
    print(f"\n3. Zone visibility from heading:")
    print(f"   Both zones forward (<90°): {both_forward} ({100*both_forward/len(df):.1f}%)")
    print(f"   One zone forward: {one_forward} ({100*one_forward/len(df):.1f}%)")
    print(f"   Neither forward: {neither_forward} ({100*neither_forward/len(df):.1f}%)")

    # When only one is forward, does agent choose it?
    one_fwd_mask = df['chosen_forward'] != df['other_forward']
    if one_fwd_mask.sum() > 0:
        chose_the_forward_one = df.loc[one_fwd_mask, 'chosen_forward'].mean()
        print(f"   When only one forward, chose it: {chose_the_forward_one:.1%} (N={one_fwd_mask.sum()})")

    # 4. Correlation analysis
    print(f"\n4. Correlations:")

    # Does initial heading predict choice direction?
    # If heading points left (rot ~ 180°) and agent goes left, that's forward bias
    heading_x = np.cos(df['agent_rot'])  # Positive = facing right
    chose_right = ~df['chose_left']
    corr_heading_choice = np.corrcoef(heading_x, chose_right.astype(float))[0,1]
    print(f"   Heading direction ↔ Choice direction: r = {corr_heading_choice:.3f}")

    # 5. Optimal choice rate
    optimal_rate = df['chose_optimal'].mean()
    print(f"\n5. Optimal choice rate: {optimal_rate:.1%}")

    # 6. Does forward bias explain "optimal" better than spatial bias?
    print(f"\n6. What explains choices?")

    # Logistic regression would be ideal, but let's do simpler analysis
    # If LEFT bias explains choices, LEFT should predict choice
    # If FORWARD bias explains choices, FORWARD should predict choice

    forward_explains = df['chose_more_forward'].mean()
    left_explains = df['chose_left'].mean()  # This assumes LEFT = chosen, which we need to check

    # Actually, let's check partial correlations
    print(f"   Forward preference strength: {abs(forward_explains - 0.5)*2:.1%} away from random")
    print(f"   Left preference strength: {abs(left_explains - 0.5)*2:.1%} away from random")

    return df


def plot_orientation_analysis(df, model_name, save_path=None):
    """Create visualization of orientation bias."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Heading distribution
    ax = axes[0, 0]
    ax.hist(df['agent_rot_deg'], bins=36, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Initial Heading (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Initial Agent Heading')
    ax.axvline(0, color='red', linestyle='--', label='Facing right')
    ax.axvline(180, color='blue', linestyle='--', label='Facing left')
    ax.axvline(-180, color='blue', linestyle='--')
    ax.legend()

    # 2. Heading vs Choice
    ax = axes[0, 1]
    colors = ['blue' if left else 'red' for left in df['chose_left']]
    ax.scatter(df['agent_rot_deg'], df['chosen_x'], c=colors, alpha=0.5)
    ax.set_xlabel('Initial Heading (degrees)')
    ax.set_ylabel('Chosen Zone X Position')
    ax.set_title('Heading vs Choice Position')
    ax.axhline(0, color='gray', linestyle='--')

    # Add correlation
    corr = np.corrcoef(np.cos(df['agent_rot']), df['chosen_x'])[0,1]
    ax.text(0.05, 0.95, f'r(cos(heading), x) = {corr:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')

    # 3. Forward preference breakdown
    ax = axes[1, 0]
    categories = ['More Forward', 'Less Forward']
    counts = [df['chose_more_forward'].sum(), (~df['chose_more_forward']).sum()]
    bars = ax.bar(categories, counts, color=['green', 'orange'], edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Forward vs Backward Preference')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}\n({100*count/len(df):.1f}%)', ha='center')

    # 4. Angle difference distribution
    ax = axes[1, 1]
    ax.hist(df['chosen_angle_diff'], bins=18, alpha=0.7, label='Chosen zone', color='green')
    ax.hist(df['other_angle_diff'], bins=18, alpha=0.7, label='Other zone', color='red')
    ax.set_xlabel('Angle from Heading (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Angle Difference from Heading')
    ax.axvline(90, color='black', linestyle='--', label='90° threshold')
    ax.legend()

    plt.suptitle(f'Orientation Bias Analysis: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()
    return fig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='fresh_baseline')
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    results = run_orientation_analysis(args.model, args.n_episodes, args.seed)
    df = analyze_results(results, args.model)

    save_path = f'interpretability/results/empirical_difficulty/orientation_bias_{args.model}.png'
    plot_orientation_analysis(df, args.model, save_path)

    # Save raw data
    csv_path = f'interpretability/results/empirical_difficulty/orientation_bias_{args.model}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")
