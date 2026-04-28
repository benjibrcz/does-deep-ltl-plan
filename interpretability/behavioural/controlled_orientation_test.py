#!/usr/bin/env python
"""
Optimality Test with Controlled Orientation

Controls for forward bias by setting the agent's initial orientation
to face the midpoint between the two intermediate zones.

This way, both zones are equally "forward" and any preference must come
from actual planning (considering which path to the goal is shorter).
"""
import sys
sys.path.insert(0, 'src/')

import numpy as np
import torch
from tqdm import tqdm
from gymnasium.wrappers import TimeLimit, FlattenObservation

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
    """Get to the base Builder environment."""
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    return unwrapped


def get_safety_gym_wrapper(env):
    """Get the SafetyGymWrapper from the environment chain."""
    current = env
    while hasattr(current, 'env'):
        if isinstance(current, SafetyGymWrapper):
            return current
        current = current.env
    return None


def angle_to_point(from_pos, to_pos):
    """Calculate angle from one point to another."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    return np.arctan2(dy, dx)


def set_agent_orientation(env, target_angle):
    """Set the agent's orientation (yaw) to face a specific direction."""
    unwrapped = get_unwrapped(env)
    # Access the task's mujoco data
    task = unwrapped.task
    # qpos is [x, y, yaw] for Point agent
    task.data.qpos[2] = target_angle
    # Reset velocity
    task.data.qvel[:] = 0


def run_controlled_optimality_test(model_name='fresh_baseline', n_episodes=100, seed=42):
    """Run optimality test with controlled orientation."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load model
    env_name = 'PointLtl2-v0'
    temp_sampler = CurriculumSampler.partial(curricula[env_name])
    temp_env = make_env_for_model(env_name, temp_sampler)

    config = model_configs[env_name]
    model_store = ModelStore(env_name, model_name, seed=0)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()

    model = build_model(temp_env, training_status, config)
    model.eval()
    propositions = list(temp_env.get_propositions())
    temp_env.close()

    results = []

    for ep in tqdm(range(n_episodes), desc=f"Running {model_name}"):
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

        # Wrap the environment
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

        # Reset and get initial state
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        # Get positions from SafetyGymWrapper
        sgw = get_safety_gym_wrapper(env)
        agent_pos = sgw.agent_pos[:2].copy()
        zone_positions = {k: np.array(v[:2]) for k, v in sgw.zone_positions.items()}

        # Get intermediate zone positions
        int_zones = [k for k in zone_positions if 'blue' in k]
        int_positions = {k: zone_positions[k] for k in int_zones}
        zone_list = list(int_positions.keys())

        # Calculate midpoint between the two intermediate zones
        midpoint = (int_positions[zone_list[0]] + int_positions[zone_list[1]]) / 2

        # Set agent orientation to face the midpoint
        target_angle = angle_to_point(agent_pos, midpoint)
        set_agent_orientation(env, target_angle)

        # Get new observation after orientation change
        # Need to step with zero action to update observation
        obs, _, _, _ = env.step(np.zeros(2))

        # Verify both zones are roughly equidistant from heading
        angle_to_zone0 = angle_to_point(agent_pos, int_positions[zone_list[0]])
        angle_to_zone1 = angle_to_point(agent_pos, int_positions[zone_list[1]])
        angle_diff_0 = abs(angle_to_zone0 - target_angle)
        angle_diff_1 = abs(angle_to_zone1 - target_angle)

        # Run episode
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

            current_pos = get_safety_gym_wrapper(env).agent_pos[:2].copy()

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

        # Compute optimality
        goal_pos = zone_positions['green_zone0']
        chosen_pos = int_positions[chosen_zone]
        other_zone = [z for z in int_zones if z != chosen_zone][0]
        other_pos = int_positions[other_zone]

        d_chosen_to_goal = np.linalg.norm(chosen_pos - goal_pos)
        d_other_to_goal = np.linalg.norm(other_pos - goal_pos)
        chose_optimal = d_chosen_to_goal < d_other_to_goal

        # Also check spatial position (for comparison)
        chose_left = chosen_pos[0] < 0

        results.append({
            'episode': ep,
            'chosen_zone': chosen_zone,
            'chose_optimal': chose_optimal,
            'chose_left': chose_left,
            'angle_diff_chosen': np.rad2deg(angle_diff_0 if chosen_zone == zone_list[0] else angle_diff_1),
            'angle_diff_other': np.rad2deg(angle_diff_1 if chosen_zone == zone_list[0] else angle_diff_0),
            'd_chosen_to_goal': d_chosen_to_goal,
            'd_other_to_goal': d_other_to_goal,
        })

        env.close()

    return results


def make_env_for_model(env_name, sampler):
    """Create environment for loading model."""
    from envs import make_env
    return make_env(env_name, sampler, sequence=True)


def analyze_results(results, model_name):
    """Analyze controlled optimality results."""
    import pandas as pd
    from scipy import stats

    df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print(f"Controlled Orientation Optimality Test: {model_name}")
    print(f"{'='*60}")
    print(f"N episodes: {len(df)}")

    # Main result: optimal choice rate
    optimal_rate = df['chose_optimal'].mean()
    n = len(df)
    k = df['chose_optimal'].sum()

    # Wilson score interval (manual calculation)
    z = 1.96  # 95% CI
    p_hat = k / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denominator
    ci_low = center - margin
    ci_high = center + margin

    print(f"\n*** MAIN RESULT ***")
    print(f"Optimal choice rate: {optimal_rate:.1%}")
    print(f"95% CI: [{ci_low:.1%}, {ci_high:.1%}]")

    # Binomial test
    binom_result = stats.binomtest(k, n, 0.5, alternative='two-sided')
    print(f"Binomial test vs 50%: p = {binom_result.pvalue:.4f}")

    if ci_low > 0.5:
        print("*** SIGNIFICANT: Agent shows ABOVE-CHANCE optimal choice! ***")
    elif ci_high < 0.5:
        print("*** SIGNIFICANT: Agent shows BELOW-CHANCE optimal choice! ***")
    else:
        print("*** NOT SIGNIFICANT: Confidence interval includes 50% (random) ***")

    # Spatial bias (for comparison)
    left_rate = df['chose_left'].mean()
    print(f"\nSpatial bias check:")
    print(f"  Chose LEFT: {left_rate:.1%}")
    print(f"  Chose RIGHT: {1-left_rate:.1%}")

    # Angle verification
    print(f"\nOrientation control verification:")
    print(f"  Mean angle diff to chosen: {df['angle_diff_chosen'].mean():.1f}°")
    print(f"  Mean angle diff to other: {df['angle_diff_other'].mean():.1f}°")

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='fresh_baseline')
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("Running optimality test with CONTROLLED ORIENTATION")
    print("Agent faces midpoint between intermediate zones")
    print("This removes forward bias - both zones equally 'forward'")
    print()

    results = run_controlled_optimality_test(args.model, args.n_episodes, args.seed)
    df = analyze_results(results, args.model)

    # Save results
    csv_path = f'interpretability/results/empirical_difficulty/controlled_orientation_{args.model}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved data to {csv_path}")
