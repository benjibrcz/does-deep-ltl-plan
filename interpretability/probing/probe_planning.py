#!/usr/bin/env python3
"""
Probe for planning-related representations using the varied optvar environment.

This tests whether the model encodes the information needed for optimal planning:
1. Distance to intermediate zones (should be easy - observable)
2. Distance from intermediates to goal (requires memory/computation)
3. Total path via each intermediate (requires computation)
4. Which intermediate is optimal vs myopic (requires comparison)

Key hypothesis: If model is myopic, it should encode distances but NOT:
- Chained distances (intermediate → goal)
- Total path lengths
- Optimal vs myopic classification
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(0, 'src')

import preprocessing
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from ltl.automata import LDBASequence
from ltl.logic import Assignment

from safety_gymnasium.utils.registration import make as sg_make
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from gymnasium.wrappers import FlattenObservation, TimeLimit
from envs.seq_wrapper import SequenceWrapper
from envs.remove_trunc_wrapper import RemoveTruncWrapper

torch.set_grad_enabled(False)

COLORS = ['blue', 'green', 'yellow', 'magenta']


class ActivationCollector:
    """Collects activations from model layers."""

    def __init__(self, model, propositions):
        self.model = model
        self.propositions = propositions

    def get_activations(self, obs):
        """Get all intermediate activations for a single observation."""
        if not isinstance(obs, list):
            obs = [obs]

        preprocessed = preprocessing.preprocess_obss(obs, self.propositions)

        # Compute env embedding
        if self.model.env_net is not None:
            env_embedding = self.model.env_net(preprocessed.features)
        else:
            env_embedding = preprocessed.features

        # Compute LTL embedding
        ltl_embedding = self.model.ltl_net(preprocessed.seq)

        # Combined embedding
        combined_embedding = torch.cat([env_embedding, ltl_embedding], dim=1)

        # Actor hidden state
        actor_hidden = self.model.actor.enc(combined_embedding)

        # Value
        value = self.model.critic(combined_embedding).squeeze(1)

        return {
            'env_embedding': env_embedding.detach().numpy(),
            'ltl_embedding': ltl_embedding.detach().numpy(),
            'combined_embedding': combined_embedding.detach().numpy(),
            'actor_hidden': actor_hidden.detach().numpy(),
            'value': value.detach().numpy(),
        }


def get_unwrapped(env):
    """Get the base environment."""
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    return unwrapped


def create_optvar_env(int_color, goal_color, layout_seed, propositions):
    """Create an optvar environment with specified colors and layout seed."""
    config = {
        'agent_name': 'Point',
        'intermediate_color': int_color,
        'goal_color': goal_color,
        'layout_seed': layout_seed,
    }

    env = sg_make('PointLtl2-v0.optvar', config=config)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)

    # Create task for this color pair
    int_reach = frozenset([Assignment.single_proposition(int_color, propositions).to_frozen()])
    goal_reach = frozenset([Assignment.single_proposition(goal_color, propositions).to_frozen()])
    task = LDBASequence([(int_reach, frozenset()), (goal_reach, frozenset())])

    def sampler(props):
        return lambda: task

    env = SequenceWrapper(env, sampler(propositions), step_penalty=0.0)
    env = TimeLimit(env, max_episode_steps=300)
    env = RemoveTruncWrapper(env)

    return env


def collect_probing_data(model, propositions, n_episodes=100, max_steps=150):
    """Collect activations and planning-relevant labels from optvar environments."""

    collector = ActivationCollector(model, set(propositions))
    data = defaultdict(list)

    # Color pairs to sample from
    color_pairs = [(a, b) for a in COLORS for b in COLORS if a != b]

    print(f"Collecting data from {n_episodes} episodes...")

    for ep in tqdm(range(n_episodes)):
        # Random color pair and layout
        int_color, goal_color = color_pairs[ep % len(color_pairs)]
        layout_seed = 1000 + ep * 7

        try:
            env = create_optvar_env(int_color, goal_color, layout_seed, propositions)
        except Exception as e:
            continue

        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        unwrapped = get_unwrapped(env)

        # Get zone positions
        zone_positions = {k: np.array(v[:2]) for k, v in unwrapped.zone_positions.items()}

        # Find intermediate and goal zones
        int_zones = [(k, v) for k, v in zone_positions.items() if int_color in k]
        goal_zones = [(k, v) for k, v in zone_positions.items() if goal_color in k]

        if len(int_zones) < 2 or len(goal_zones) < 1:
            env.close()
            continue

        goal_pos = goal_zones[0][1]

        for step in range(max_steps):
            agent_pos = unwrapped.agent_pos[:2].copy()

            # Get activations
            activations = collector.get_activations(obs)

            # Compute planning-relevant labels
            labels = {}

            # Basic distances (should be easy - observable)
            for i, (name, pos) in enumerate(int_zones):
                labels[f'd_agent_to_int{i}'] = np.linalg.norm(agent_pos - pos)

            labels['d_agent_to_goal'] = np.linalg.norm(agent_pos - goal_pos)

            # Chained distances (requires computation/memory)
            for i, (name, pos) in enumerate(int_zones):
                labels[f'd_int{i}_to_goal'] = np.linalg.norm(pos - goal_pos)

            # Total path via each intermediate (key for optimality)
            totals = []
            for i, (name, pos) in enumerate(int_zones):
                total = np.linalg.norm(agent_pos - pos) + np.linalg.norm(pos - goal_pos)
                labels[f'total_via_int{i}'] = total
                totals.append((i, total))

            # Which intermediate is optimal (shorter total) vs myopic (closer)
            agent_dists = [(i, np.linalg.norm(agent_pos - int_zones[i][1])) for i in range(len(int_zones))]
            myopic_idx = min(agent_dists, key=lambda x: x[1])[0]
            optimal_idx = min(totals, key=lambda x: x[1])[0]

            labels['myopic_idx'] = myopic_idx
            labels['optimal_idx'] = optimal_idx
            labels['is_contested'] = float(myopic_idx != optimal_idx)

            # Optimality gap (how much better is optimal path)
            if len(totals) >= 2:
                sorted_totals = sorted(totals, key=lambda x: x[1])
                labels['optimality_gap'] = sorted_totals[1][1] - sorted_totals[0][1]

            # Value prediction targets
            labels['value'] = activations['value'][0]

            # Store data
            for key, val in activations.items():
                data[f'act_{key}'].append(val.flatten())
            for key, val in labels.items():
                data[f'label_{key}'].append(val)

            data['agent_pos'].append(agent_pos.copy())
            data['int_color'].append(int_color)
            data['goal_color'].append(goal_color)
            data['step'].append(step)
            data['episode'].append(ep)

            # Take action
            with torch.no_grad():
                obs_list = [obs] if not isinstance(obs, list) else obs
                preprocessed = preprocessing.preprocess_obss(obs_list, set(propositions))
                dist, _ = model(preprocessed)
                action = dist.mode.numpy().flatten()

            obs, reward, done, info = env.step(action)

            if done:
                break

        env.close()

    # Convert to numpy arrays
    result = {}
    for key, values in data.items():
        try:
            result[key] = np.array(values)
        except:
            result[key] = values

    return result


def train_probe(X, y, probe_type='regression'):
    """Train a linear probe and return performance metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if probe_type == 'regression':
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        return {'r2': score, 'y_test': y_test, 'y_pred': y_pred}
    else:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        return {'accuracy': score, 'y_test': y_test, 'y_pred': y_pred}


def run_probing(data, output_dir):
    """Run probing analysis for all layer-label combinations."""

    results = {}

    act_keys = [k for k in data.keys() if k.startswith('act_')]
    label_keys = [k for k in data.keys() if k.startswith('label_')]

    print(f"\nLayers: {[k.replace('act_', '') for k in act_keys]}")
    print(f"Labels: {[k.replace('label_', '') for k in label_keys]}")

    for act_key in act_keys:
        layer_name = act_key.replace('act_', '')
        X = data[act_key]
        print(f"\n--- {layer_name} (dim={X.shape[1]}) ---")

        layer_results = {}

        for label_key in label_keys:
            label_name = label_key.replace('label_', '')
            y = data[label_key]

            if np.std(y) < 1e-6:
                continue

            # Classification for index labels
            if label_name in ['myopic_idx', 'optimal_idx']:
                result = train_probe(X, y.astype(int), 'classification')
                metric = 'accuracy'
            else:
                result = train_probe(X, y, 'regression')
                metric = 'r2'

            layer_results[label_name] = result
            score = result.get(metric, result.get('r2', result.get('accuracy', 0)))
            print(f"  {label_name}: {metric}={score:.3f}")

        results[layer_name] = layer_results

    return results


def plot_results(results, output_dir):
    """Create visualizations."""

    layers = list(results.keys())

    # Key planning probes to highlight
    planning_labels = [
        'd_agent_to_int0', 'd_agent_to_int1',  # Observable
        'd_int0_to_goal', 'd_int1_to_goal',    # Requires memory
        'total_via_int0', 'total_via_int1',    # Requires computation
        'optimality_gap',                       # Key for optimal choice
    ]

    # Filter to labels that exist
    available_labels = set()
    for layer_results in results.values():
        available_labels.update(layer_results.keys())
    planning_labels = [l for l in planning_labels if l in available_labels]

    if not planning_labels:
        print("No planning labels found!")
        return

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    matrix = np.zeros((len(layers), len(planning_labels)))
    for i, layer in enumerate(layers):
        for j, label in enumerate(planning_labels):
            if label in results[layer]:
                r = results[layer][label]
                matrix[i, j] = r.get('r2', r.get('accuracy', 0))

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(planning_labels)))
    ax.set_xticklabels(planning_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=10)
    ax.set_title('Planning Probe Results (R² / Accuracy)', fontsize=14, fontweight='bold')

    # Add value annotations
    for i in range(len(layers)):
        for j in range(len(planning_labels)):
            val = matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'optvar_probe_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'optvar_probe_heatmap.png'}")

    # Bar chart: Observable vs Computed features
    fig, ax = plt.subplots(figsize=(10, 6))

    observable = ['d_agent_to_int0', 'd_agent_to_int1', 'd_agent_to_goal']
    computed = ['d_int0_to_goal', 'd_int1_to_goal', 'total_via_int0', 'total_via_int1', 'optimality_gap']

    obs_scores = []
    comp_scores = []

    for label in observable:
        if label in available_labels:
            best = max(results[l].get(label, {}).get('r2', 0) for l in layers)
            obs_scores.append(best)

    for label in computed:
        if label in available_labels:
            best = max(results[l].get(label, {}).get('r2', 0) for l in layers)
            comp_scores.append(best)

    x = np.arange(2)
    width = 0.6

    obs_mean = np.mean(obs_scores) if obs_scores else 0
    comp_mean = np.mean(comp_scores) if comp_scores else 0

    bars = ax.bar(x, [obs_mean, comp_mean], width, color=['#4CAF50', '#f44336'])
    ax.set_xticks(x)
    ax.set_xticklabels(['Observable\n(distances from agent)', 'Computed\n(chained distances, totals)'])
    ax.set_ylabel('Mean Best R² Score')
    ax.set_ylim(0, 1)
    ax.set_title('Observable vs Computed Planning Features', fontsize=14, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='R²=0.5 threshold')

    for bar, val in zip(bars, [obs_mean, comp_mean]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'optvar_observable_vs_computed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'optvar_observable_vs_computed.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='fresh_baseline')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=150)
    args = parser.parse_args()

    output_dir = Path('interpretability/results/optvar_probing')
    output_dir.mkdir(parents=True, exist_ok=True)

    env_name = 'PointLtl2-v0'

    print("=" * 70)
    print("PROBING PLANNING REPRESENTATIONS (Optvar Environment)")
    print("=" * 70)
    print(f"Model: {args.exp}")
    print(f"Episodes: {args.n_episodes}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    from sequence.samplers import CurriculumSampler, curricula
    from envs import make_env

    temp_sampler = CurriculumSampler.partial(curricula[env_name])
    temp_env = make_env(env_name, temp_sampler, sequence=True)

    config = model_configs[env_name]
    model_store = ModelStore(env_name, args.exp, args.seed)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()
    model = build_model(temp_env, training_status, config)
    model.eval()

    propositions = list(temp_env.get_propositions())
    temp_env.close()

    # Collect data
    data = collect_probing_data(model, propositions, args.n_episodes, args.max_steps)

    print(f"\nCollected {len(data['agent_pos'])} data points")

    # Save raw data
    np.savez(output_dir / f'probe_data_{args.exp}.npz', **data)
    print(f"Saved: {output_dir / f'probe_data_{args.exp}.npz'}")

    # Run probing
    print("\n" + "=" * 70)
    print("PROBING ANALYSIS")
    print("=" * 70)

    results = run_probing(data, output_dir)

    # Save results
    np.save(output_dir / f'probe_results_{args.exp}.npy', results, allow_pickle=True)

    # Plot
    plot_results(results, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nKey findings:")
    print("- Observable (distances from agent): Should have high R²")
    print("- Computed (chained distances, totals): Low R² indicates myopic behavior")
    print("\nIf the model planned optimally, it would need to encode:")
    print("  1. d(intermediate → goal) for each intermediate")
    print("  2. total_via_intermediate = d(agent→int) + d(int→goal)")
    print("  3. Which intermediate minimizes total path")


if __name__ == '__main__':
    main()
