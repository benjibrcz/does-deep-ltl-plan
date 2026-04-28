#!/usr/bin/env python
"""
Optimality Training Sweep

Systematically trains models with different hyperparameters to investigate
optimal behavior in ZoneEnv.

Key variables:
- Discount factor (gamma): {0.94, 0.97, 0.99, 0.995, 0.998}
- Curriculum variant: {baseline, twostep, mixed}

Usage:
    # Run a single experiment
    python run_optimality_sweep.py --discount 0.95 --curriculum twostep --name my_exp

    # Run full sweep (generates all combinations)
    python run_optimality_sweep.py --sweep

    # Run discount sweep with specific curriculum
    python run_optimality_sweep.py --sweep_discount --curriculum twostep

    # Run curriculum sweep with specific discount
    python run_optimality_sweep.py --sweep_curriculum --discount 0.95
"""
import os
import subprocess
import sys
import json
import itertools
from dataclasses import dataclass, asdict
from datetime import datetime
import simple_parsing


# Sweep configurations
DISCOUNT_VALUES = [0.94, 0.97, 0.99, 0.995, 0.998]
CURRICULUM_VARIANTS = {
    'baseline': 'PointLtl2-v0',
    'twostep': 'PointLtl2-v0.twostep',
    'mixed': 'PointLtl2-v0.mixed',
}


@dataclass
class Args:
    # Experiment identification
    name: str = None  # Auto-generated if not provided
    seed: int = 0

    # Hyperparameters
    discount: float = 0.998  # Default baseline
    curriculum: str = 'baseline'  # baseline, twostep, or mixed
    entropy_coef: float = 0.003
    lr: float = 0.0003

    # Training settings
    num_steps: int = 15_000_000
    num_procs: int = 8
    device: str = 'cpu'

    # Sweep modes
    sweep: bool = False  # Run full sweep (all discount x curriculum combinations)
    sweep_discount: bool = False  # Sweep discount values only
    sweep_curriculum: bool = False  # Sweep curriculum variants only

    # Control
    dry_run: bool = False  # Print commands without executing


def generate_experiment_name(args: Args) -> str:
    """Generate descriptive experiment name from config."""
    parts = [
        f"opt",
        f"d{args.discount}".replace(".", ""),
        args.curriculum,
    ]
    if args.entropy_coef != 0.003:
        parts.append(f"e{args.entropy_coef}".replace(".", ""))
    if args.lr != 0.0003:
        parts.append(f"lr{args.lr}".replace(".", ""))
    return "_".join(parts)


def run_single_experiment(args: Args):
    """Run a single training experiment."""
    if args.name is None:
        args.name = generate_experiment_name(args)

    curriculum_key = CURRICULUM_VARIANTS.get(args.curriculum, args.curriculum)

    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'

    command = [
        sys.executable, 'src/train/train_ppo.py',
        '--env', 'PointLtl2-v0',
        '--steps_per_process', '4096',
        '--batch_size', '2048',
        '--lr', str(args.lr),
        '--discount', str(args.discount),
        '--entropy_coef', str(args.entropy_coef),
        '--log_interval', '1',
        '--save_interval', '2',
        '--epochs', '10',
        '--num_steps', str(args.num_steps),
        '--model_config', 'PointLtl2-v0',
        '--curriculum', curriculum_key,
        '--name', args.name,
        '--seed', str(args.seed),
        '--device', args.device,
        '--num_procs', str(args.num_procs),
        '--log_csv',
    ]

    print(f"\n{'='*60}")
    print(f"Experiment: {args.name}")
    print(f"  discount={args.discount}, curriculum={args.curriculum}")
    print(f"  entropy_coef={args.entropy_coef}, lr={args.lr}")
    print(f"  num_steps={args.num_steps:,}")
    print(f"{'='*60}")

    if args.dry_run:
        print("DRY RUN - would execute:")
        print(" ".join(command))
        return

    # Save experiment config
    exp_dir = f'experiments/ppo/PointLtl2-v0/{args.name}'
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, 'sweep_config.json')
    with open(config_path, 'w') as f:
        config = {
            'name': args.name,
            'discount': args.discount,
            'curriculum': args.curriculum,
            'curriculum_key': curriculum_key,
            'entropy_coef': args.entropy_coef,
            'lr': args.lr,
            'num_steps': args.num_steps,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat(),
        }
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")

    subprocess.run(command, env=env)


def run_sweep(args: Args, discount_values: list, curriculum_variants: list):
    """Run a sweep over multiple configurations."""
    experiments = list(itertools.product(discount_values, curriculum_variants))
    print(f"\nSweep: {len(experiments)} experiments")
    print(f"  Discounts: {discount_values}")
    print(f"  Curricula: {curriculum_variants}")

    for i, (discount, curriculum) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}]")
        exp_args = Args(
            discount=discount,
            curriculum=curriculum,
            entropy_coef=args.entropy_coef,
            lr=args.lr,
            num_steps=args.num_steps,
            num_procs=args.num_procs,
            device=args.device,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        run_single_experiment(exp_args)


def main():
    args = simple_parsing.parse(Args)

    if args.sweep:
        # Full sweep: all discount x curriculum combinations
        run_sweep(args, DISCOUNT_VALUES, list(CURRICULUM_VARIANTS.keys()))
    elif args.sweep_discount:
        # Sweep discount values with fixed curriculum
        run_sweep(args, DISCOUNT_VALUES, [args.curriculum])
    elif args.sweep_curriculum:
        # Sweep curriculum variants with fixed discount
        run_sweep(args, [args.discount], list(CURRICULUM_VARIANTS.keys()))
    else:
        # Single experiment
        run_single_experiment(args)


if __name__ == '__main__':
    main()
