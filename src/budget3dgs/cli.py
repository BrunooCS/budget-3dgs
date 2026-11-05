"""Command-line entry point: ``train`` one model, ``sweep`` a benchmark, ``report`` results.

Examples:
    budget3dgs train --data data/tandt/truck --output runs/truck/rvr --method rvr --budget 100000
    budget3dgs sweep --config configs/gpu_6gb.json --output runs/gpu_6gb --data_root data
    budget3dgs report runs/gpu_6gb --output docs/assets
"""

import argparse
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from .train import METHODS, TrainConfig


def _train(args):
    from .train import train
    overrides = {f.name: getattr(args, f.name) for f in fields(TrainConfig) if getattr(args, f.name) is not None}
    train(TrainConfig(**overrides))


def _sweep(args):
    """Run every (budget, scene, seed, method) of a sweep config in its own process; completed runs are skipped.

    Budgets are the outer loop so that a complete table for the first budget exists early.
    ``ablations`` are added to ``methods`` only at ``ablation_budgets``.
    """
    spec = json.loads(Path(args.config).read_text())
    root = Path(args.output)
    for budget in spec['budgets']:
        for scene, info in spec['scenes'].items():
            for seed in spec['seeds']:
                for method in spec['methods'] + (spec['ablations'] if budget in spec['ablation_budgets'] else []):
                    out = root / scene / f'{method}-b{budget}-s{seed}'
                    if (out / 'metrics.json').exists():
                        continue
                    cmd = [sys.executable, '-m', 'budget3dgs.cli', 'train', '--data', str(Path(args.data_root) / info['path']),
                           '--output', str(out), '--method', method, '--budget', str(budget), '--seed', str(seed)]
                    # shared training overrides, then per-scene ones (images_dir, downsample)
                    for key, value in (spec['train'] | {k: v for k, v in info.items() if k != 'path'}).items():
                        cmd += [f'--{key}', str(value)]
                    subprocess.run(cmd, check=True)


def _report(args):
    from .report import report
    report(Path(args.runs), Path(args.output))


def main():
    parser = argparse.ArgumentParser(prog='budget3dgs')
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser('train', help='train and evaluate one model')
    for f in fields(TrainConfig):
        kind = (lambda s: s.lower() == 'true') if f.type is bool else f.type
        p.add_argument(f'--{f.name}', type=kind, choices=METHODS if f.name == 'method' else None)
    p.set_defaults(fn=_train)

    p = sub.add_parser('sweep', help='run a benchmark grid from a JSON config')
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--data_root', default='data')
    p.set_defaults(fn=_sweep)

    p = sub.add_parser('report', help='aggregate runs into tables and figures')
    p.add_argument('runs')
    p.add_argument('--output', required=True)
    p.set_defaults(fn=_report)

    args = parser.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
