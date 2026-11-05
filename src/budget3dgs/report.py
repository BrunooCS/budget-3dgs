"""Aggregate sweep runs into a CSV, Markdown tables and figures.

Expects ``runs/<scene>/<method>-b<budget>-s<seed>/metrics.json``. Writes into
``output``: ``results.csv`` (one row per run), ``results.md`` (dataset averages,
paired Budget-3DGS - MCMC differences, ablations, per-scene tables), ``budget_curves.png``
and ``qualitative.png``.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATASETS = {'Mip-NeRF 360': ['bicycle', 'garden', 'stump', 'room', 'counter', 'kitchen', 'bonsai'],
            'Tanks & Temples': ['truck', 'train'], 'Deep Blending': ['drjohnson', 'playroom']}
LABELS = {'mcmc': '3DGS-MCMC', 'rvr': 'Budget-3DGS', 'abl-opacity': 'Opacity targets (control)',
          'abl-error': 'L1-error targets', 'abl-recycle': 'Budget-3DGS + visibility recycling'}
KEYS = ('psnr', 'ssim', 'lpips', 'train_time_s', 'fps', 'peak_memory_gb')


def _load(runs):
    rows = []
    for p in sorted(runs.glob('*/*/metrics.json')):
        r = json.loads(p.read_text())
        rows.append({k: v for k, v in r.items() if k != 'per_view'} | {'dir': str(p.parent)})
    return rows


def _means(rows):
    """{(scene, budget, method): {metric: mean over seeds, 'psnr_std': PSNR std over seeds, 'seeds': count}}."""
    groups = defaultdict(list)
    for r in rows:
        groups[r['scene'], r['budget'], r['method']].append(r)
    std = lambda v: float(np.std([r['psnr'] for r in v], ddof=1)) if len(v) > 1 else float('nan')
    return {k: {m: float(np.mean([r[m] for r in v])) for m in KEYS} | {'psnr_std': std(v), 'seeds': len(v)} for k, v in groups.items()}


def _complete(means, scenes, budget, methods):
    return [s for s in scenes if all((s, budget, m) in means for m in methods)]


# ── Tables ───────────────────────────────────────────

def _main_table(means, methods, budgets):
    head = '| Budget | Method | ' + ' | '.join(f'{d} PSNR↑ / SSIM↑ / LPIPS↓' for d in DATASETS) + ' |'
    lines = [head, '|---:|---|' + '---:|' * len(DATASETS)]
    for b in budgets:
        for m in methods:
            cells = []
            for scenes in DATASETS.values():
                done = _complete(means, scenes, b, methods)
                if len(done) < len(scenes):
                    cells.append(f'({len(done)}/{len(scenes)} scenes)')
                    continue
                avg = {k: np.mean([means[s, b, m][k] for s in scenes]) for k in ('psnr', 'ssim', 'lpips')}
                cells.append(f"{avg['psnr']:.2f} / {avg['ssim']:.3f} / {avg['lpips']:.3f}")
            lines.append(f'| {b // 1000}K | {LABELS[m]} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def _paired_table(means, budgets):
    """Per budget: mean Budget-3DGS - MCMC difference over all scenes, the number of scenes it wins, and the mean
    per-scene PSNR standard deviation across seeds (pooled over both methods) as a noise reference."""
    lines = ['| Budget | Scenes | Seeds | ΔPSNR (dB) | PSNR wins | Seed std PSNR | ΔSSIM | SSIM wins | ΔLPIPS | LPIPS wins |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in budgets:
        scenes = _complete(means, sum(DATASETS.values(), []), b, ['mcmc', 'rvr'])
        if not scenes:
            continue
        d = {k: np.array([means[s, b, 'rvr'][k] - means[s, b, 'mcmc'][k] for s in scenes]) for k in ('psnr', 'ssim', 'lpips')}
        n, seeds = len(scenes), min(means[s, b, m]['seeds'] for s in scenes for m in ('mcmc', 'rvr'))
        std = np.mean([means[s, b, m]['psnr_std'] for s in scenes for m in ('mcmc', 'rvr')])
        lines.append(f"| {b // 1000}K | {n} | {seeds} | {d['psnr'].mean():+.3f} | {(d['psnr'] > 0).sum()}/{n} | "
                     f"{'–' if np.isnan(std) else f'{std:.3f}'} | {d['ssim'].mean():+.4f} | "
                     f"{(d['ssim'] > 0).sum()}/{n} | {d['lpips'].mean():+.4f} | {(d['lpips'] < 0).sum()}/{n} |")
    return '\n'.join(lines)


def _ablation_table(means, budget, methods):
    scenes = _complete(means, sum(DATASETS.values(), []), budget, methods)
    lines = [f'Budget {budget // 1000}K, {len(scenes)} scenes, seed 0 for every variant.', '',
             '| Variant | Parent distribution | PSNR↑ | SSIM↑ | LPIPS↓ | ΔPSNR vs MCMC |', '|---|---|---:|---:|---:|---:|']
    rule = {'mcmc': 'σ(o) (gsplat)', 'abl-opacity': 'σ(o) (this repo)', 'abl-error': 'Σ w·|r|', 'rvr': 'Σ w·Var[r]',
            'abl-recycle': 'Σ w·Var[r] + 1% low-visibility donors'}
    for m in methods:
        avg = {k: np.mean([means[s, budget, m][k] for s in scenes]) for k in ('psnr', 'ssim', 'lpips')}
        delta = np.mean([means[s, budget, m]['psnr'] - means[s, budget, 'mcmc']['psnr'] for s in scenes])
        lines.append(f"| {LABELS[m]} | {rule[m]} | {avg['psnr']:.2f} | {avg['ssim']:.3f} | {avg['lpips']:.3f} | {delta:+.3f} |")
    return '\n'.join(lines)


def _scene_table(means, methods, budgets, key, fmt):
    scenes = [s for s in sum(DATASETS.values(), []) if any((s, b, m) in means for b in budgets for m in methods)]
    lines = ['| Budget | Method | ' + ' | '.join(scenes) + ' |', '|---:|---|' + '---:|' * len(scenes)]
    for b in budgets:
        for m in methods:
            lines.append(f'| {b // 1000}K | {LABELS[m]} | ' + ' | '.join(
                format(means[s, b, m][key], fmt) if (s, b, m) in means else '–' for s in scenes) + ' |')
    return '\n'.join(lines)


# ── Figures ──────────────────────────────────────────

INK, MUTED, HAIR, BLUE, SLATE = '#191b21', '#626b7a', '#dfe3ea', '#164be6', '#9ba6b7'
PLY_BYTES = 248  # float32 PLY with SH degree 3: 62 values per Gaussian


def _style(ax):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=0, labelsize=8.5)
    ax.grid(axis='y', color=HAIR, lw=0.8)
    ax.set_axisbelow(True)


def _curves(means, budgets, path):
    """PSNR vs budget per dataset; x ticks carry the matching PLY size."""
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 3.3))
    for ax, (name, scenes) in zip(axes, DATASETS.items()):
        bs = [b for b in budgets if len(_complete(means, scenes, b, ['mcmc', 'rvr'])) == len(scenes)]
        x = np.array(bs) / 1000
        y = {m: np.array([np.mean([means[s, b, m]['psnr'] for s in scenes]) for b in bs]) for m in ('mcmc', 'rvr')}
        ax.fill_between(x, y['mcmc'], y['rvr'], color=BLUE, alpha=0.08, lw=0)
        for m, color, lw in (('mcmc', SLATE, 1.8), ('rvr', BLUE, 2.2)):
            ax.plot(x, y[m], 'o-', color=color, lw=lw, ms=5, mec='white', mew=1.2, zorder=3)
            ax.annotate(LABELS[m], (x[-1], y[m][-1]), xytext=(7, 0), textcoords='offset points', va='center',
                        color=color, fontsize=8.5, fontweight='bold' if m == 'rvr' else 'normal')
        for xi, a, b in zip(x, y['mcmc'], y['rvr']):
            ax.annotate(f'+{b - a:.2f} dB', (xi, (a + b) / 2), xytext=(6, 0), textcoords='offset points', va='center',
                        color=BLUE, fontsize=7.5, zorder=4, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=HAIR, lw=0.6))
        _style(ax)
        ax.set_xscale('log')
        ax.set_xticks(x, [f'{b // 1000}K\n{b * PLY_BYTES / 1e6:.0f} MB' for b in bs])
        ax.minorticks_off()
        ax.set_xlim(x[0] / 1.25, x[-1] * 2.1)
        ax.set_title(f'{name}  ·  {len(scenes)} scenes', fontsize=10, color=INK, loc='left')
    axes[0].set_ylabel('Test PSNR (dB)', color=MUTED, fontsize=9)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)


def _best_window(err_a, err_b, size):
    """Top-left corner of the size x size window where err_a - err_b is largest (box filter on a summed-area table)."""
    d = np.pad((err_a - err_b).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    box = d[size:, size:] - d[:-size, size:] - d[size:, :-size] + d[:-size, :-size]
    return np.unravel_index(box.argmax(), box.shape)


def _qualitative(rows, scenes, budget, path, view=0):
    """Held-out view per scene with a zoom on the window where Budget-3DGS reduces the error most."""
    by_key = {(r['scene'], r['method'], r['budget'], r['seed']): r for r in rows}
    load = lambda p: np.asarray(Image.open(p)).astype(np.float32) / 255
    fig, axes = plt.subplots(len(scenes), 4, figsize=(11, 2.45 * len(scenes)), squeeze=False,
                             gridspec_kw={'width_ratios': [1.5, 1, 1, 1]})
    for i, s in enumerate(scenes):
        runs = {m: Path(by_key[s, m, budget, 0]['dir']) for m in ('mcmc', 'rvr')}
        gt = load(runs['mcmc'] / 'renders' / f'gt_{view:03d}.png')
        img = {m: load(runs[m] / 'renders' / f'{view:03d}.png') for m in runs}
        size = min(gt.shape[:2]) // 4
        y, x = _best_window(np.abs(img['mcmc'] - gt).mean(-1), np.abs(img['rvr'] - gt).mean(-1), size)
        scores = {m: json.loads((runs[m] / 'metrics.json').read_text())['per_view'][view][0] for m in runs}

        axes[i, 0].imshow(gt)
        axes[i, 0].add_patch(plt.Rectangle((x, y), size, size, fill=False, ec=BLUE, lw=1.6))
        axes[i, 0].set_title(s, fontsize=9.5, color=INK, loc='left')
        panels = [('Ground truth', gt, INK), (f"3DGS-MCMC · {scores['mcmc']:.2f} dB", img['mcmc'], MUTED),
                  (f"Budget-3DGS · {scores['rvr']:.2f} dB", img['rvr'], BLUE)]
        for ax, (title, im, color) in zip(axes[i, 1:], panels):
            ax.imshow(im[y:y + size, x:x + size])
            ax.set_title(title, fontsize=9, color=color)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_color(HAIR)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    fig.savefig(path, dpi=130, facecolor='white')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────

def report(runs: Path, output: Path, qualitative_scenes=('garden', 'kitchen', 'bonsai', 'stump')):
    """Write results.csv, results.md, budget_curves.png and qualitative.png into ``output``."""
    output.mkdir(parents=True, exist_ok=True)
    rows = _load(runs)
    with (output / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([r | {'dir': str(Path(r['dir']).relative_to(runs))} for r in rows])

    means = _means(rows)
    budgets = sorted({k[1] for k in means})
    ablation = [m for m in LABELS if m != 'rvr' and m != 'mcmc' and any(k[2] == m for k in means)]
    ablation_budget = next(b for b in budgets if any((k[1], k[2]) == (b, ablation[0]) for k in means))
    sections = [
        '## Dataset averages\n', _main_table(means, ['mcmc', 'rvr'], budgets),
        '\n## Budget-3DGS - MCMC, paired over scenes\n', _paired_table(means, budgets),
        '\n## Ablations\n', _ablation_table(_means([r for r in rows if r['seed'] == 0]), ablation_budget, ['mcmc', 'abl-opacity', 'abl-error', 'rvr', 'abl-recycle']),
        '\n## Per-scene PSNR\n', _scene_table(means, ['mcmc', 'rvr'], budgets, 'psnr', '.2f'),
        '\n## Per-scene LPIPS\n', _scene_table(means, ['mcmc', 'rvr'], budgets, 'lpips', '.3f'),
        '\n## Training time (s)\n', _scene_table(means, ['mcmc', 'rvr'], budgets, 'train_time_s', '.0f'),
    ]
    (output / 'results.md').write_text('\n'.join(sections) + '\n')
    _curves(means, budgets, output / 'budget_curves.png')
    _qualitative(rows, [s for s in qualitative_scenes if (s, budgets[0], 'rvr') in means], budgets[0], output / 'qualitative.png')
