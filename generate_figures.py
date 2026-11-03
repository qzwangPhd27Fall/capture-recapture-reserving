"""
Generate all paper figures from MC results.
Run after full MC experiment completes.
Usage: python generate_figures.py
"""
import sys, os, json
sys.path.insert(0, 'cr_reserving')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = sns.color_palette('colorblind', 6)
MC = {'closed_cr': COLORS[0], 'open_cr': COLORS[1],
      'min_chisq': COLORS[2], 'doubly_robust': COLORS[3]}
ML = {'closed_cr': 'Schnabel', 'open_cr': 'Open CR',
      'min_chisq': 'Min-ChiSq', 'doubly_robust': 'Doubly-Robust'}

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SID_ORDER = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
METH_ORDER_ALL = ['closed_cr', 'min_chisq', 'doubly_robust', 'open_cr']
METH_ORDER_3 = ['closed_cr', 'min_chisq', 'doubly_robust']


def load_results():
    path = os.path.join(BASE, 'mc_results', 'summary_all.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fall back to per-scenario files
    summaries = {}
    for sid in SID_ORDER:
        p = os.path.join(BASE, 'mc_results', f'summary_{sid}.json')
        if os.path.exists(p):
            with open(p) as f:
                summaries[sid] = json.load(f)
    return summaries if summaries else None


def fig1_coverage_heatmap(summaries):
    """Coverage probability heatmap (method x scenario)."""
    meth_order = METH_ORDER_ALL
    data = np.zeros((len(meth_order), len(SID_ORDER)))
    annot = np.empty((len(meth_order), len(SID_ORDER)), dtype=object)

    for i, m in enumerate(meth_order):
        for j, sid in enumerate(SID_ORDER):
            cov = summaries.get(sid, {}).get(m, {}).get('Cov', np.nan)
            data[i, j] = cov if not np.isnan(cov) else 0
            annot[i, j] = f'{cov:.3f}' if not np.isnan(cov) else '-'

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    for i in range(len(meth_order)):
        for j in range(len(SID_ORDER)):
            v = data[i, j]
            c = 'white' if v < 0.5 or v > 0.9 else 'black'
            ax.text(j, i, annot[i, j], ha='center', va='center', fontsize=9,
                    color=c, fontweight='bold')

    ax.set_xticks(range(len(SID_ORDER)))
    ax.set_xticklabels(SID_ORDER, fontsize=11)
    ax.set_yticks(range(len(meth_order)))
    ax.set_yticklabels([ML.get(m, m) for m in meth_order], fontsize=11)
    ax.set_title('95% CI Coverage Probability', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85)
    path = os.path.join(OUTPUT_DIR, 'fig1_coverage_heatmap.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def fig2_metrics_bars(summaries):
    """Bias, RMSE, Coverage bar charts."""
    meth_order = METH_ORDER_3
    metrics = [('Bias', 'Bias'), ('RMSE', 'RMSE'), ('Cov', 'Coverage')]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(SID_ORDER))
    bw = 0.22

    for ax_i, (key, title) in enumerate(metrics):
        ax = axes[ax_i]
        for m_i, m in enumerate(meth_order):
            vals = []
            for sid in SID_ORDER:
                v = summaries.get(sid, {}).get(m, {}).get(key, np.nan)
                vals.append(v if not np.isnan(v) else 0)
            ax.bar(x + (m_i - 1) * bw, vals, bw,
                   label=ML.get(m, m), color=MC.get(m, COLORS[0]),
                   edgecolor='white', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(SID_ORDER, fontsize=10)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        if title == 'Coverage':
            ax.axhline(y=0.95, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_ylim(0, 1.05)
        if title == 'Bias':
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        if ax_i == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Monte Carlo Results: 7 Scenarios x 500 Reps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2_metrics_bars.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def fig3_dashboard(summaries):
    """4-panel dashboard: Coverage, RMSE, Bias, CI Width."""
    meth_order = METH_ORDER_3
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    x = np.arange(len(SID_ORDER))
    bw = 0.22

    panels = [
        (0, 0, 'Cov', 'Coverage', 0.95),
        (0, 1, 'RMSE', 'RMSE', None),
        (1, 0, 'Bias', 'Bias', 0),
        (1, 1, 'CI_W', 'CI Width', None),
    ]

    for r, c, key, title, ref in panels:
        ax = axes[r, c]
        for m_i, m in enumerate(meth_order):
            vals = []
            for sid in SID_ORDER:
                v = summaries.get(sid, {}).get(m, {}).get(key, np.nan)
                vals.append(v if not np.isnan(v) and v < 5000 else 0)
            ax.bar(x + (m_i - 1) * bw, vals, bw,
                   label=ML.get(m, m), color=MC.get(m, COLORS[0]),
                   edgecolor='white', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(SID_ORDER)
        ax.set_title(f'({chr(97 + 2 * r + c)}) {title}')
        ax.grid(axis='y', alpha=0.3)
        if ref is not None:
            ax.axhline(y=ref, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        if r == 0 and c == 0:
            ax.legend(fontsize=8)

    fig.suptitle('CR Reserving: Monte Carlo Simulation Dashboard', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig3_dashboard.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def fig4_bias_distribution():
    """Distribution of N_hat across reps (requires per-scenario .npz files)."""
    # Load per-scenario estimate arrays
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_flat = axes.flatten()

    for i, sid in enumerate(SID_ORDER):
        ax = axes_flat[i]
        npz_path = os.path.join(BASE, 'mc_results', f'mc_{sid}.npz')
        if not os.path.exists(npz_path):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(sid)
            continue

        data = np.load(npz_path)
        methods = ['closed_cr', 'min_chisq', 'doubly_robust']
        for m_i, m in enumerate(methods):
            key = f'{m}_est'
            if key in data:
                vals = data[key]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    ax.hist(vals, bins=30, alpha=0.5, density=True,
                            label=ML.get(m, m), color=MC.get(m, COLORS[0]))
        ax.axvline(x=500, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title(sid)
        if i == 0:
            ax.legend(fontsize=7)

    # Hide extra subplot
    axes_flat[7].set_visible(False)

    fig.suptitle('Distribution of $\\hat{N}$ Across 500 Monte Carlo Replicates',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4_bias_distribution.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def generate_all():
    summaries = load_results()
    if summaries is None:
        print("ERROR: No MC results found. Run run_full_mc.py first.")
        return

    print(f"Generating figures from {len(summaries)} scenarios...")
    fig1_coverage_heatmap(summaries)
    fig2_metrics_bars(summaries)
    fig3_dashboard(summaries)
    fig4_bias_distribution()
    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    generate_all()
