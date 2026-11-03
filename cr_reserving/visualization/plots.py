
"""
Publication-quality figures for the CR reserving paper.
"""
import os, json
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

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_results():
    """Load Monte Carlo summary results."""
    results_dir = os.path.join(BASE, 'mc_results')
    summary_path = os.path.join(results_dir, 'summary_all.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return None


def plot_coverage_heatmap(summaries):
    """Coverage probability heatmap (method x scenario)."""
    sid_order = ['S1','S2','S3','S4','S5','S6','S7']
    meth_order = ['closed_cr','min_chisq','doubly_robust','open_cr']

    data = np.zeros((len(meth_order), len(sid_order)))
    annot = np.empty((len(meth_order), len(sid_order)), dtype=object)

    for i, m in enumerate(meth_order):
        for j, sid in enumerate(sid_order):
            cov = summaries.get(sid,{}).get(m,{}).get('Cov', np.nan)
            data[i,j] = cov if not np.isnan(cov) else 0
            annot[i,j] = f'{cov:.3f}' if not np.isnan(cov) else '-'

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    for i in range(len(meth_order)):
        for j in range(len(sid_order)):
            v = data[i,j]
            c = 'white' if v < 0.5 or v > 0.9 else 'black'
            ax.text(j, i, annot[i,j], ha='center', va='center', fontsize=9,
                   color=c, fontweight='bold')

    ax.set_xticks(range(len(sid_order))); ax.set_xticklabels(sid_order, fontsize=11)
    ax.set_yticks(range(len(meth_order))); ax.set_yticklabels([ML.get(m,m) for m in meth_order], fontsize=11)
    ax.set_title('95% CI Coverage Probability', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85)
    path = os.path.join(OUTPUT_DIR, 'fig_coverage_heatmap.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_metrics_bars(summaries):
    """Side-by-side bar charts for Bias, RMSE, Coverage."""
    sid_order = ['S1','S2','S3','S4','S5','S6','S7']
    meth_order = ['closed_cr','min_chisq','doubly_robust']
    metrics = [('Bias','Bias'), ('RMSE','RMSE'), ('Cov','Coverage')]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(sid_order)); bw = 0.22

    for ax_i, (key, title) in enumerate(metrics):
        ax = axes[ax_i]
        for m_i, m in enumerate(meth_order):
            vals = []
            for sid in sid_order:
                v = summaries.get(sid,{}).get(m,{}).get(key, np.nan)
                vals.append(v if not np.isnan(v) else 0)
            ax.bar(x + (m_i-1)*bw, vals, bw,
                   label=ML.get(m,m), color=MC.get(m,COLORS[0]),
                   edgecolor='white', linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(sid_order, fontsize=10)
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
    path = os.path.join(OUTPUT_DIR, 'fig_metrics_bars.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_dashboard(summaries):
    """4-panel dashboard."""
    sid_order = ['S1','S2','S3','S4','S5','S6','S7']
    meth_order = ['closed_cr','min_chisq','doubly_robust']

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    x = np.arange(len(sid_order)); bw = 0.22

    panels = [
        (0,0, 'Cov', 'Coverage', 0.95),
        (0,1, 'RMSE', 'RMSE', None),
        (1,0, 'Bias', 'Bias', 0),
        (1,1, 'CI_W', 'CI Width', None),
    ]

    for r, c, key, title, ref in panels:
        ax = axes[r,c]
        for m_i, m in enumerate(meth_order):
            vals = []
            for sid in sid_order:
                v = summaries.get(sid,{}).get(m,{}).get(key, np.nan)
                vals.append(v if not np.isnan(v) and v < 5000 else 0)
            ax.bar(x + (m_i-1)*bw, vals, bw,
                   label=ML.get(m,m), color=MC.get(m,COLORS[0]),
                   edgecolor='white', linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(sid_order)
        ax.set_title(f'({chr(97+2*r+c)}) {title}')
        ax.grid(axis='y', alpha=0.3)
        if ref is not None:
            ax.axhline(y=ref, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        if r==0 and c==0: ax.legend(fontsize=8)

    fig.suptitle('CR Reserving: Monte Carlo Simulation Dashboard', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_dashboard.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def generate_all():
    """Generate all figures."""
    summaries = load_results()
    if summaries is None:
        print('No MC results found. Using dummy data for testing.')
        summaries = {
            'S1': {'closed_cr': {'Bias': 5, 'RMSE': 25, 'Cov': 0.94, 'CI_W': 100},
                   'min_chisq': {'Bias': -10, 'RMSE': 30, 'Cov': 0.96, 'CI_W': 120},
                   'doubly_robust': {'Bias': -8, 'RMSE': 28, 'Cov': 0.95, 'CI_W': 110}}}
    plot_coverage_heatmap(summaries)
    plot_metrics_bars(summaries)
    plot_dashboard(summaries)
    return summaries


if __name__ == '__main__':
    generate_all()
