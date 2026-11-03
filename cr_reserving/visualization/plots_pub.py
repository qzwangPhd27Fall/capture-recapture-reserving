"""
Publication-quality figures for CR Reserving paper.
Target: JASA / Biometrika / IME level.

Style: clean, serif typography, muted professional palette,
multi-panel composites, no chartjunk.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from typing import Optional, Dict, List
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ──────────────────────────────────────────────────────────
# Global style setup — Nature / Science inspired
# ──────────────────────────────────────────────────────────

def set_pub_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIXGeneral', 'Palatino Linotype'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8,
        'legend.title_fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

# Muted professional palette — inspired by Economist / Nature
PALETTE = {
    'closed_cr':   '#1B6CA8',   # steel blue
    'open_cr':     '#2A9D8F',   # teal
    'min_chisq_cr':'#E9A820',   # warm gold
    'hmm_cr':      '#7B2D8E',   # plum
    'bayesian_quick':'#C44E52', # muted red
    'chain_ladder':'#8B8B8B',   # neutral grey
}

PALETTE_LIGHT = {k: v + '40' for k, v in PALETTE.items()}

METHOD_LABELS = {
    'closed_cr': 'CR (Closed)',
    'open_cr': 'CR (Open)',
    'min_chisq_cr': r'CR (Min-$\chi^2$)',
    'hmm_cr': 'CR (HMM)',
    'bayesian_quick': 'CR (Bayesian)',
    'chain_ladder': 'Chain-Ladder',
}

METHOD_MARKERS = {
    'closed_cr': 'o',
    'open_cr': 's',
    'min_chisq_cr': 'D',
    'hmm_cr': '^',
    'bayesian_quick': 'v',
    'chain_ladder': 'X',
}

SCENARIO_LABELS = {
    'S1': 'S1\nClosed\nHomog.',
    'S2': 'S2\nClosed\nHetero.',
    'S3': 'S3\nOpen\nStable',
    'S4': 'S4\nOpen\nShock',
    'S5': 'S5\nShort\nTail',
    'S6': 'S6\nLong\nTail',
    'S7': 'S7\nHigh-dim\nHetero.',
}

SCENARIO_SHORT = {
    'S1': 'S1', 'S2': 'S2', 'S3': 'S3', 'S4': 'S4',
    'S5': 'S5', 'S6': 'S6', 'S7': 'S7',
}


def _label(method):
    return METHOD_LABELS.get(method, method)

def _color(method):
    return PALETTE.get(method, '#555555')

def _marker(method):
    return METHOD_MARKERS.get(method, 'o')


# ──────────────────────────────────────────────────────────
# Figure 1: Four-panel composite (Bias / RMSE / Coverage / CI Width)
# ──────────────────────────────────────────────────────────

def fig_four_panel(results_df: pd.DataFrame,
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    2x2 panel: grouped dot-and-line plots for Bias, RMSE, Coverage, CI Width.
    """
    set_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)
    metrics = [('bias', 'Bias', None), ('rmse', 'RMSE', None),
               ('coverage', 'Coverage', 0.95), ('ci_width', 'CI Width', None)]
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    scenarios = results_df['scenario'].unique()
    methods = results_df['method'].unique()
    x = np.arange(len(scenarios))

    for idx, (ax, (metric, ylabel, refline)) in enumerate(zip(axes.flat, metrics)):
        for j, method in enumerate(methods):
            vals = []
            for s in scenarios:
                row = results_df[(results_df['scenario'] == s) & (results_df['method'] == method)]
                vals.append(row[metric].values[0] if len(row) > 0 else np.nan)
            offset = (j - len(methods) / 2 + 0.5) * 0.12
            ax.plot(x + offset, vals, marker=_marker(method), color=_color(method),
                    label=_label(method), markersize=5, linewidth=0.8,
                    markeredgecolor='white', markeredgewidth=0.3, zorder=3)

        if refline is not None:
            ax.axhline(refline, color='#CC0000', linestyle='--', linewidth=0.7,
                       alpha=0.7, zorder=1)
            if metric == 'coverage':
                ax.axhspan(0.93, 0.97, color='#CC0000', alpha=0.04, zorder=0)

        if metric == 'bias':
            ax.axhline(0, color='grey', linestyle='-', linewidth=0.4, alpha=0.5, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_SHORT.get(s, s) for s in scenarios])
        ax.set_ylabel(ylabel)
        ax.text(-0.12, 1.05, panel_labels[idx], transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

        if metric == 'coverage':
            ax.set_ylim(-0.05, 1.05)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(methods), 4),
               frameon=False, bbox_to_anchor=(0.5, -0.02), columnspacing=1.5)

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        _save_pdf(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────
# Figure 2: Violin + strip plot (estimate distributions)
# ──────────────────────────────────────────────────────────

def fig_violin_strip(mc_results: Dict, N_true: int,
                     scenario_name: str = '',
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Half-violin + jittered strip plot for one scenario's estimates.
    """
    set_pub_style()
    methods = list(mc_results.keys())
    n_m = len(methods)

    fig, ax = plt.subplots(figsize=(max(3.5, 1.2 * n_m), 4))

    data_frames = []
    for method in methods:
        mc = mc_results[method]
        n_show = min(len(mc.estimates), 200)
        idx = np.random.choice(len(mc.estimates), n_show, replace=False)
        df_tmp = pd.DataFrame({
            'Method': _label(method),
            'Estimate': mc.estimates[idx],
            'method_key': method,
        })
        data_frames.append(df_tmp)
    plot_df = pd.concat(data_frames, ignore_index=True)

    order = [_label(m) for m in methods]
    palette_mapped = {_label(m): _color(m) for m in methods}

    parts = ax.violinplot(
        [mc_results[m].estimates for m in methods],
        positions=range(n_m), showextrema=False, showmedians=False, widths=0.7
    )
    for i, (pc, method) in enumerate(zip(parts['bodies'], methods)):
        pc.set_facecolor(_color(method))
        pc.set_alpha(0.25)
        pc.set_edgecolor(_color(method))
        pc.set_linewidth(0.5)

    for i, method in enumerate(methods):
        est = mc_results[method].estimates
        n_show = min(len(est), 80)
        idx = np.random.RandomState(42).choice(len(est), n_show, replace=False)
        jitter = np.random.RandomState(42).uniform(-0.12, 0.12, n_show)
        ax.scatter(i + jitter, est[idx], s=4, color=_color(method),
                   alpha=0.35, edgecolors='none', zorder=2)
        q25, med, q75 = np.percentile(est, [25, 50, 75])
        ax.hlines(med, i - 0.2, i + 0.2, color=_color(method),
                  linewidth=1.5, zorder=4)
        ax.vlines(i, q25, q75, color=_color(method), linewidth=2.5,
                  alpha=0.6, zorder=3)

    ax.axhline(N_true, color='#CC0000', linestyle='--', linewidth=0.8,
               label=f'True $N = {N_true}$', zorder=5)

    ax.set_xticks(range(n_m))
    ax.set_xticklabels([_label(m) for m in methods], rotation=20, ha='right')
    ax.set_ylabel(r'Estimated $\hat{N}$')
    if scenario_name:
        ax.set_title(scenario_name, fontsize=10, pad=8)
    ax.legend(loc='upper right', frameon=False, fontsize=7.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        _save_pdf(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────
# Figure 3: Multi-scenario violin grid
# ──────────────────────────────────────────────────────────

def fig_violin_grid(mc_results_by_scenario: Dict,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Grid of violin plots: one column per scenario, shared y-axis.
    """
    set_pub_style()
    scenarios = list(mc_results_by_scenario.keys())
    n_sc = len(scenarios)
    n_cols = min(4, n_sc)
    n_rows = (n_sc + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 3.5 * n_rows),
                             constrained_layout=True, sharey=False)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    panel_labels = [chr(ord('a') + i) for i in range(n_sc)]

    for i, sid in enumerate(scenarios):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        mc_results = mc_results_by_scenario[sid]
        methods = list(mc_results.keys())
        N_true = list(mc_results.values())[0].true_N

        parts = ax.violinplot(
            [mc_results[m].estimates for m in methods],
            positions=range(len(methods)), showextrema=False,
            showmedians=False, widths=0.7
        )
        for j, (pc, method) in enumerate(zip(parts['bodies'], methods)):
            pc.set_facecolor(_color(method))
            pc.set_alpha(0.3)
            pc.set_edgecolor(_color(method))
            pc.set_linewidth(0.4)

        for j, method in enumerate(methods):
            est = mc_results[method].estimates
            med = np.median(est)
            q25, q75 = np.percentile(est, [25, 75])
            ax.hlines(med, j - 0.18, j + 0.18, color=_color(method),
                      linewidth=1.2, zorder=4)
            ax.vlines(j, q25, q75, color=_color(method),
                      linewidth=2, alpha=0.6, zorder=3)

        ax.axhline(N_true, color='#CC0000', linestyle='--', linewidth=0.6, zorder=1)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([_label(m).replace(' ', '\n') for m in methods],
                           fontsize=6.5, rotation=0)
        ax.set_title(f'({panel_labels[i]}) {sid}', fontsize=9, pad=4)
        if c == 0:
            ax.set_ylabel(r'$\hat{N}$')

    for i in range(n_sc, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        _save_pdf(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────
# Figure 4: Enhanced convergence plot
# ──────────────────────────────────────────────────────────

def fig_convergence(capture_histories: np.ndarray, N_true: int,
                    save_path: Optional[str] = None,
                    title: str = '') -> plt.Figure:
    """
    Convergence of CR estimator with shaded 95% CI band.
    Professional single-panel design.
    """
    set_pub_style()
    n, D = capture_histories.shape

    seen = set()
    Y_list, x_list = [], []
    estimates, se_list = [], []

    for d in range(D):
        active = np.where(capture_histories[:, d] == 1)[0]
        for j in active:
            is_recap = 1 if j in seen else 0
            Y_list.append(len(seen))
            x_list.append(is_recap)
            seen.add(j)
            sum_Y = sum(Y_list)
            sum_x = sum(x_list)
            N_h = sum_Y / sum_x if sum_x > 0 else float('nan')
            estimates.append(N_h)

            if sum_x > 0 and not np.isnan(N_h):
                Y_arr = np.array(Y_list, dtype=float)
                q = Y_arr / max(N_h, 1)
                q = np.clip(q, 0, 1)
                sq = q.sum()
                if sq > 0:
                    var = N_h ** 2 * np.sum(q * (1 - q)) / sq ** 2
                    se_list.append(np.sqrt(max(var, 0)))
                else:
                    se_list.append(0)
            else:
                se_list.append(0)

    m = len(estimates)
    x_axis = np.arange(1, m + 1)
    est = np.array(estimates)
    se = np.array(se_list)
    valid = ~np.isnan(est)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.fill_between(x_axis[valid],
                    est[valid] - 1.96 * se[valid],
                    est[valid] + 1.96 * se[valid],
                    color=PALETTE['closed_cr'], alpha=0.12, zorder=1,
                    label='95% CI')
    ax.plot(x_axis[valid], est[valid], color=PALETTE['closed_cr'],
            linewidth=1.0, zorder=2, label=r'$\hat{N}$')
    ax.axhline(N_true, color='#CC0000', linestyle='--', linewidth=0.8,
               zorder=3, label=f'True $N = {N_true}$')

    ax.set_xlabel('Number of capture events')
    ax.set_ylabel(r'Population estimate $\hat{N}$')
    if title:
        ax.set_title(title, fontsize=10, pad=6)

    ax.legend(frameon=False, loc='upper right', fontsize=8)

    ymin = max(0, N_true - 3 * np.nanstd(est[valid]))
    ymax = N_true + 3 * np.nanstd(est[valid])
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        _save_pdf(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────
# Figure 5: Radar / spider chart (multi-metric comparison)
# ──────────────────────────────────────────────────────────

def fig_radar(results_df: pd.DataFrame,
              scenario: str = 'S1',
              save_path: Optional[str] = None) -> plt.Figure:
    """
    Radar chart comparing methods on 4 normalized metrics for one scenario.
    """
    set_pub_style()
    sc_df = results_df[results_df['scenario'] == scenario].copy()
    if len(sc_df) == 0:
        return plt.figure()

    metrics = ['bias', 'rmse', 'coverage', 'ci_width']
    display = ['|Bias|', 'RMSE', '1 - Coverage', 'CI Width']

    sc_df['bias'] = sc_df['bias'].abs()
    sc_df['coverage'] = 1 - sc_df['coverage']

    methods = sc_df['method'].unique()
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    max_vals = sc_df[metrics].max()
    max_vals = max_vals.replace(0, 1)

    for method in methods:
        row = sc_df[sc_df['method'] == method]
        if len(row) == 0:
            continue
        vals = [row[m].values[0] / max_vals[m] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', color=_color(method), label=_label(method),
                markersize=4, linewidth=1.0, markeredgecolor='white',
                markeredgewidth=0.3)
        ax.fill(angles, vals, color=_color(method), alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display, fontsize=8)
    ax.set_yticklabels([])
    ax.set_rlabel_position(30)
    ax.spines['polar'].set_linewidth(0.4)
    ax.grid(linewidth=0.3, alpha=0.5)

    ax.set_title(f'Scenario {scenario}', fontsize=10, pad=15)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0), frameon=False, fontsize=7.5)

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        _save_pdf(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────
# Figure 6: CR advantage heatmap (paired difference vs CL)
# ──────────────────────────────────────────────────────────

def fig_advantage_heatmap(results_df: pd.DataFrame,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Heatmap showing RMSE reduction (%) of each CR method vs Chain-Ladder.
    Green = CR wins, Red = CL wins.
    """
    set_pub_style()
    scenarios = results_df['scenario'].unique()
    cr_methods = [m for m in results_df['method'].unique() if m != 'chain_ladder']

    cl_rmse = {}
    for s in scenarios:
        row = results_df[(results_df['scenario'] == s) & (results_df['method'] == 'chain_ladder')]
        cl_rmse[s] = row['rmse'].values[0] if len(row) > 0 else np.nan

    matrix = np.full((len(cr_methods), len(scenarios)), np.nan)
    for i, method in enumerate(cr_methods):
        for j, s in enumerate(scenarios):
            row = results_df[(results_df['scenario'] == s) & (results_df['method'] == method)]
            if len(row) > 0 and not np.isnan(cl_rmse.get(s, np.nan)):
                cr_rmse = row['rmse'].values[0]
                reduction = (cl_rmse[s] - cr_rmse) / cl_rmse[s] * 100
                matrix[i, j] = reduction

    fig, ax = plt.subplots(figsize=(6, 2.5 + 0.4 * len(cr_methods)))

    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios)
    ax.set_yticks(range(len(cr_methods)))
    ax.set_yticklabels([_label(m) for m in cr_methods])

    for i in range(len(cr_methods)):
        for j in range(len(scenarios)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:+.0f}%', ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)

    ax.set_title('RMSE Reduction vs. Chain-Ladder (%)', fontsize=10, pad=8)
    ax.set_xlabel('Scenario')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Reduction (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        _save_pdf(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────
# Figure 7: Coverage dot plot with Wilson interval
# ──────────────────────────────────────────────────────────

def fig_coverage_dot(results_df: pd.DataFrame,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Cleveland dot plot of coverage probabilities with nominal band.
    """
    set_pub_style()
    scenarios = results_df['scenario'].unique()
    methods = results_df['method'].unique()

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.axhspan(0.925, 0.975, color='#CC0000', alpha=0.06, zorder=0)
    ax.axhline(0.95, color='#CC0000', linestyle='--', linewidth=0.6,
               alpha=0.6, zorder=1, label='Nominal 95%')

    for method in methods:
        coverages = []
        for s in scenarios:
            row = results_df[(results_df['scenario'] == s) & (results_df['method'] == method)]
            coverages.append(row['coverage'].values[0] if len(row) > 0 else np.nan)
        ax.plot(range(len(scenarios)), coverages, marker=_marker(method),
                color=_color(method), label=_label(method), linewidth=0.8,
                markersize=6, markeredgecolor='white', markeredgewidth=0.3,
                zorder=3)

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('Coverage probability')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Scenario')

    ax.legend(frameon=False, loc='lower left', fontsize=7.5, ncol=2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        _save_pdf(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────
# Figure 8: Bias–RMSE scatter (efficiency frontier)
# ──────────────────────────────────────────────────────────

def fig_bias_rmse_scatter(results_df: pd.DataFrame,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Scatter of |Bias| vs RMSE across all scenario-method combinations.
    Points near the diagonal = low variance; far above = high variance.
    """
    set_pub_style()
    fig, ax = plt.subplots(figsize=(5, 4.2))

    for _, row in results_df.iterrows():
        method = row['method']
        scenario = row['scenario']
        absbias = abs(row['bias'])
        rmse = row['rmse']
        ax.scatter(absbias, rmse, color=_color(method), marker=_marker(method),
                   s=50, edgecolors='white', linewidths=0.3, zorder=3)
        ax.annotate(scenario, (absbias, rmse), fontsize=5.5, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')

    max_val = max(results_df['rmse'].max(), results_df['bias'].abs().max()) * 1.1
    diag = np.linspace(0, max_val, 100)
    ax.plot(diag, diag, 'k:', linewidth=0.5, alpha=0.4, label='|Bias| = RMSE')

    methods = results_df['method'].unique()
    handles = [plt.Line2D([0], [0], marker=_marker(m), color='w',
               markerfacecolor=_color(m), markersize=7, label=_label(m))
               for m in methods]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc='upper left')

    ax.set_xlabel('|Bias|')
    ax.set_ylabel('RMSE')
    ax.set_title('Bias-RMSE Frontier', fontsize=10, pad=6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        _save_pdf(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────
# Figure 9: RMSE heatmap (publication-quality)
# ──────────────────────────────────────────────────────────

def fig_rmse_heatmap(results_df: pd.DataFrame,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Clean heatmap of RMSE with diverging colourmap.
    """
    set_pub_style()
    pivot = results_df.pivot(index='scenario', columns='method', values='rmse')
    col_order = [m for m in ['closed_cr', 'open_cr', 'min_chisq_cr', 'hmm_cr',
                             'bayesian_quick', 'chain_ladder'] if m in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([_label(m) for m in col_order], rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(col_order)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                thresh = np.nanmean(pivot.values)
                color = 'white' if val > thresh else '#333333'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)

    ax.set_title('RMSE by Scenario and Method', fontsize=10, pad=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('RMSE', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.tick_params(top=False, bottom=True, left=True, right=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        _save_pdf(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────
# Figure 10: Paired lollipop (RMSE: CR vs CL)
# ──────────────────────────────────────────────────────────

def fig_lollipop(results_df: pd.DataFrame,
                 cr_method: str = 'closed_cr',
                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Horizontal lollipop chart showing RMSE of CR vs Chain-Ladder per scenario.
    """
    set_pub_style()
    scenarios = results_df['scenario'].unique()

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    y = np.arange(len(scenarios))

    cr_vals, cl_vals = [], []
    for s in scenarios:
        row_cr = results_df[(results_df['scenario'] == s) & (results_df['method'] == cr_method)]
        row_cl = results_df[(results_df['scenario'] == s) & (results_df['method'] == 'chain_ladder')]
        cr_vals.append(row_cr['rmse'].values[0] if len(row_cr) > 0 else np.nan)
        cl_vals.append(row_cl['rmse'].values[0] if len(row_cl) > 0 else np.nan)

    cr_vals = np.array(cr_vals)
    cl_vals = np.array(cl_vals)

    for i in range(len(scenarios)):
        ax.plot([cr_vals[i], cl_vals[i]], [y[i], y[i]], '-',
                color='#CCCCCC', linewidth=1.5, zorder=1)

    ax.scatter(cr_vals, y, color=_color(cr_method), s=60, zorder=3,
               label=_label(cr_method), marker='o', edgecolors='white', linewidths=0.5)
    ax.scatter(cl_vals, y, color=_color('chain_ladder'), s=60, zorder=3,
               label=_label('chain_ladder'), marker='X', edgecolors='white', linewidths=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(scenarios)
    ax.set_xlabel('RMSE')
    ax.set_title(f'{_label(cr_method)} vs. Chain-Ladder', fontsize=10, pad=6)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        _save_pdf(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────
# Figure 11: Bias bar chart (diverging horizontal)
# ──────────────────────────────────────────────────────────

def fig_bias_bars(results_df: pd.DataFrame,
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Diverging horizontal bar chart of bias by method and scenario.
    """
    set_pub_style()
    scenarios = results_df['scenario'].unique()
    methods = results_df['method'].unique()

    n_sc = len(scenarios)
    n_m = len(methods)
    fig, ax = plt.subplots(figsize=(6, 0.4 * n_sc * n_m + 1))

    y_pos = 0
    y_ticks, y_labels = [], []
    for s in scenarios:
        for j, method in enumerate(methods):
            row = results_df[(results_df['scenario'] == s) & (results_df['method'] == method)]
            if len(row) == 0:
                continue
            bias = row['bias'].values[0]
            color = _color(method)
            ax.barh(y_pos, bias, height=0.7, color=color, alpha=0.8,
                    edgecolor='white', linewidth=0.3)
            y_ticks.append(y_pos)
            y_labels.append(f'{s} / {_label(method)}')
            y_pos += 1
        y_pos += 0.5

    ax.axvline(0, color='black', linewidth=0.5, zorder=0)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=6.5)
    ax.set_xlabel('Bias')
    ax.set_title('Bias by Scenario and Method', fontsize=10, pad=6)
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        _save_pdf(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────

def _save_pdf(fig, png_path):
    pdf_path = png_path.rsplit('.', 1)[0] + '.pdf'
    try:
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    except Exception:
        pass


# ──────────────────────────────────────────────────────────
# Master generator
# ──────────────────────────────────────────────────────────

def generate_all_pub_figures(results_df: pd.DataFrame,
                             output_dir: str,
                             mc_results_by_scenario: Optional[Dict] = None):
    """Generate all publication-quality figures."""
    os.makedirs(output_dir, exist_ok=True)
    print('Generating publication-quality figures...')

    fig_four_panel(results_df,
                   save_path=os.path.join(output_dir, 'fig_four_panel.png'))
    print('  [1/9] Four-panel composite')

    fig_coverage_dot(results_df,
                     save_path=os.path.join(output_dir, 'fig_coverage.png'))
    print('  [2/9] Coverage dot plot')

    fig_advantage_heatmap(results_df,
                          save_path=os.path.join(output_dir, 'fig_advantage.png'))
    print('  [3/9] CR advantage heatmap')

    fig_rmse_heatmap(results_df,
                     save_path=os.path.join(output_dir, 'fig_rmse_heatmap.png'))
    print('  [4/9] RMSE heatmap')

    fig_bias_rmse_scatter(results_df,
                          save_path=os.path.join(output_dir, 'fig_bias_rmse_scatter.png'))
    print('  [5/9] Bias-RMSE scatter')

    fig_lollipop(results_df, cr_method='closed_cr',
                 save_path=os.path.join(output_dir, 'fig_lollipop.png'))
    print('  [6/9] Lollipop (CR vs CL)')

    fig_bias_bars(results_df,
                  save_path=os.path.join(output_dir, 'fig_bias_bars.png'))
    print('  [7/9] Bias diverging bars')

    if mc_results_by_scenario:
        fig_violin_grid(mc_results_by_scenario,
                        save_path=os.path.join(output_dir, 'fig_violin_grid.png'))
        print('  [8/9] Violin grid')

        for sid, mc_results in mc_results_by_scenario.items():
            N_true = list(mc_results.values())[0].true_N
            fig_violin_strip(mc_results, N_true, scenario_name=sid,
                             save_path=os.path.join(output_dir, f'fig_violin_{sid}.png'))

        scenarios = list(mc_results_by_scenario.keys())
        for sid in scenarios[:3]:
            fig_radar(results_df, scenario=sid,
                      save_path=os.path.join(output_dir, f'fig_radar_{sid}.png'))
        print('  [9/9] Radar charts')
    else:
        for sid in results_df['scenario'].unique()[:3]:
            fig_radar(results_df, scenario=sid,
                      save_path=os.path.join(output_dir, f'fig_radar_{sid}.png'))
        print('  [8/9] Radar charts')
        print('  [9/9] Skipped violins (no MC results)')

    plt.close('all')
    print(f'All publication figures saved to {output_dir}/')
