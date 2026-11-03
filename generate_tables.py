"""
Generate LaTeX tables from MC results for direct inclusion in paper_draft.tex.
Usage: python generate_tables.py
"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'mc_results')

SID_ORDER = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
SID_LABELS = {
    'S1': 'S1 (Homogeneous)',
    'S2': 'S2 (Heterogeneous)',
    'S3': 'S3 (Open, stable)',
    'S4': 'S4 (Open, shock)',
    'S5': 'S5 (Short-tail)',
    'S6': 'S6 (Long-tail)',
    'S7': 'S7 (Extreme heterog.)',
}
METH_LABELS = {
    'closed_cr': 'Schnabel',
    'min_chisq': 'Min-$\\chi^2$',
    'doubly_robust': 'Doubly-Robust',
    'open_cr': 'Open CR',
}


def load_summaries():
    path = os.path.join(RESULTS, 'summary_all.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    summaries = {}
    for sid in SID_ORDER:
        p = os.path.join(RESULTS, f'summary_{sid}.json')
        if os.path.exists(p):
            with open(p) as f:
                summaries[sid] = json.load(f)
    return summaries


def generate_bias_rmse_table(summaries):
    """Generate Table 1: Bias and RMSE."""
    methods = ['closed_cr', 'min_chisq', 'doubly_robust']
    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'\centering')
    lines.append(r'\caption{Monte Carlo Results: Estimation Bias and RMSE (500 Reps)}')
    lines.append(r'\label{tab:mc_main}')
    lines.append(r'\begin{tabular}{l' + 'cc' * len(methods) + '}')
    lines.append(r'\toprule')
    header = 'Scenario'
    for m in methods:
        header += f' & \\multicolumn{{2}}{{c}}{{{METH_LABELS[m]}}}'
    lines.append(header + r' \\')
    lines.append('& ' + ' & '.join(['Bias & RMSE'] * len(methods)) + r' \\')
    lines.append(r'\midrule')

    for sid in SID_ORDER:
        label = SID_LABELS[sid]
        row = label
        for m in methods:
            s = summaries.get(sid, {}).get(m, {})
            bias = s.get('Bias', float('nan'))
            rmse = s.get('RMSE', float('nan'))
            if not (bias != bias):  # not nan
                row += f' & {bias:+.1f} & {rmse:.1f}'
            else:
                row += ' & --- & ---'
        row += r' \\'
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def generate_ci_table(summaries):
    """Generate Table 2: CI Coverage and Width."""
    methods = ['closed_cr', 'min_chisq', 'doubly_robust']
    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'\centering')
    lines.append(r'\caption{Monte Carlo Results: 95\% CI Coverage and Width (500 Reps)}')
    lines.append(r'\label{tab:mc_ci}')
    lines.append(r'\begin{tabular}{l' + 'cc' * len(methods) + '}')
    lines.append(r'\toprule')
    header = 'Scenario'
    for m in methods:
        header += f' & \\multicolumn{{2}}{{c}}{{{METH_LABELS[m]}}}'
    lines.append(header + r' \\')
    lines.append('& ' + ' & '.join(['Coverage & CI Width'] * len(methods)) + r' \\')
    lines.append(r'\midrule')

    for sid in SID_ORDER:
        label = SID_LABELS[sid]
        row = label
        for m in methods:
            s = summaries.get(sid, {}).get(m, {})
            cov = s.get('Cov', float('nan'))
            ciw = s.get('CI_W', float('nan'))
            if not (cov != cov):
                row += f' & {cov:.3f} & {ciw:.1f}'
            else:
                row += ' & --- & ---'
        row += r' \\'
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def generate_latex_tables():
    summaries = load_summaries()
    if not summaries:
        print("No MC results found.")
        return

    # Write table fragments
    bias_table = generate_bias_rmse_table(summaries)
    ci_table = generate_ci_table(summaries)

    tables_dir = os.path.join(BASE, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    with open(os.path.join(tables_dir, 'table_bias_rmse.tex'), 'w') as f:
        f.write(bias_table)
    with open(os.path.join(tables_dir, 'table_ci.tex'), 'w') as f:
        f.write(ci_table)

    print(f"Tables written to {tables_dir}/")
    print("\n--- Table: Bias & RMSE ---")
    print(bias_table)
    print("\n--- Table: CI ---")
    print(ci_table)


if __name__ == '__main__':
    generate_latex_tables()
