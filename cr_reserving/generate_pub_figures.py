"""Generate publication-quality figures from existing experiment results."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from cr_reserving.visualization.plots_pub import generate_all_pub_figures, fig_convergence
from cr_reserving.simulation.scenarios import get_scenario
from cr_reserving.simulation.data_generator import generate_data
from cr_reserving.simulation.monte_carlo import run_single_experiment

project_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(project_dir, 'results')
pub_dir = os.path.join(results_dir, 'pub_figures')
os.makedirs(pub_dir, exist_ok=True)

results_df = pd.read_csv(os.path.join(results_dir, 'results.csv'))
print(f'Loaded {len(results_df)} rows from results.csv')
print(f'Scenarios: {results_df["scenario"].unique()}')
print(f'Methods: {results_df["method"].unique()}')

print('\nRe-running MC experiments (200 reps) to collect distribution data...')
mc_results_by_scenario = {}
scenarios = results_df['scenario'].unique()

for i, sid in enumerate(scenarios):
    config = get_scenario(sid)
    sc_methods = ['closed_cr', 'min_chisq_cr', 'chain_ladder']
    if config.is_open:
        sc_methods.insert(1, 'open_cr')
    print(f'  [{i+1}/{len(scenarios)}] {sid}...', end=' ', flush=True)
    results = run_single_experiment(config, n_reps=200, methods=sc_methods,
                                    base_seed=42, verbose=False)
    mc_results_by_scenario[sid] = results
    print('done')

print('\nGenerating figures...')
generate_all_pub_figures(results_df, pub_dir, mc_results_by_scenario)

for sid in ['S1', 'S4']:
    config = get_scenario(sid)
    data = generate_data(config, seed=42)
    fig_convergence(data.capture_histories, data.N_true,
                    save_path=os.path.join(pub_dir, f'fig_convergence_{sid}.png'),
                    title=f'Scenario {sid}: {config.description}')
    print(f'  Convergence plot for {sid}')

print(f'\nAll figures saved to: {pub_dir}/')
