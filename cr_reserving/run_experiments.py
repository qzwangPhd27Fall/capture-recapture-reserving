"""
Main experiment runner for CR Reserving project.

Usage:
    python run_experiments.py                     # Quick test (50 reps, S1-S2)
    python run_experiments.py --full              # Full experiment (1000 reps, S1-S7)
    python run_experiments.py --scenario S3       # Single scenario
    python run_experiments.py --reps 200          # Custom reps
    python run_experiments.py --convergence       # Generate convergence plot only
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='CR Reserving Experiments')
    parser.add_argument('--full', action='store_true', help='Full experiment (1000 reps, all scenarios)')
    parser.add_argument('--scenario', type=str, default=None, help='Run single scenario (e.g., S1)')
    parser.add_argument('--reps', type=int, default=None, help='Number of MC replications')
    parser.add_argument('--convergence', action='store_true', help='Generate convergence plot only')
    parser.add_argument('--bayesian', action='store_true', help='Include Bayesian CR method')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output or os.path.join(project_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    from cr_reserving.simulation.scenarios import get_scenario, get_all_scenarios
    from cr_reserving.simulation.data_generator import generate_data
    from cr_reserving.simulation.monte_carlo import (
        run_single_experiment, run_all_experiments, format_results_table
    )
    from cr_reserving.visualization.plots import (
        plot_convergence, plot_scenario_comparison, plot_estimate_distributions,
        plot_posterior, plot_bias_rmse_heatmap, plot_coverage_comparison,
        generate_all_figures,
    )

    if args.convergence:
        print('Generating convergence plot...')
        config = get_scenario(args.scenario or 'S2')
        data = generate_data(config, seed=args.seed)
        print(f'  N_true={data.N_true}, n_observed={data.n_observed}, n_ibnr={data.n_ibnr}')
        fig = plot_convergence(
            data.capture_histories, data.N_true,
            save_path=os.path.join(output_dir, 'convergence.png'),
            title=f'CR Estimator Convergence ({config.name}: {config.description})'
        )
        print(f'  Saved to {output_dir}/convergence.png')
        return

    if args.full:
        n_reps = args.reps or 1000
        scenarios = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
    elif args.scenario:
        n_reps = args.reps or 200
        scenarios = [args.scenario]
    else:
        n_reps = args.reps or 50
        scenarios = ['S1', 'S2']

    methods = ['closed_cr', 'min_chisq_cr', 'chain_ladder']
    if args.bayesian:
        methods.append('bayesian_quick')

    print(f'\n{"="*60}')
    print(f'  CR RESERVING - Monte Carlo Experiments')
    print(f'  Scenarios: {scenarios}')
    print(f'  Methods:   {methods}')
    print(f'  Reps:      {n_reps}')
    print(f'  Seed:      {args.seed}')
    print(f'  Output:    {output_dir}')
    print(f'{"="*60}\n')

    t_start = time.time()

    mc_results_by_scenario = {}
    all_results = []

    for i, sid in enumerate(scenarios):
        config = get_scenario(sid)

        sc_methods = methods.copy()
        if config.is_open and 'open_cr' not in sc_methods:
            sc_methods.insert(1, 'open_cr')

        print(f'\n[{i+1}/{len(scenarios)}] Scenario {sid}: {config.description}')
        print(f'  N_true={config.N_true}, D={config.D}, open={config.is_open}')

        results = run_single_experiment(
            config, n_reps=n_reps, methods=sc_methods,
            base_seed=args.seed, verbose=True
        )

        mc_results_by_scenario[sid] = results

        for method, mcr in results.items():
            all_results.append({
                'scenario': mcr.scenario,
                'method': mcr.method,
                'N_true': mcr.true_N,
                'bias': mcr.bias,
                'rmse': mcr.rmse,
                'coverage': mcr.coverage,
                'ci_width': mcr.ci_width,
                'mean_est': mcr.mean_estimate,
                'median_est': mcr.median_estimate,
                'n_reps': mcr.n_reps,
                'time_sec': mcr.elapsed_sec,
            })

    results_df = pd.DataFrame(all_results)

    csv_path = os.path.join(output_dir, 'results.csv')
    results_df.to_csv(csv_path, index=False)

    elapsed = time.time() - t_start
    print(f'\n{"="*60}')
    print(f'  RESULTS SUMMARY (total time: {elapsed:.1f}s)')
    print(f'{"="*60}')
    print(format_results_table(results_df))
    print(f'\nResults saved to: {csv_path}')

    print('\nGenerating figures...')
    try:
        generate_all_figures(results_df, output_dir, mc_results_by_scenario)
    except Exception as e:
        print(f'  Warning: figure generation failed: {e}')

    # Generate convergence plots for first 2 scenarios
    for sid in scenarios[:2]:
        config = get_scenario(sid)
        data = generate_data(config, seed=args.seed)
        try:
            plot_convergence(
                data.capture_histories, data.N_true,
                save_path=os.path.join(output_dir, f'convergence_{sid}.png'),
                title=f'Convergence ({sid}: {config.description})'
            )
        except Exception as e:
            print(f'  Warning: convergence plot for {sid} failed: {e}')

    print(f'\nAll outputs saved to: {output_dir}/')
    print('Done.')


if __name__ == '__main__':
    main()
