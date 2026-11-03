"""
Full-scale Monte Carlo experiment: 7 scenarios x 500 reps.
Generates all simulation evidence for the CR reserving paper.

Progress bars at scenario and rep level. Saves structured results.
"""
import sys, os, json, time
sys.path.insert(0, 'cr_reserving')
import numpy as np
from simulation.scenarios import get_scenario
from simulation.data_generator import generate_data
from theory.estimators import closed_system_cr, open_system_cr
from theory.doubly_robust import doubly_robust_cr, min_chisq_cr_improved

N_REPS = 500
N_BOOT = 200  # sufficient for stable CIs, validated at n_boot=300 in pilot

SCENARIO_METHODS = {
    'S1': ['closed_cr', 'min_chisq', 'doubly_robust'],
    'S2': ['closed_cr', 'min_chisq', 'doubly_robust'],
    'S3': ['closed_cr', 'open_cr', 'min_chisq', 'doubly_robust'],
    'S4': ['closed_cr', 'open_cr', 'min_chisq', 'doubly_robust'],
    'S5': ['closed_cr', 'min_chisq', 'doubly_robust'],
    'S6': ['closed_cr', 'min_chisq', 'doubly_robust'],
    'S7': ['closed_cr', 'min_chisq', 'doubly_robust'],
}

def run_one_rep(config, methods, rep, base_seed=42):
    """Run one Monte Carlo replicate for all methods."""
    data = generate_data(config, seed=base_seed + rep)
    ever_obs = data.capture_histories_full.any(axis=1) if config.is_open else None
    obs_entry = data.entry_times[ever_obs] if config.is_open and ever_obs is not None else None
    obs_exit = data.exit_times[ever_obs] if config.is_open and ever_obs is not None else None

    results = {}
    for method in methods:
        try:
            if method == 'closed_cr':
                res = closed_system_cr(data.capture_histories, n_boot=N_BOOT, seed=base_seed+rep)
                results[method] = {
                    'N_hat': res.N_hat, 'ci_lower': res.ci_lower, 'ci_upper': res.ci_upper,
                    'calib': 1.0, 'fail': 0
                }
            elif method == 'open_cr':
                res = open_system_cr(data.capture_histories, entry_times=obs_entry,
                                     exit_times=obs_exit, n_boot=N_BOOT, seed=base_seed+rep)
                results[method] = {
                    'N_hat': res.N_hat, 'ci_lower': res.ci_lower, 'ci_upper': res.ci_upper,
                    'calib': 1.0, 'fail': 0
                }
            elif method == 'min_chisq':
                res = min_chisq_cr_improved(data.capture_histories, n_boot=N_BOOT,
                                            seed=base_seed+rep, ci_method='asymptotic')
                results[method] = {
                    'N_hat': res['N_hat'], 'ci_lower': res['ci_lower'], 'ci_upper': res['ci_upper'],
                    'calib': res.get('calibration_factor', 1.0), 'fail': 0
                }
            elif method == 'doubly_robust':
                res = doubly_robust_cr(data.capture_histories, n_boot=N_BOOT,
                                       seed=base_seed+rep, ci_method='asymptotic')
                results[method] = {
                    'N_hat': res['N_hat'], 'ci_lower': res['ci_lower'], 'ci_upper': res['ci_upper'],
                    'calib': res.get('calibration_factor', 1.0), 'fail': 0
                }
        except Exception:
            results[method] = {
                'N_hat': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'calib': np.nan, 'fail': 1
            }
    return results

def progress_bar(pct, width=30, elapsed=0, eta=0):
    n = int(pct / 100 * width)
    bar = '#' * n + '-' * (width - n)
    eta_str = f'ETA {eta:.0f}s' if eta > 0 else ''
    return f'[{bar}] {pct:5.1f}% {eta_str}'

def run_scenario(scenario_id, config):
    """Run one scenario's Monte Carlo experiment."""
    methods = SCENARIO_METHODS[scenario_id]
    N_true = config.N_true

    # Initialize storage
    store = {m: {'est': np.zeros(N_REPS), 'lo': np.zeros(N_REPS),
                 'hi': np.zeros(N_REPS), 'cal': np.zeros(N_REPS),
                 'fail': 0} for m in methods}

    t0 = time.time()
    for rep in range(N_REPS):
        res = run_one_rep(config, methods, rep, base_seed=1000 * (list(SCENARIO_METHODS.keys()).index(scenario_id) + 1))
        for method in methods:
            r = res[method]
            store[method]['est'][rep] = r['N_hat']
            store[method]['lo'][rep] = r['ci_lower']
            store[method]['hi'][rep] = r['ci_upper']
            store[method]['cal'][rep] = r['calib']
            store[method]['fail'] += r['fail']

        # Progress bar every 5 reps
        if rep % 5 == 0 or rep == N_REPS - 1:
            pct = (rep + 1) / N_REPS * 100
            elapsed = time.time() - t0
            eta = elapsed / max(rep + 1, 1) * (N_REPS - rep - 1)
            bar = progress_bar(pct, 30, elapsed, eta)
            sys.stdout.write(f'\r  {scenario_id}: {bar}')
            sys.stdout.flush()

    elapsed = time.time() - t0
    sys.stdout.write(f'\r  {scenario_id}: Done in {elapsed:.0f}s{" " * 40}\n')
    sys.stdout.flush()

    # Compute summary
    summary = {}
    for method in methods:
        est = store[method]['est']
        valid = ~np.isnan(est)
        ev = est[valid]
        cl = store[method]['lo'][valid]
        cu = store[method]['hi'][valid]
        if len(ev) == 0:
            summary[method] = {'Bias': np.nan, 'RMSE': np.nan, 'SE_emp': np.nan,
                               'Cov': np.nan, 'CI_W': np.nan, 'Mean': np.nan,
                               'Calib': np.nan, 'Fail': store[method]['fail']}
        else:
            summary[method] = {
                'Bias': float(np.mean(ev - N_true)),
                'RMSE': float(np.sqrt(np.mean((ev - N_true)**2))),
                'SE_emp': float(np.std(ev, ddof=1)),
                'Cov': float(np.mean((cl <= N_true) & (cu >= N_true))),
                'CI_W': float(np.mean(cu - cl)),
                'Mean': float(np.mean(ev)),
                'Calib': float(np.nanmean(store[method]['cal'][valid])),
                'Fail': store[method]['fail'],
                'N_valid': len(ev),
            }

    return store, summary

def save_results(scenario_id, store, summary, out_dir):
    """Save scenario results as compressed numpy arrays."""
    npz_path = os.path.join(out_dir, f'mc_{scenario_id}.npz')
    save_dict = {}
    for method, s in store.items():
        for key, arr in s.items():
            if isinstance(arr, np.ndarray):
                save_dict[f'{method}_{key}'] = arr
    np.savez_compressed(npz_path, **save_dict)

    # Summary as JSON
    sum_path = os.path.join(out_dir, f'summary_{scenario_id}.json')
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)

def print_final_table(all_summaries):
    """Print final results table for all scenarios."""
    print('\n' + '=' * 120)
    print('FINAL MONTE CARLO RESULTS — 7 Scenarios x 500 Reps')
    print('=' * 120)

    for scenario_id in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']:
        config = get_scenario(scenario_id)
        print(f'\n--- {scenario_id}: {config.description} (D={config.D}, N={config.N_true}) ---')
        methods = SCENARIO_METHODS[scenario_id]
        header = f'  {"Method":<18} {"Bias":>8} {"RMSE":>8} {"SE":>8} {"Cov":>7} {"CI_W":>8} {"Mean":>9} {"Calib":>6} {"Fail":>5}'
        print(header)
        print('  ' + '-' * 85)
        for method in methods:
            s = all_summaries[scenario_id][method]
            if np.isnan(s['Bias']):
                print(f'  {method:<18} {"ALL FAILED":>60}')
            else:
                print(f'  {method:<18} {s["Bias"]:>+8.1f} {s["RMSE"]:>8.1f} {s["SE_emp"]:>8.1f} '
                      f'{s["Cov"]:>7.3f} {s["CI_W"]:>8.1f} {s["Mean"]:>9.1f} {s["Calib"]:>5.2f}x {s["Fail"]:>5}')

def main():
    out_dir = r'C:\Users\user\Desktop\合作论文\在捕捉\20180126捕获再捕获分析\mc_results'
    os.makedirs(out_dir, exist_ok=True)

    all_summaries = {}
    scenario_ids = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']

    t_total = time.time()
    for i, sid in enumerate(scenario_ids):
        print(f'\n[Scenario {i+1}/7] {sid}')
        config = get_scenario(sid)
        store, summary = run_scenario(sid, config)
        save_results(sid, store, summary, out_dir)
        all_summaries[sid] = summary

    print(f'\nTotal time: {(time.time() - t_total) / 60:.1f} min')
    print_final_table(all_summaries)

    # Save aggregate summary
    with open(os.path.join(out_dir, 'summary_all.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f'\nResults saved to: {out_dir}')

if __name__ == '__main__':
    main()
