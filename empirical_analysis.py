"""
Empirical analysis: Apply CR reserving framework to French MTPL data.
Uses the freMPL dataset or a synthetic claims triangle as fallback.
"""
import os, json, sys
sys.path.insert(0, 'cr_reserving')
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_mtpl_data():
    """Load French MTPL data if available, otherwise generate synthetic triangle."""
    mtpl_path = os.path.join(DATA_DIR, 'data', 'freMPL.csv')
    if os.path.exists(mtpl_path):
        df = pd.read_csv(mtpl_path)
        print(f"Loaded MTPL data: {df.shape[0]} claims, columns: {list(df.columns)}")
        return df
    return None


def triangle_to_capture_histories(triangle, n_accident_years=10):
    """
    Convert a claims development triangle to capture histories.

    Each cell (i,j) in the triangle is the number of claims from accident
    year i reported in development year j. We expand this into individual
    capture histories where each claim is an individual and each
    development year is a capture occasion.
    """
    D = triangle.shape[1]
    histories = []

    for i in range(n_accident_years):
        ay_claims = triangle[i, :]
        n_claims_reported = int(np.sum(ay_claims))

        # For each claim in this accident year
        cumulative = 0
        for j in range(D):
            n_new = int(ay_claims[j])
            for _ in range(n_new):
                # This claim was first captured at period j
                hist = np.zeros(D, dtype=int)
                hist[j] = 1  # captured at first report

                # For simplicity, assume no further captures (need run-off data)
                # In practice, would use individual claim development data
                histories.append(hist)

    if len(histories) == 0:
        return None
    return np.array(histories)


def chain_ladder_reserve(triangle):
    """
    Standard Chain-Ladder ultimate claims estimate.

    Parameters
    ----------
    triangle : np.ndarray of shape (A, D)
        Cumulative claims triangle (accident years x development years).

    Returns
    -------
    ultimate : np.ndarray of shape (A,)
        Estimated ultimate claims per accident year.
    reserve : np.ndarray of shape (A,)
        Reserve per accident year.
    """
    A, D = triangle.shape

    # Development factors
    factors = []
    for d in range(D - 1):
        numerator = np.sum(triangle[:A - d - 1, d + 1])
        denominator = np.sum(triangle[:A - d - 1, d])
        factors.append(numerator / denominator if denominator > 0 else 1.0)

    # Project to ultimate
    ultimate = triangle[:, -1].copy().astype(float)
    for i in range(A):
        remaining = np.prod(factors[-(D - i - 1):]) if i < D - 1 else 1.0
        ultimate[i] = triangle[i, -1] * remaining

    reserve = ultimate - triangle[:, -1]
    return ultimate, reserve


def cr_reserve_from_capture(capture_histories, method='schnabel'):
    """
    Estimate reserves using capture-recapture.

    For each accident year, we treat the claims triangle as capture
    histories across development periods.
    """
    from theory.estimators import closed_system_cr

    D = capture_histories.shape[1]
    n_obs = capture_histories.sum(axis=0)  # captures per occasion
    n_distinct = int((capture_histories.sum(axis=1) > 0).sum())

    # Schnabel estimate
    result = closed_system_cr(capture_histories, n_boot=200)

    n_unobserved = max(0, result.N_hat - n_distinct)
    return {
        'N_hat': result.N_hat,
        'N_observed': n_distinct,
        'N_unobserved': n_unobserved,
        'ci_lower': result.ci_lower,
        'ci_upper': result.ci_upper,
    }


def generate_summary_table(cr_results, cl_results):
    """Generate comparison table between CR and Chain-Ladder."""
    print("\n" + "=" * 80)
    print("CR vs Chain-Ladder Reserve Comparison")
    print("=" * 80)
    print(f"{'Method':<25} {'IBNR':>10} {'Lower':>10} {'Upper':>10}")
    print("-" * 57)
    print(f"{'Chain-Ladder':<25} {cl_results['ibnr']:>10.0f} {'-':>10} {'-':>10}")
    print(f"{'Schnabel CR':<25} {cr_results['N_unobserved']:>10.0f} "
          f"{cr_results['ci_lower']:>10.0f} {cr_results['ci_upper']:>10.0f}")


def main():
    """Run empirical analysis."""
    print("=" * 60)
    print("Empirical Analysis: French MTPL Data")
    print("=" * 60)

    # Try to load real data
    mtpl = load_mtpl_data()

    if mtpl is not None:
        print("\nProcessing real MTPL data...")
        # TODO: Transform MTPL data into claims triangle format
        # This requires understanding the data schema
    else:
        print("\nMTPL data not available. Using synthetic triangle for demonstration.")
        # Generate a synthetic claims triangle
        np.random.seed(42)
        A, D = 10, 10
        # Simulated cumulative claims triangle
        triangle = np.zeros((A, D))
        for i in range(A):
            n_claims = np.random.poisson(500)
            # Development pattern: geometric decay
            for j in range(min(D, D - i)):
                triangle[i, j] = np.random.poisson(
                    n_claims * 0.3 * (0.85) ** j)

        print("Synthetic triangle generated.")
        print(f"Shape: {triangle.shape}")
        print(f"Total reported claims: {triangle.sum():.0f}")

        # Chain-Ladder estimate
        ult_cl, reserve_cl = chain_ladder_reserve(triangle)
        print(f"\nChain-Ladder ultimate: {ult_cl.sum():.0f}")
        print(f"Chain-Ladder reserve (IBNR): {reserve_cl.sum():.0f}")

    return mtpl


if __name__ == '__main__':
    main()
