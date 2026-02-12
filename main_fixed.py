import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.preprocessing import StandardScaler

#region [Data Loading]
def load_superconductivity():
    """
    Loads the Superconductivity Data Set.
    Target: Critical Temperature
    """
    print("Loading Superconductivity dataset...")
    # Try fetching from OpenML first
    data = fetch_openml(name="superconduct", version=1, as_frame=True, parser='auto')
    X = data.data.values
    y = data.target.values
    feature_names = data.feature_names

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_centered = y - np.mean(y)
    
    print(f"Superconductivity: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    return X_scaled, y_centered, feature_names

def load_abalone():
    """
    Loads the Abalone Data Set.
    Target: Rings (Age)
    """
    print("Loading Abalone dataset...")
    # ID 183 is Abalone, very standard and reliable
    data = fetch_openml(data_id=183, as_frame=True, parser='auto')
    
    # Abalone has a categorical 'Sex' column (M, F, I). We need to one-hot encode or drop.
    # For simplicity in this optimization demo, let's just drop it or use pandas get_dummies.
    X_df = data.data
    y = data.target.values
    
    # Simple encoding
    X_df = pd.get_dummies(X_df, columns=['Sex'], drop_first=True)
    
    X = X_df.values.astype(float)
    feature_names = X_df.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    y = y.astype(float)
    y_centered = y - np.mean(y)
    
    print(f"Abalone: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    return X_scaled, y_centered, feature_names

def load_california_housing():
    """
    Loads the California Housing Data Set.
    Target: Median House Value
    """
    print("Loading California Housing dataset...")
    
    # Use sklearn built-in which is much cleaner/safer than OpenML
    data = fetch_california_housing(as_frame=True) 

    X = data.data.values
    y = data.target.values
    feature_names = data.feature_names
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # y is already float (median house value in 100k)
    y_centered = y - np.mean(y)
    
    print(f"California Housing: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    return X_scaled, y_centered, feature_names
#endregion

#region [Solvers]
def compute_objective_function(A, x, y):
    """
    Computes the least squares objective: f(x) = ||Ax - y||^2
    """
    residual = A @ x - y
    return np.dot(residual, residual)

def compute_gradient(A, AtA, Aty, x):
    """
    Computes the gradient of f(x) = ||Ax - y||^2
    Gradient = 2 * (A^T A x - A^T y)
    """
    return 2 * (AtA @ x - Aty)

def linear_minimization_oracle(gradient, radius):
    """
    Solves the LMO problem: min <gradient, s> subject to ||s||_1 <= radius.

    For the L1 ball, an extreme point is +/- radius * e_i at
    i = argmax |gradient_i|. This function returns the chosen atom and
    a key (idx, sgn) that distinguishes + and - vertices.
    """
    idx = int(np.argmax(np.abs(gradient)))
    g = float(gradient[idx])

    # Choose sign to minimize <gradient, s>
    # If g > 0 -> take negative vertex; if g < 0 -> take positive vertex.
    sgn = -1 if g > 0 else (1 if g < 0 else 1)

    atom = np.zeros_like(gradient)
    atom[idx] = radius * sgn
    return atom, idx, sgn

def projection_l1_ball(v, radius):
    """
    Projects vector v onto the L1 ball of radius 'radius'.
    Solves: min ||x - v||_2^2 s.t. ||x||_1 <= radius
    """
    if np.sum(np.abs(v)) <= radius:
        return v
    
    n = len(v)
    u = np.abs(v)
    if n == 0:
        return v
    
    # Sort u in descending order
    u_sorted = np.sort(u)[::-1]
    
    # Cumulative sum of sorted absolute values
    cssv = np.cumsum(u_sorted)
    
    # Find the number of components to keep
    rho = np.nonzero(u_sorted * np.arange(1, n + 1) > (cssv - radius))[0][-1]
    
    # Calculate theta
    theta = (cssv[rho] - radius) / (rho + 1)
    
    # Soft thresholding
    w = np.sign(v) * np.maximum(u - theta, 0)
    return w

def line_search(A, x, d, y, max_step=1.0):
    """
    Exact line search for the quadratic objective f(x + gamma * d).
    """
    Ad = A @ d
    numerator = -np.dot(A @ x - y, Ad)
    denominator = np.dot(Ad, Ad)
    
    if denominator < 1e-12:
        return 0.0
        
    optimal_step = numerator / denominator
    return np.clip(optimal_step, 0.0, max_step)

def standard_frank_wolfe(A, y, l1_radius, max_iterations=200, tolerance=1e-4, time_limit=None):
    n_samples, n_features = A.shape
    AtA = A.T @ A
    Aty = A.T @ y
    
    x = np.zeros(n_features)
    
    objective_history = []
    gap_history = []
    times = []
    
    start_time = time.time()
    
    for k in range(max_iterations):
        grad = compute_gradient(A, AtA, Aty, x)
        s, _, _ = linear_minimization_oracle(grad, l1_radius)
        d = s - x
        gap = -np.dot(grad, d)
        
        obj = compute_objective_function(A, x, y)
        objective_history.append(obj)
        gap_history.append(gap)
        times.append(time.time() - start_time)
        
        if gap < tolerance:
            break

        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        step_size = line_search(A, x, d, y, max_step=1.0)
        x = x + step_size * d
        
    return x, objective_history, gap_history, times

def away_steps_frank_wolfe(A, y, l1_radius, max_iterations=200, tolerance=1e-4, time_limit=None):
    n_samples, n_features = A.shape
    AtA = A.T @ A
    Aty = A.T @ y

    x = np.zeros(n_features)

    # Active set over L1-ball vertices: keys are (idx, sgn) where vertex is sgn*l1_radius*e_idx
    active_set = {}    # (idx, sgn) -> weight
    active_atoms = {}  # (idx, sgn) -> atom vector

    # Initialize with first FW atom
    grad = compute_gradient(A, AtA, Aty, x)
    s, idx, sgn = linear_minimization_oracle(grad, l1_radius)
    key = (idx, sgn)

    x = s.copy()
    active_set[key] = 1.0
    active_atoms[key] = s

    objective_history = []
    gap_history = []
    times = []

    start_time = time.time()

    for k in range(max_iterations):
        obj = compute_objective_function(A, x, y)
        objective_history.append(obj)
        times.append(time.time() - start_time)

        grad = compute_gradient(A, AtA, Aty, x)

        # FW direction (towards best vertex)
        s_fw, idx_fw, sgn_fw = linear_minimization_oracle(grad, l1_radius)
        key_fw = (idx_fw, sgn_fw)
        d_fw = s_fw - x
        gap_fw = -np.dot(grad, d_fw)  # >= 0
        gap_history.append(gap_fw)

        if gap_fw < tolerance:
            break
        if time_limit and (time.time() - start_time) > time_limit:
            break

        # Away direction (away from worst active vertex)
        max_inner_prod = -np.inf
        key_away = None
        for key_a in active_set:
            val = float(np.dot(grad, active_atoms[key_a]))
            if val > max_inner_prod:
                max_inner_prod = val
                key_away = key_a

        v_away = active_atoms[key_away]
        d_away = x - v_away

        # IMPORTANT: compare as positive improvements
        gap_away = -np.dot(grad, d_away)  # = <grad, v_away - x>, >=0 when away is beneficial

        # Choose direction
        if gap_fw >= gap_away:
            d = d_fw
            max_step = 1.0
            is_away = False
        else:
            d = d_away
            w = active_set[key_away]
            max_step = w / (1.0 - w) if w < 1.0 else 1e10
            is_away = True

        step_size = line_search(A, x, d, y, max_step=max_step)
        x = x + step_size * d

        # Update active set weights
        if not is_away:
            # FW step: shrink all weights, then add weight to FW atom
            for key_a in list(active_set.keys()):
                active_set[key_a] *= (1 - step_size)

            if key_fw in active_set:
                active_set[key_fw] += step_size
            else:
                active_set[key_fw] = step_size
                active_atoms[key_fw] = s_fw
        else:
            # Away step: inflate all weights, then remove weight from away atom
            for key_a in list(active_set.keys()):
                active_set[key_a] *= (1 + step_size)

            active_set[key_away] -= step_size

            if active_set[key_away] < 1e-10:
                del active_set[key_away]
                del active_atoms[key_away]

        # Cleanup small weights
        to_remove = [key_a for key_a, w in active_set.items() if w < 1e-10]
        for key_a in to_remove:
            del active_set[key_a]
            del active_atoms[key_a]

    return x, objective_history, gap_history, times

def pairwise_frank_wolfe(A, y, l1_radius, max_iterations=200, tolerance=1e-4, time_limit=None):
    n_samples, n_features = A.shape
    AtA = A.T @ A
    Aty = A.T @ y

    x = np.zeros(n_features)

    # Active set over L1-ball vertices: keys are (idx, sgn)
    active_set = {}    # (idx, sgn) -> weight
    active_atoms = {}  # (idx, sgn) -> atom vector

    # Initialize with first FW atom
    grad = compute_gradient(A, AtA, Aty, x)
    s, idx, sgn = linear_minimization_oracle(grad, l1_radius)
    key = (idx, sgn)

    x = s.copy()
    active_set[key] = 1.0
    active_atoms[key] = s

    objective_history = []
    gap_history = []
    times = []

    start_time = time.time()

    for k in range(max_iterations):
        obj = compute_objective_function(A, x, y)
        objective_history.append(obj)
        times.append(time.time() - start_time)

        grad = compute_gradient(A, AtA, Aty, x)

        # FW atom
        s_fw, idx_fw, sgn_fw = linear_minimization_oracle(grad, l1_radius)
        key_fw = (idx_fw, sgn_fw)

        d_fw = s_fw - x
        gap = -np.dot(grad, d_fw)
        gap_history.append(gap)

        if gap < tolerance:
            break
        if time_limit and (time.time() - start_time) > time_limit:
            break

        # Away atom: maximize <grad, v> over active vertices
        max_inner_prod = -np.inf
        key_away = None
        for key_a in active_set:
            val = float(np.dot(grad, active_atoms[key_a]))
            if val > max_inner_prod:
                max_inner_prod = val
                key_away = key_a

        v_away = active_atoms[key_away]

        # Pairwise direction moves mass from away vertex to FW vertex
        d_pw = s_fw - v_away
        max_step = active_set[key_away]

        step_size = line_search(A, x, d_pw, y, max_step=max_step)
        x = x + step_size * d_pw

        # Update weights
        if key_fw in active_set:
            active_set[key_fw] += step_size
        else:
            active_set[key_fw] = step_size
            active_atoms[key_fw] = s_fw

        active_set[key_away] -= step_size
        if active_set[key_away] < 1e-10:
            del active_set[key_away]
            del active_atoms[key_away]

    return x, objective_history, gap_history, times

def projected_gradient_descent(A, y, l1_radius, max_iterations=200, tolerance=1e-4, time_limit=None):
    """
    Projected Gradient Descent for LASSO:
    min ||Ax - y||^2 s.t. ||x||_1 <= l1_radius
    """
    n_samples, n_features = A.shape
    AtA = A.T @ A
    Aty = A.T @ y
    
    # Lipschitz constant (approximate)
    L = np.linalg.norm(AtA, ord=2) * 2
    step_size = 1.0 / L
    
    x = np.zeros(n_features)
    
    objective_history = []
    gap_history = [] # Use norm of gradient mapping as proxy for gap/stationarity
    times = []
    
    start_time = time.time()
    
    for k in range(max_iterations):
        grad = compute_gradient(A, AtA, Aty, x)
        
        # Gradient Step
        x_next_unconstrained = x - step_size * grad
        
        # Projection Step
        x_next = projection_l1_ball(x_next_unconstrained, l1_radius)
        
        # Convergence check using gradient mapping norm
        # The 'gradient mapping' is 1/step_size * (x - x_next)
        grad_mapping = (x - x_next) / step_size
        stationarity = np.linalg.norm(grad_mapping)
        
        x = x_next
        
        obj = compute_objective_function(A, x, y)
        objective_history.append(obj)
        gap_history.append(stationarity) # Note: this is not the FW gap
        times.append(time.time() - start_time)
        
        # Optional: Stationarity check (heuristic tolerance)
        if stationarity < tolerance:
            break
            
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
    return x, objective_history, gap_history, times
#endregion

#region [Main]
def get_datasets():
    datasets = {}
    
    X, y, names = load_superconductivity()
    if X is not None:
        datasets['Superconductivity'] = (X, y, names)

    X, y, names = load_abalone()
    if X is not None:
        datasets['Abalone'] = (X, y, names)

    X, y, names = load_california_housing()
    if X is not None:
        datasets['California Housing'] = (X, y, names)
        
    return datasets


def run_experiments():
    datasets = get_datasets()
    
    if not datasets:
        print("No datasets loaded. Exiting.")
        return

    # Algorithm configurations
    MAX_ITER = 2000
    TOL = 1e-4
    TIME_LIMIT = 20 # seconds per algorithm
    
    # We need to set a reasonable regularization parameter (tau) for each dataset
    # A heuristic is a fraction of the max possible L1 norm of the optimal unconstrained solution,
    # or just a fixed value. Different datasets might need different scales.
    # For LASSO: min ||Ax - y||^2 s.t. ||x||_1 <= tau
    
    for name, (X, y, feature_names) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Running experiments on: {name}")
        print(f"Shape: {X.shape}")
        
        # Heuristic for tau: 10% of the L1 norm of the unconstrained OLS solution 
        # (calculated on a subset if N is huge, or just pick a reasonable constant)
        # Since calculating OLS might be expensive or unstable, let's use a fixed sparsity
        # target or normalized scalar.
        # Let's try tau = 5% of max possible correlation?
        # A simple fixed tau works for demonstration.
        
        tau = 50.0 # Start with this, maybe adjust based on n_features
        if X.shape[1] > 1000:
            tau = 100.0
            
        print(f"Using L1 radius (tau): {tau}")
        
        results = {}
        
        # 1. Standard FW
        print(" > Running Standard FW...")
        x_fw, hist_fw, gap_fw, time_fw = standard_frank_wolfe(X, y, tau, MAX_ITER, TOL, TIME_LIMIT)
        results['FW'] = (x_fw, hist_fw, gap_fw, time_fw)
        
        # 2. Away-Steps FW
        print(" > Running Away-Steps FW...")
        x_afw, hist_afw, gap_afw, time_afw = away_steps_frank_wolfe(X, y, tau, MAX_ITER, TOL, TIME_LIMIT)
        results['AFW'] = (x_afw, hist_afw, gap_afw, time_afw)
        
        # 3. Pairwise FW
        print(" > Running Pairwise FW...")
        x_pfw, hist_pfw, gap_pfw, time_pfw = pairwise_frank_wolfe(X, y, tau, MAX_ITER, TOL, TIME_LIMIT)
        results['PFW'] = (x_pfw, hist_pfw, gap_pfw, time_pfw)
        
        # 4. Projected Gradient
        print(" > Running Projected Gradient...")
        x_pg, hist_pg, gap_pg, time_pg = projected_gradient_descent(X, y, tau, MAX_ITER, TOL, TIME_LIMIT)
        results['PG'] = (x_pg, hist_pg, gap_pg, time_pg)
        
        # --- PLOTTING ---
        plot_results(name, results)
        analyze_results(name, results, feature_names)

def plot_results(dataset_name, results):
    plt.figure(figsize=(15, 5))
    
    # 1. Objective vs Iteration
    plt.subplot(1, 3, 1)
    for alg, (x, hist, gap, t) in results.items():
        if len(hist) > 0:
            # Shift objective by min observed to show convergence better (log scale)
            # Find global min across all algorithms
            all_min = min(min(h) for _, (_, h, _, _) in results.items())
            plt.semilogy(np.array(hist) - all_min + 1e-10, label=alg)
            
    plt.xlabel('Iteration')
    plt.ylabel('Objective - Min(Objective)')
    plt.title(f'Convergence (Iter) - {dataset_name}')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # 2. Objective vs Time
    plt.subplot(1, 3, 2)
    for alg, (x, hist, gap, t) in results.items():
        if len(hist) > 0:
            all_min = min(min(h) for _, (_, h, _, _) in results.items())
            plt.semilogy(t, np.array(hist) - all_min + 1e-10, label=alg)
            
    plt.xlabel('Time (s)')
    plt.ylabel('Objective - Min(Objective)')
    plt.title(f'Convergence (Time) - {dataset_name}')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # 3. Sparsity / Cardinality
    plt.subplot(1, 3, 3)
    algs = []
    sparsity = []
    
    for alg, (x, hist, gap, t) in results.items():
        nnz = np.sum(np.abs(x) > 1e-5)
        algs.append(alg)
        sparsity.append(nnz)
        
    plt.bar(algs, sparsity, color=['C0', 'C1', 'C2', 'C3'])
    plt.title(f'Sparsity (NNZ) - {dataset_name}')
    plt.ylabel('Number of Non-Zero Coeffs')
    
    plt.tight_layout()
    plt.savefig(f"{dataset_name.replace(' ', '_')}_results.png")
    plt.show()

def analyze_results(dataset_name, results, feature_names):
    print(f"\nAnalysis for {dataset_name}:")
    print(f"{'Algorithm':<10} | {'Final Obj':<12} | {'Time(s)':<8} | {'Iters':<5} | {'NNZ':<5}")
    print("-" * 50)
    
    best_obj = float('inf')
    
    for alg, (x, hist, gap, t) in results.items():
        final_obj = hist[-1] if hist else float('inf')
        total_time = t[-1] if t else 0
        iters = len(hist)
        nnz = np.sum(np.abs(x) > 1e-5)
        
        print(f"{alg:<10} | {final_obj:.4e}   | {total_time:.4f}   | {iters:<5} | {nnz:<5}")
        
        best_obj = min(best_obj, final_obj)
        
#endregion

if __name__ == "__main__":
    run_experiments()
