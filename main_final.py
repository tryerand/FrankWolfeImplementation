import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
import requests, zipfile, io
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, r2_score

# --- Constants ---
MAX_ITER = 500
TOLERANCE = 1e-4
TIME_LIMIT = 15.0
SPARSITY_THRESHOLD = 1e-5
STAGNATION_TOLERANCE = 1e-6
STAGNATION_LIMIT = 10

#region [Data Loading]
def load_superconductivity():
    """
    Loads the Superconductivity dataset from OpenML.
    Returns: X (features), y (target), feature_names
    """
    print("Loading Superconductivity dataset...")
    data = fetch_openml(name="superconduct", version=1, as_frame=True, parser='auto')
    X = data.data.values
    y = data.target.values
    print(f"Superconductivity: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, data.feature_names

def load_swarm_behaviour():
    """
    Loads the Swarm Behaviour dataset. Downloads it if not present.
    Returns: X (features), y (target), feature_names
    """
    print("Loading Swarm Behaviour dataset...")
    file_path = "Swarm Behavior Data/Flocking.csv"
    
    if not os.path.exists(file_path) and not os.path.exists("Flocking.csv"):
        print("Dataset not found. Downloading...")
        try:
            url = "https://archive.ics.uci.edu/static/public/524/swarm+behaviour.zip"
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(".")
        except Exception as e:
            print(f"Error downloading: {e}")
            return None, None, None

    if os.path.exists("Flocking.csv"): file_path = "Flocking.csv"
    
    try:
        df = pd.read_csv(file_path)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        target_col = df.columns[-1]
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        print(f"Swarm Behaviour: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, [str(c) for c in df.columns[:-1]]
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, None, None

def load_california_housing():
    """
    Loads the California Housing dataset from sklearn.
    Returns: X (features), y (target), feature_names
    """
    print("Loading California Housing dataset...")
    data = fetch_california_housing(as_frame=True) 
    X = data.data.values
    y = data.target.values
    print(f"California Housing: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, data.feature_names

#endregion

#region [Solvers]
def compute_objective_function(A, x, y):
    """Computes the Least Squares objective: 0.5 * ||Ax - y||^2 / m"""
    m = A.shape[0]
    residual = A @ x - y
    return 0.5 * np.sum(residual ** 2) / m

def linear_minimization_oracle(gradient, radius):
    """Solves the Linear Minimization Oracle for the L1 ball constraint."""
    idx = np.abs(gradient).argmax()
    sgn = -1.0 if gradient[idx] >= 0 else 1.0
    atom = np.zeros_like(gradient)
    atom[idx] = radius * sgn
    return atom, idx, sgn

def standard_frank_wolfe(A, y, l1_radius, max_iterations=MAX_ITER, tolerance=TOLERANCE, time_limit=TIME_LIMIT):
    """Executes the Standard Frank-Wolfe algorithm for Least Squares with L1 constraint."""
    m, n = A.shape
    AtA = A.T @ A
    Aty = A.T @ y
    
    x = np.zeros(n)
    objective_history = []
    gap_history = []
    times = []
    
    start_time = time.time()
    stagn = 0
    prev_gap = None

    for k in range(max_iterations):
        elapsed = time.time() - start_time
        if time_limit and elapsed > time_limit: break
            
        grad = (AtA @ x - Aty) / m
        
        s, idx, sgn = linear_minimization_oracle(grad, l1_radius)
        d = s - x
        
        gap = -np.dot(grad, d)
        
        if prev_gap is not None and abs(gap - prev_gap) < 1e-6:
            stagn += 1
        else:
            stagn = 0
        prev_gap = gap
        
        if gap <= tolerance or stagn >= 10:
            break
            
        # Exact Line Search
        Ad = AtA @ d
        denom = (np.dot(d, Ad) + 1e-12) / m
        alpha = np.clip(gap / denom, 0.0, 1.0)
        
        x = x + alpha * d
        
        obj = compute_objective_function(A, x, y)
        objective_history.append(obj)
        gap_history.append(gap)
        times.append(elapsed)
        
    return x, objective_history, gap_history, times

def away_steps_frank_wolfe(A, y, l1_radius, max_iterations=MAX_ITER, tolerance=TOLERANCE, time_limit=TIME_LIMIT):
    """Executes the Away-Steps Frank-Wolfe algorithm."""
    m, n = A.shape
    AtA = A.T @ A
    Aty = A.T @ y
    
    x = np.zeros(n)
    active_set = {} # (idx, sgn) -> weight
    
    objective_history = []
    gap_history = []
    times = []
    start_time = time.time()
    
    stagn = 0
    prev_gap = None
    
    for k in range(max_iterations):
        elapsed = time.time() - start_time
        if time_limit and elapsed > time_limit: break
            
        grad = (AtA @ x - Aty) / m
        
        s_fw, idx_fw, sgn_fw = linear_minimization_oracle(grad, l1_radius)
        d_fw = s_fw - x
        key_fw = (idx_fw, sgn_fw)
        
        if not active_set:
            d = d_fw
            max_step = 1.0
            is_away = False
        else:
            best_away_val = -float('inf')
            key_away = None
            
            for key, weight in active_set.items():
                idx_a, sgn_a = key
                val = l1_radius * sgn_a * grad[idx_a]
                if val > best_away_val:
                    best_away_val = val
                    key_away = key
            
            v_away = np.zeros_like(x)
            v_away[key_away[0]] = l1_radius * key_away[1]
            d_away = x - v_away
            
            # Check if FW direction is better than Away direction
            if np.dot(grad, d_fw) <= np.dot(grad, d_away) + 1e-10:
                d = d_fw
                max_step = 1.0
                is_away = False
            else:
                d = d_away
                alpha_max = active_set[key_away]
                max_step = alpha_max / (1.0 - alpha_max) if alpha_max < 1.0 else 1e10
                is_away = True
        
        gap = np.dot(grad, x - s_fw)
        
        if prev_gap is not None and abs(gap - prev_gap) < 1e-6:
            stagn += 1
        else:
            stagn = 0
        prev_gap = gap
        
        if gap <= tolerance or stagn >= 10: break
            
        Ad = AtA @ d
        denom = (np.dot(d, Ad) + 1e-12) / m
        current_descent = -np.dot(grad, d)
        alpha = np.clip(current_descent / denom, 0.0, max_step)
        
        x = x + alpha * d
        
        if not is_away:
            for k in active_set: active_set[k] *= (1 - alpha)
            active_set[key_fw] = active_set.get(key_fw, 0.0) + alpha
        else:
            for k in active_set: active_set[k] *= (1 + alpha)
            active_set[key_away] -= alpha
            
        active_set = {k: w for k, w in active_set.items() if w > 1e-10}
            
        obj = compute_objective_function(A, x, y)
        objective_history.append(obj)
        gap_history.append(gap)
        times.append(elapsed)
        
    return x, objective_history, gap_history, times

def pairwise_frank_wolfe(A, y, l1_radius, max_iterations=MAX_ITER, tolerance=TOLERANCE, time_limit=TIME_LIMIT):
    """Executes the Pairwise Frank-Wolfe algorithm."""
    m, n = A.shape
    AtA = A.T @ A
    Aty = A.T @ y
    
    x = np.zeros(n)
    active_set = {} 
    
    objective_history = []
    gap_history = []
    times = []
    start_time = time.time()
    prev_gap = None
    stagn = 0
    
    for k in range(max_iterations):
        elapsed = time.time() - start_time
        if time_limit and elapsed > time_limit:
            break
            
        grad = (AtA @ x - Aty) / m
        
        s_fw, idx_fw, sgn_fw = linear_minimization_oracle(grad, l1_radius)
        key_fw = (idx_fw, sgn_fw)
        
        current_weight_sum = sum(active_set.values())
        
        # Find best away candidate within active set
        best_atom_val = -float('inf')
        best_atom_key = None
        
        for key, weight in active_set.items():
            idx_a, sgn_a = key
            val = l1_radius * sgn_a * grad[idx_a]
            if val > best_atom_val:
                best_atom_val = val
                best_atom_key = key
        
        # Consider '0' atom if not fully saturated
        use_zero_atom = False
        if current_weight_sum < 1.0 - 1e-10:
            if 0.0 > best_atom_val:
                use_zero_atom = True
        
        if use_zero_atom:
            key_away = None
            v_away = np.zeros_like(x)
            alpha_max = 1.0 - current_weight_sum
        else:
            key_away = best_atom_key
            if key_away is None:
                v_away = np.zeros_like(x)
                alpha_max = 1.0
            else:
                idx_a, sgn_a = key_away
                v_away = np.zeros_like(x)
                v_away[idx_a] = l1_radius * sgn_a
                alpha_max = active_set[key_away]

        d = s_fw - v_away
        
        # Standard FW gap
        gap_fw_metric = np.dot(grad, x - s_fw)
        
        current_gap_direction = -np.dot(grad, d)
        
        if prev_gap is not None and abs(gap_fw_metric - prev_gap) < 1e-6:
            stagn += 1
        else:
            stagn = 0
        prev_gap = gap_fw_metric
        
        if gap_fw_metric <= tolerance or stagn >= 10:
            break
            
        Ad = AtA @ d
        denom = (np.dot(d, Ad) + 1e-12) / m
        alpha = np.clip(current_gap_direction / denom, 0.0, alpha_max)
        
        x = x + alpha * d
        
        active_set[key_fw] = active_set.get(key_fw, 0.0) + alpha
        
        if key_away:
            active_set[key_away] -= alpha
            if active_set[key_away] < 1e-10:
                del active_set[key_away]
                
        obj = compute_objective_function(A, x, y)
        objective_history.append(obj)
        gap_history.append(gap_fw_metric)
        times.append(elapsed)
        
    return x, objective_history, gap_history, times

def projection_l1_ball(v, radius):
    """Projects vector v onto the L1 ball of given radius."""
    if np.sum(np.abs(v)) <= radius: return v
    n = len(v)
    u = np.abs(v)
    u_sorted = np.sort(u)[::-1]
    cssv = np.cumsum(u_sorted)
    rho = np.nonzero(u_sorted * np.arange(1, n + 1) > (cssv - radius))[0][-1]
    theta = (cssv[rho] - radius) / (rho + 1)
    return np.sign(v) * np.maximum(u - theta, 0)

def projected_gradient_descent(A, y, l1_radius, max_iterations=MAX_ITER, tolerance=TOLERANCE, time_limit=TIME_LIMIT):
    """Executes Projected Gradient Descent for Least Squares with L1 constraint."""
    m, n = A.shape
    AtA = A.T @ A
    Aty = A.T @ y
    L = np.linalg.norm(AtA, 2) / m
    step_size = 1.0 / L
    
    x = np.zeros(n)
    objective_history = []
    gap_history = []
    times = []
    start_time = time.time()
    
    for k in range(max_iterations):
        elapsed = time.time() - start_time
        if time_limit and elapsed > time_limit: break
            
        grad = (AtA @ x - Aty) / m
        x_next = projection_l1_ball(x - step_size * grad, l1_radius)
        
        stationarity = np.linalg.norm((x - x_next) / step_size)
        x = x_next
        
        obj = compute_objective_function(A, x, y)
        objective_history.append(obj)
        gap_history.append(stationarity)
        times.append(elapsed)
        
        if stationarity < tolerance: break
            
    return x, objective_history, gap_history, times
#endregion

#region [Utils]
def get_datasets():
    """Loads and returns a dictionary of available datasets."""
    datasets = {}
    
    X, y, names = load_superconductivity()
    if X is not None: datasets['Superconductivity'] = (X, y, names)

    X, y, names = load_swarm_behaviour()
    if X is not None: datasets['Swarm Behaviour'] = (X, y, names)

    X, y, names = load_california_housing()
    if X is not None: datasets['California Housing'] = (X, y, names)
        
    return datasets

def grid_search_tau(X, y, tau_max, is_classification):
    """Performs a simple Grid Search (Single Train/Val Split) to find optimal tau."""
    print(f"   Running Grid Search for tau (max={tau_max:.4f})...")
    
    # Grid: 10 values from 0.1 to 1.0 of tau_max
    tau_values = np.linspace(0.1, 1.0, 10) * tau_max
    best_tau = None
    best_score = -float('inf') if is_classification else float('inf')
    
    # Simple 80/20 split for validation during grid search
    split_args = {'test_size': 0.2, 'random_state': 42}
    if is_classification:
        split_args['stratify'] = y
        
    X_t, X_v, y_t, y_v = train_test_split(X, y, **split_args)
        
    for tau in tau_values:
        # We use Projected Gradient as a fast proxy solver
        x_sol, _, _, _ = projected_gradient_descent(X_t, y_t, tau, max_iterations=200, tolerance=1e-3)
        
        if is_classification:
            y_pred_val = X_v @ x_sol
            y_pred_class = (y_pred_val >= 0.5).astype(int)
            score = accuracy_score(y_v, y_pred_class)
            
            if score > best_score:
                best_score = score
                best_tau = tau
        else:
            y_pred = X_v @ x_sol
            score = mean_squared_error(y_v, y_pred)
            
            if score < best_score:
                best_score = score
                best_tau = tau
                
    print(f"   Best tau found: {best_tau:.4f} (Score: {best_score:.4f})")
    return best_tau

def plot_sparsity_comparison(sparsity_data):
    """Generates bar plot comparing sparsity levels."""
    datasets = list(sparsity_data.keys())
    algorithms = ['FW', 'AFW', 'PFW', 'PG']
    alg_labels = ['Classic FW', 'FW Away', 'FW Pairwise', 'Projected Gradient']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    if not datasets:
        return

    x = np.arange(len(datasets))
    width = 0.2
    
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    
    for i, alg in enumerate(algorithms):
        vals = [sparsity_data[d].get(alg, 0.0) for d in datasets]
        rects = ax.bar(x + offsets[i], vals, width, label=alg_labels[i], color=colors[i])
        
    ax.set_ylabel('Percentage of Non-Zero Features (%)')
    ax.set_title('Sparsity Comparison Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("Sparsity_Comparison.png", dpi=300)
    plt.close()

def print_sparsity_table(sparsity_data):
    """Prints formatted table of sparsity percentages."""
    if not sparsity_data:
        return

    print("\nSparsity comparison (percentages of predictors maintained):")
    
    algorithms = ['FW', 'AFW', 'PFW', 'PG']
    headers = ['Dataset', 'Classic FW', 'FW Away', 'FW Pairwise', 'Proj. Grad.']
    
    dataset_width = max(len("Dataset"), max(len(d) for d in sparsity_data.keys())) + 2
    col_width = 15
    
    header_str = f"| {headers[0]:<{dataset_width}} | {headers[1]:<{col_width}} | {headers[2]:<{col_width}} | {headers[3]:<{col_width}} | {headers[4]:<{col_width}} |"
    
    print("-" * len(header_str), flush=True)
    print(header_str, flush=True)
    print("-" * len(header_str), flush=True)
    
    for dataset in sparsity_data:
        vals = sparsity_data[dataset]
        row = f"| {dataset:<{dataset_width}} "
        for i, alg in enumerate(algorithms):
            v = vals.get(alg, 0.0)
            row += f"| {v:<{col_width-1}.1f}% "
        row += "|"
        print(row, flush=True)
    print("-" * len(header_str), flush=True)

def plot_results(dataset_name, results):
    """Plots Duality Gap and Objective Function convergence."""
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    for alg, (x, hist, gap, t) in results.items():
        if len(gap) > 0:
            plt.semilogy(gap, label=alg)
    plt.xlabel('Iteration')
    plt.ylabel('Duality Gap (log)')
    plt.title(f'{dataset_name}: Duality Gap')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    plt.subplot(1, 2, 2)
    for alg, (x, hist, gap, t) in results.items():
        if len(hist) > 0:
            plt.plot(hist, label=alg)
    plt.xlabel('Iteration')
    plt.ylabel('Objective f(x)')
    plt.title(f'{dataset_name}: Convergence')
    plt.legend()
    plt.grid(True, ls="--")
    
    plt.tight_layout()
    plt.savefig(f"{dataset_name.replace(' ', '_')}_results.png")
    plt.close()

def analyze_results(dataset_name, results, X_val, y_val, y_mean, is_classification):
    """Computes and prints performance metrics."""
    print(f"\n{'-'*80}")
    print(f"Results for {dataset_name}:")
    print(f"{'-'*80}")
    
    if is_classification:
        header = f"| {'Method':<12} | {'Time (s)':<10} | {'Accuracy':<10} | {'F1':<10} | {'NNZ':<6} | {'Iter':<6} |"
        print(header)
        print(f"|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*8}|{'-'*8}|")
    else:
        header = f"| {'Method':<12} | {'Time (s)':<10} | {'MSE':<12} | {'R²':<10} | {'Adj. R²':<10} | {'NNZ':<6} | {'Iter':<6} |"
        print(header)
        print(f"|{'-'*14}|{'-'*12}|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*8}|{'-'*8}|")
        
    for alg, (x_sol, hist, gap, t) in results.items():
        time_taken = t[-1] if t else 0
        iters = len(hist)
        nnz = np.sum(np.abs(x_sol) > 1e-5)
        
        if is_classification:
            y_pred_val = X_val @ x_sol + y_mean 
            y_pred_class = (y_pred_val >= 0.5).astype(int)
            
            acc = accuracy_score(y_val, y_pred_class)
            f1 = f1_score(y_val, y_pred_class, zero_division=0)
            print(f"| {alg:<12} | {time_taken:<10.4f} | {acc:<10.4f} | {f1:<10.4f} | {nnz:<6} | {iters:<6} |")
        else:
            y_pred = X_val @ x_sol + y_mean
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # Adjusted R2
            n_samples, n_features = X_val.shape
            adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
            
            print(f"| {alg:<12} | {time_taken:<10.4f} | {mse:<12.4e} | {r2:<10.4f} | {adj_r2:<10.4f} | {nnz:<6} | {iters:<6} |")
    print(f"{'-'*80}")

#endregion

#region [Main]
def prepare_data(name, X_raw, y_raw):
    """Prepares the dataset for training: split, scale, and center."""
    is_classification = (name == 'Swarm Behaviour')
    
    if is_classification:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_raw, y_raw, test_size=0.3, random_state=42, stratify=y_raw)
        y_train_c = y_train
        y_mean = 0.0
    else:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_raw, y_raw, test_size=0.3, random_state=42)
        y_mean = np.mean(y_train)
        y_train_c = y_train - y_mean
        
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    return X_train, X_val, y_train_c, y_val, y_mean, is_classification

def run_solvers(X_train, y_train, best_tau):
    """Runs all four solvers and returns their results."""
    results = {}
    results['FW'] = standard_frank_wolfe(X_train, y_train, best_tau)
    results['AFW'] = away_steps_frank_wolfe(X_train, y_train, best_tau)
    results['PFW'] = pairwise_frank_wolfe(X_train, y_train, best_tau)
    results['PG'] = projected_gradient_descent(X_train, y_train, best_tau)
    return results

def compute_sparsity_metrics(results):
    """Computes percentage of non-zero elements for each solver."""
    sparsity = {}
    for alg, (x, _, _, _) in results.items():
        nnz = np.sum(np.abs(x) > SPARSITY_THRESHOLD)
        sparsity[alg] = (nnz / len(x)) * 100
    return sparsity

def run_experiments():
    """Main function to run experiments for all datasets and algorithms."""
    datasets = get_datasets()
    if not datasets: return
    
    sparsity_data = {}
    
    for name, (X_raw, y_raw, _) in datasets.items():
        print(f"\nProcessing {name}...")
        
        # 1. Prepare Data
        X_train, X_val, y_train, y_val, y_mean, is_classification = prepare_data(name, X_raw, y_raw)
        
        # 2. Determine Tau (Constraint) using OLS as baseline
        reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)
        w_ols = reg.coef_
            
        tau_max = np.linalg.norm(w_ols, 1)
        print(f"Tau Max (OLS L1): {tau_max:.4f}")
        
        # 3. Grid Search for optimal Tau
        best_tau = grid_search_tau(X_train, y_train, tau_max, is_classification)
        
        # 4. Run Solvers
        results = run_solvers(X_train, y_train, best_tau)
        
        # 5. Analysis & Plotting
        plot_results(name, results)
        analyze_results(name, results, X_val, y_val, y_mean, is_classification)
        
        # 6. Collect Sparsity Data
        sparsity_data[name] = compute_sparsity_metrics(results)

    # 7. Final Comparison
    plot_sparsity_comparison(sparsity_data)
    print_sparsity_table(sparsity_data)

if __name__ == "__main__":
    run_experiments()