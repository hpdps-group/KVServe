import itertools
import ast
import argparse
import logging
import json
import time
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import warnings
import sys
import os
import io
import gc
from contextlib import redirect_stdout, redirect_stderr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# ================= Configuration =================

# Target constraints and search budget.
BASELINE_ACC = 100
ACC_TOLERANCE = 3
TARGET_ACC_THRESHOLD = BASELINE_ACC - ACC_TOLERANCE
PRUNING_EPSILON = 0.2
MAX_ITER = 5
COLD_START_POINTS = 3
EXPLORATION_WEIGHT = 1
SEED = 42
WHETHER_TO_EXPLORE = True
# Parameter search space.
SEARCH_SPACE = {
    "transform_type": ["hadamard"],
    "heads_selection": [0.3, 0.5, 0.7, 0.9], 
    "high_key_max_value": [12, 10, 8],
    "high_value_max_value": [12, 10, 8],
    "low_key_max_value": [6, 4],
    "low_value_max_value": [6, 4],
    "axis_key_options": [(2,)],
    "axis_value_options": [(1, 3)]
}
MODEL_NAME = "Qwen2.5-7B-Instruct"
TASK_TO_SEARCH = ["longbench_2wikimqa"]
DATASET_LIMIT = 5
BATCH_SIZE = 2
CACHE_CSV_PATH = "search_space.csv"
FINAL_JSON_PATH = f"tolerance_{ACC_TOLERANCE}_results.json"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = "/data/models"
BASE_CONFIG_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "duo_config"))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CR_CACHE_PATH = None
ATTN_IMPLEMENTATION = "sdpa"
EVALUATION_MODE = "model"


def configure_from_cli(argv=None):
    """Load the small set of runtime knobs used by the AE entrypoint."""
    parser = argparse.ArgumentParser(description="KVServe offline compression-parameter search")
    parser.add_argument("--config", help="JSON search configuration")
    parser.add_argument("--results-dir", help="Override the output directory")
    args = parser.parse_args(argv)

    config = {}
    if args.config:
        config_path = os.path.abspath(args.config)
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)

    global MODEL_NAME, TASK_TO_SEARCH, DATASET_LIMIT, BATCH_SIZE
    global MAX_ITER, COLD_START_POINTS, ACC_TOLERANCE, TARGET_ACC_THRESHOLD, SEED
    global BASE_MODEL_PATH, BASE_CONFIG_PATH, RESULTS_DIR, CR_CACHE_PATH
    global CACHE_CSV_PATH, FINAL_JSON_PATH, ATTN_IMPLEMENTATION, EVALUATION_MODE

    model_path = config.get("model_path")
    if model_path:
        model_path = os.path.abspath(model_path)
        configured_name = config.get("model_name", os.path.basename(model_path.rstrip(os.sep)))
        if os.path.basename(model_path.rstrip(os.sep)) != configured_name:
            raise ValueError("model_path basename must match model_name")
        MODEL_NAME = configured_name
        BASE_MODEL_PATH = os.path.dirname(model_path.rstrip(os.sep))
    else:
        MODEL_NAME = config.get("model_name", MODEL_NAME)
        BASE_MODEL_PATH = os.path.abspath(config.get("base_model_path", BASE_MODEL_PATH))

    tasks = config.get("tasks", config.get("workloads", TASK_TO_SEARCH))
    TASK_TO_SEARCH = [tasks] if isinstance(tasks, str) else list(tasks)
    DATASET_LIMIT = int(config.get("dataset_limit", DATASET_LIMIT))
    BATCH_SIZE = int(config.get("batch_size", BATCH_SIZE))
    MAX_ITER = int(config.get("max_iter", MAX_ITER))
    COLD_START_POINTS = int(config.get("cold_start_points", COLD_START_POINTS))
    ACC_TOLERANCE = float(config.get("acc_tolerance", ACC_TOLERANCE))
    TARGET_ACC_THRESHOLD = BASELINE_ACC - ACC_TOLERANCE
    SEED = int(config.get("seed", SEED))
    BASE_CONFIG_PATH = os.path.abspath(config.get("base_config_path", BASE_CONFIG_PATH))
    RESULTS_DIR = os.path.abspath(args.results_dir or config.get("output", RESULTS_DIR))
    CR_CACHE_PATH = config.get("cr_cache_path")
    if CR_CACHE_PATH:
        CR_CACHE_PATH = os.path.abspath(CR_CACHE_PATH)
    CACHE_CSV_PATH = config.get("cache_csv", CACHE_CSV_PATH)
    FINAL_JSON_PATH = config.get("final_json", f"tolerance_{ACC_TOLERANCE:g}_results.json")
    ATTN_IMPLEMENTATION = config.get("attn_implementation", ATTN_IMPLEMENTATION)
    EVALUATION_MODE = config.get("evaluation_mode", EVALUATION_MODE)

    model_dir = os.path.join(BASE_MODEL_PATH, MODEL_NAME)
    scores_path = os.path.join(BASE_CONFIG_PATH, f"{MODEL_NAME}_scores.csv")
    if EVALUATION_MODE not in {"smoke", "model"}:
        raise ValueError("evaluation_mode must be 'smoke' or 'model'")
    if EVALUATION_MODE == "model":
        if not os.path.isfile(os.path.join(model_dir, "config.json")):
            raise FileNotFoundError(f"Model not found: {model_dir}")
        if not os.path.isfile(scores_path):
            raise FileNotFoundError(f"Head-score file not found: {scores_path}")
    if DATASET_LIMIT < 1 or BATCH_SIZE < 1 or MAX_ITER < 0 or COLD_START_POINTS < 1:
        raise ValueError(
            "dataset_limit, batch_size, and cold_start_points must be positive; "
            "max_iter must be non-negative"
        )


def model_results_dir():
    return os.path.join(RESULTS_DIR, MODEL_NAME)


def cr_cache_path():
    return CR_CACHE_PATH or os.path.join(model_results_dir(), CACHE_CSV_PATH)


class SmokeAccuracyEvaluator:
    """Fast deterministic proxy used only to exercise the search controller."""

    def evaluate(self, params):
        # Produce a smooth decision boundary with both feasible and infeasible
        # candidates.  This is intentionally not a measured model accuracy.
        accuracy = (
            101.2
            - 0.45 * float(params["cr"])
            - 0.25 * float(params["heads_selection"])
            + 0.04 * (float(params["low_key_max_value"]) + float(params["low_value_max_value"]))
        )
        return round(float(np.clip(accuracy, 0.0, 100.0)), 2)

# ================= Utilities =================

def is_valid_config(params):
    try:
        if not (0 <= params["heads_selection"] <= 1): return False
        if params["high_key_max_value"] < params["low_key_max_value"]: return False
        if params["high_value_max_value"] < params["low_value_max_value"]: return False
        for k in ["high_key_max_value", "high_value_max_value", "low_key_max_value", "low_value_max_value"]:
            if not (0 < params[k] < 256): return False
        if not all(0 <= x <= 3 for x in params["axis_key"]): return False
        if not all(0 <= x <= 3 for x in params["axis_value"]): return False
        return True
    except KeyError: return False

def silent_call(func, *args, **kwargs):
    # Suppress logging output while running noisy evaluation code.
    logging.disable(logging.CRITICAL)
    
    temp_stdout = io.StringIO()
    temp_stderr = io.StringIO()
    try:
        with redirect_stdout(temp_stdout), redirect_stderr(temp_stderr):
            return func(*args, **kwargs)
    finally:
        # Restore logging after the wrapped call finishes.
        logging.disable(logging.NOTSET)

def run_cr(evaluator, params):
    logging.info(f"   >>> Running Compression Evaluation ")
    try:
        # Evaluate compression ratio for the provided parameter set.
        cr = evaluator.evaluate(params)
        return cr
    except Exception as e:
        logging.error(f"Error running CR evaluation: {e}")
        raise

def run_acc(evaluator, params):
    logging.info(f"   >>> Running Accuracy Evaluation ")
    try:
        acc = evaluator.evaluate(params)
        logging.info(f"   >>> Result Accuracy: {acc:.4f}")
        return acc
    except Exception as e:
        logging.error(f"Error running ACC evaluation: {e}")
        raise

def bigger_pruning_select_by(df, current_params, current_idx, current_iteration, isolation_columns):
    if current_iteration < MAX_ITER / 5:
        return []
    
    # Build the isolation mask: keep rows that match current_params on all isolation columns.
    isolation_mask = pd.Series(True, index=df.index)
    for col in isolation_columns:
        if col in df.columns:
            isolation_mask &= (df[col] == current_params[col])

    # Build the weaker mask according to the monotonicity assumption over compression strength.
    max_value = max(current_params['high_key_max_value'], current_params['high_value_max_value'], current_params['low_key_max_value'], current_params['low_value_max_value'])
    min_value = min(current_params['high_key_max_value'], current_params['high_value_max_value'], current_params['low_key_max_value'], current_params['low_value_max_value'])
    
    if max_value - min_value < 8 or current_iteration > MAX_ITER / 2:
        weaker_mask = pd.Series(True, index=df.index)
        weaker_mask &= (df['cr'] >= current_params['cr'] + 2 * PRUNING_EPSILON)
    else:
        return []
    
    # Exclude the current row.
    not_self = (df.index != current_idx)    

    # Combine all pruning conditions.
    final_mask = isolation_mask & weaker_mask & not_self
    
    # Return matching candidate indices.
    return df[final_mask].index.tolist()

def smaller_pruning_select_by(df, current_params, current_idx, current_iteration, isolation_columns):
    if current_iteration <= MAX_ITER / 5:
        return []
    
    isolation_mask = pd.Series(True, index=df.index)
    for col in isolation_columns:
        if col in df.columns:
            isolation_mask &= (df[col] == current_params[col])

    # Keep candidates with a lower, more conservative compression ratio.
    # The epsilon margin prevents pruning near-tie configurations.
    conservative_mask = pd.Series(True, index=df.index)
    conservative_mask &= (df['cr'] <= current_params['cr'] - PRUNING_EPSILON / 2)
    
    not_self = (df.index != current_idx)    
    final_mask = isolation_mask & conservative_mask & not_self
    return df[final_mask].index.tolist()

def process_one_hot_encoding(df, encode_cols, current_feature_cols):
    """
    Add one-hot encoded feature columns for categorical search parameters.
    
    Args:
        df (pd.DataFrame): Search-space dataframe.
        encode_cols (list): Column names to one-hot encode, e.g. ['axis_key', 'axis_value'].
        current_feature_cols (list): Existing numeric feature columns used by the GP model.
        
    Returns:
        df (pd.DataFrame): Dataframe with newly generated one-hot columns.
        final_feature_cols (list): Updated feature list including the one-hot columns.
    """
    # Copy the feature list to avoid mutating the caller-owned object.
    final_feature_cols = list(current_feature_cols)
    
    for col in encode_cols:
        if col not in df.columns:
            print(f"[Warning] Column '{col}' not found in DataFrame. Skipping.")
            continue
            
        # Build a value-to-ID mapping. Lists are converted to tuples so they are hashable.
        try:
            # Fast path for already-hashable values.
            unique_vals = sorted(list(set(df[col])))
        except TypeError:
            # Handle list-valued cells by normalizing them to tuples first.
            temp_series = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
            unique_vals = sorted(list(set(temp_series)))

        val_map = {v: i for i, v in enumerate(unique_vals)}
        
        # Generate an ID column, e.g. axis_key -> axis_key_id.
        # Use apply instead of map to avoid pandas tuple-indexing edge cases.
        id_col_name = f"{col}_id"
        
        # Match the lookup key type with the keys stored in val_map.
        def get_id(x):
            key = tuple(x) if isinstance(x, list) else x
            return val_map[key]
            
        df[id_col_name] = df[col].apply(get_id)
        
        # Generate one-hot columns such as "axis_key_0" and "axis_key_1".
        dummies = pd.get_dummies(df[id_col_name], prefix=col)
        
        # Append the generated columns to the dataframe.
        df = pd.concat([df, dummies], axis=1)
        
        # Add the generated dummy columns to the GP feature list.
        new_cols = list(dummies.columns)
        final_feature_cols.extend(new_cols)
        
        # The ID columns are intentionally not added to the GP feature list; only
        # the one-hot columns are returned as numerical model features.
        
    return df, final_feature_cols

def check_early_stopping(df, bo):
    """
    Check whether the search should stop early.

    Conditions:
        1. The remaining CR range is no larger than 2 * PRUNING_EPSILON.
        2. The consecutive failure count exceeds MAX_ITER / 10.
    """
    # Identify remaining candidates that have not been evaluated or pruned.
    visited = set(bo.observed_indices) | bo.skipped_indices
    remaining_mask = ~df.index.isin(visited)
    
    if not remaining_mask.any():
        return False
        
    remaining_crs = df.loc[remaining_mask, 'cr']
    max_cr = remaining_crs.max()
    min_cr = remaining_crs.min()
    diff = max_cr - min_cr
    
    if diff > 2 * PRUNING_EPSILON + PRUNING_EPSILON / 2:
        bo.consecutive_fail_count = 0
        return False
    elif bo.consecutive_fail_count > MAX_ITER / 10:
        logging.info(f"🛑 Early Stopping Triggered: Remaining CR Range [{min_cr:.4f}, {max_cr:.4f}] (Diff={diff:.4f} <= {2 * PRUNING_EPSILON + PRUNING_EPSILON / 2:.4f}) AND Consecutive Failures ({bo.consecutive_fail_count} > {MAX_ITER / 10})")
        return True
        
    return False

# ================= Bayesian Optimization =================

class ConstraintAwareBO:
    def __init__(self, candidate_pool_df, seed=42):
        self.candidates = candidate_pool_df
        self.observed_indices = [] 
        self.observed_accs = []
        self.skipped_indices = set()
        self.seed = seed
        self.feature_cols = None
        self.consecutive_fail_count = 0
        
        kernel = Matern(nu=1.5, length_scale=1.0) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=self.seed)

        self.scaler = MinMaxScaler()
        self.scaler_fitted = False

    def _prepare_data(self):
        """Fit the feature scaler before it is used for the first time."""
        # The scaler can only be fit after feature_cols has been assigned.
        if not self.scaler_fitted:
            if not self.feature_cols:
                return # Features are not ready yet.
            
            # Fit normalization statistics over the full search space.
            X_all = self.candidates[self.feature_cols].values
            self.scaler.fit(X_all)
            self.scaler_fitted = True

    def fit(self):
        if not self.observed_indices: return
        self._prepare_data()
        X_raw = self.candidates.iloc[self.observed_indices][self.feature_cols].values
        X_scaled = self.scaler.transform(X_raw)
        y = np.array(self.observed_accs)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The GP model is trained on normalized features.
            self.gp.fit(X_scaled, y)

    def propose_next(self, threshold, exploration_weight=0.1):
        self._prepare_data()
        X_all_raw = self.candidates[self.feature_cols].values
        X_all_scaled = self.scaler.transform(X_all_raw)
        mu, sigma = self.gp.predict(X_all_scaled, return_std=True)
        
        with np.errstate(divide='ignore'):
            z_scores = (mu - threshold) / sigma
        prob_feasible = norm.cdf(z_scores)
        
        # Base score: expected compression ratio under feasibility probability.
        base_score = self.candidates["cr"].values * prob_feasible
        
        # Exploration bonus: predictive uncertainty weighted by exploration_weight.
        norm_sigma = sigma / sigma.max() if sigma.max() > 0 else sigma
        exploration_score = norm_sigma * exploration_weight * self.candidates["cr"].mean()
        if not WHETHER_TO_EXPLORE: exploration_score = 0
        
        acquisition_scores = base_score + exploration_score
        
        # Mask candidates that have already been observed or pruned.
        for idx in self.observed_indices: acquisition_scores[idx] = -1.0
        for idx in self.skipped_indices: acquisition_scores[idx] = -1.0

        best_idx = np.argmax(acquisition_scores)
        if acquisition_scores[best_idx] == -1.0: return None, 0.0, 0.0 
        return best_idx, prob_feasible[best_idx], mu[best_idx]

# ================= Main Flow =================

def main():
    configure_from_cli()
    started_at = time.monotonic()
    # ====================================================
    # Logging Config
    # ====================================================
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True
    )
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("lm_eval").setLevel(logging.ERROR)

    logging.info(
        "Search setup: mode=%s model=%s candidates=CR-cache cold_start=%d bo_iter=%d",
        EVALUATION_MODE, MODEL_NAME, COLD_START_POINTS, MAX_ITER,
    )
    if EVALUATION_MODE == "smoke":
        logging.info("Smoke mode uses deterministic proxy accuracy; it does not produce measured profiles.")

    logging.info("=== Phase 1: Generating & Profiling Search Space ===")
    
    keys = ["transform_type", "heads_selection", "high_key_max_value", "high_value_max_value", "low_key_max_value", "low_value_max_value", "axis_key", "axis_value"]
    values = [SEARCH_SPACE["transform_type"], SEARCH_SPACE["heads_selection"], SEARCH_SPACE["high_key_max_value"], SEARCH_SPACE["high_value_max_value"], SEARCH_SPACE["low_key_max_value"], SEARCH_SPACE["low_value_max_value"], SEARCH_SPACE["axis_key_options"], SEARCH_SPACE["axis_value_options"]]
    
    raw_combinations = list(itertools.product(*values))
    valid_data = [dict(zip(keys, c)) for c in raw_combinations if is_valid_config(dict(zip(keys, c)))]
    df_all_configs = pd.DataFrame(valid_data)

    if df_all_configs.empty:
        raise RuntimeError("No valid configurations generated from search space")

    df_to_evaluate = df_all_configs
    df_cached = pd.DataFrame()

    source_cache_path = cr_cache_path()
    if EVALUATION_MODE == "smoke" and not os.path.isfile(source_cache_path):
        raise FileNotFoundError(f"Smoke mode requires a precomputed CR cache: {source_cache_path}")
    if os.path.exists(source_cache_path):
        try:
            logging.info(f"Found CR cache: {source_cache_path}")
            df_cache = pd.read_csv(source_cache_path)
            
            # Ensure tuple columns from CSV (read as strings) are converted back to tuples
            for col in ['axis_key', 'axis_value']:
                if col in df_cache.columns:
                    df_cache[col] = df_cache[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
            # Deduplicate cache just in case to avoid merge explosion
            df_cache = df_cache.drop_duplicates(subset=keys)

            # Left merge to find which configs are new (will have NaN for 'cr')
            df_merged = pd.merge(df_all_configs, df_cache[keys + ['cr']], on=keys, how='left')

            # Split into already cached and to-be-evaluated
            cached_mask = df_merged['cr'].notna()
            df_cached = df_merged[cached_mask].reset_index(drop=True)
            
            to_evaluate_indices = df_merged[df_merged['cr'].isna()].index
            df_to_evaluate = df_all_configs.iloc[to_evaluate_indices].reset_index(drop=True)
            
            logging.info(f"{len(df_cached)} configs found in cache. {len(df_to_evaluate)} new configs to evaluate.")
            
        except Exception as e:
            logging.warning(f"Could not load or parse CR cache '{source_cache_path}'. Re-evaluating all. Error: {e}")
            df_to_evaluate = df_all_configs
            df_cached = pd.DataFrame()
    
    df_newly_evaluated = pd.DataFrame()
    if not df_to_evaluate.empty:
        import torch
        from offline_search.evaluation.param_search.cr_evaluator import CompressionEvaluator

        logging.info(f"Evaluating CR for {len(df_to_evaluate)} configurations...")
        logging.info("=== Initializing Compression Evaluator ===")
        cr_evaluator = silent_call(
            CompressionEvaluator,
            model_name=MODEL_NAME,
            base_model_path=BASE_MODEL_PATH,
            base_config_path=BASE_CONFIG_PATH,
        )
        crs = []
        for i, (_, row) in enumerate(df_to_evaluate.iterrows()):
            if (i + 1) % 10 == 0: logging.info(f"Processing new CR... {i+1}/{len(df_to_evaluate)}")
            cr_val = silent_call(run_cr, cr_evaluator, row.to_dict())
            crs.append(cr_val)

        logging.info(f"Finished evaluating {len(df_to_evaluate)} new configurations.")

        # Explicitly release evaluator resources before accuracy evaluation.
        del cr_evaluator
        gc.collect()
        torch.cuda.empty_cache()        
        
        df_newly_evaluated = df_to_evaluate.copy()
        df_newly_evaluated['cr'] = crs
        
        # Append new valid results to the cache file immediately
        df_save = df_newly_evaluated[df_newly_evaluated['cr'] > 0].copy()
        if not df_save.empty:
            save_cols = keys + ['cr']
            os.makedirs(model_results_dir(), exist_ok=True)
            local_cache_path = os.path.join(model_results_dir(), CACHE_CSV_PATH)
            file_exists = os.path.exists(local_cache_path)
            df_save[save_cols].to_csv(local_cache_path, mode='a', index=False, header=not file_exists)
            logging.info(f"Appended {len(df_save)} new configurations to {local_cache_path}")
            # return

    # Combine cached and newly evaluated results
    df = pd.concat([df_cached, df_newly_evaluated], ignore_index=True)

    if EVALUATION_MODE == "smoke":
        logging.info("=== Initializing Deterministic Accuracy Proxy ===")
        acc_evaluator = SmokeAccuracyEvaluator()
    else:
        logging.info("=== Initializing Model Accuracy Evaluator ===")
        from offline_search.evaluation.param_search.acc_evaluator import AccuracyEvaluator

        acc_evaluator = silent_call(
            AccuracyEvaluator,
            model_name=MODEL_NAME,
            tasks=TASK_TO_SEARCH,
            limit=DATASET_LIMIT,
            batch_size=BATCH_SIZE,
            random_seed=SEED,
            base_model_path=BASE_MODEL_PATH,
            base_config_path=BASE_CONFIG_PATH,
            attn_implementation=ATTN_IMPLEMENTATION,
        )

    # The rest of the processing happens on the combined dataframe
    if len(df) == 0:
        raise RuntimeError("No valid compression-ratio configurations found")

    # Filter out invalid CRs for internal usage
    df = df[df["cr"] > 0].reset_index(drop=True)
    
    logging.info("Processing One-Hot Encodings for GP Model...")

    # Define numeric base feature columns.
    base_feature_cols = [
        "heads_selection", 
        "high_key_max_value", "high_value_max_value",
        "low_key_max_value", "low_value_max_value"
    ]
    
    # Define categorical or tuple-valued columns that should be one-hot encoded.
    cols_to_encode = ["transform_type", "axis_key", "axis_value"]
    
    # Generate one-hot columns and return the updated dataframe and GP feature list.
    df, gp_feature_cols = process_one_hot_encoding(df, cols_to_encode, base_feature_cols)
    
    logging.info(f"Features for GP Model: {gp_feature_cols}")

    # ================= Phase 2: Bayesian Optimization =================
    logging.info("=== Phase 2: Bayesian Optimization Search ===")
    
    logging.info(f"Target Accuracy: >= {TARGET_ACC_THRESHOLD:.4f}")
    
    bo = ConstraintAwareBO(df, seed=SEED)
    bo.feature_cols = gp_feature_cols 
    
    best_feasible_cr = 0.0
    best_config = None
    
    # ----------------- Cold Start -----------------
    feasible_configs = []
    evaluated_configs = []
    config_id = 0
    if len(df) > COLD_START_POINTS:
        # Sort by compression ratio.
        sorted_df = df.sort_values("cr")
        # Span the CR range and evaluate from aggressive to conservative.
        positions = np.linspace(len(sorted_df) - 1, 0, COLD_START_POINTS, dtype=int)
        candidates = [sorted_df.index[position] for position in positions]
        selected_indices = []
        seen = set()
        for idx in candidates:
            if idx not in seen:
                selected_indices.append(idx)
                seen.add(idx)
        
        initial_indices = np.array(selected_indices)
    else:
        initial_indices = np.random.choice(df.index, min(COLD_START_POINTS, len(df)), replace=False)
    logging.info("--- Cold Start ---")
    for cold_index, idx in enumerate(initial_indices, start=1):
        params = df.iloc[idx].to_dict()
        acc = silent_call(run_acc, acc_evaluator, params)
        bo.observed_indices.append(idx)
        bo.observed_accs.append(acc)
        evaluated_configs.append({**params, "accuracy": acc, "candidate_index": int(idx)})
        
        status = "FEASIBLE" if acc >= TARGET_ACC_THRESHOLD else "INFEASIBLE"
        logging.info(
            "[SEARCH][cold %d/%d] candidate=%d cr=%.4f accuracy=%.2f status=%s",
            cold_index, len(initial_indices), idx, params["cr"], acc, status,
        )
        if acc >= TARGET_ACC_THRESHOLD:
            # Record feasible config
            record = params.copy()
            record['accuracy'] = acc
            record['config_id'] = config_id
            config_id += 1
            feasible_configs.append(record)

            if params['cr'] > best_feasible_cr:
                best_feasible_cr = params['cr']
                best_config = params
                # Batch Pruning logic
                # cutoff_cr = best_feasible_cr - PRUNING_EPSILON / 2
                # prune_indices = df[df['cr'] < cutoff_cr].index.tolist()
                # new_skips = [x for x in prune_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
                # bo.skipped_indices.update(new_skips)
                # if new_skips:
                #     logging.info(f"   [Pruning] Skipped {len(new_skips)} configs with CR < {cutoff_cr:.4f}")
        elif acc <= TARGET_ACC_THRESHOLD - ACC_TOLERANCE:
            # condition_indices = bigger_pruning_select_by(df, params, idx, ["axis_key", "axis_value"])
            # new_skips = [x for x in condition_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
            # bo.skipped_indices.update(new_skips)
            # if new_skips:
            #     logging.info(f"   [Pruning] Conditionally Skipped {len(new_skips)} configs with CR > {(params['cr'] + 2 * PRUNING_EPSILON):.4f}")        
            continue

    # ----------------- BO Loop -----------------
    i = 0
    while(i < MAX_ITER):
        total_configs = len(df)
        visited_count = len(set(bo.observed_indices) | bo.skipped_indices)
        remaining_count = total_configs - visited_count
        logging.info(f"--- BO Iteration {i+1}/{MAX_ITER} | Remaining: {remaining_count}/{total_configs} ---")
        
        bo.fit()
        # Decay the exploration weight as the search progresses.
        if i < MAX_ITER / 5:
            current_weight = max(0.1, EXPLORATION_WEIGHT * (0.99 ** i))
        else:
            current_weight = max(0.1, EXPLORATION_WEIGHT * (0.99 ** i) / 2)
        result = bo.propose_next(TARGET_ACC_THRESHOLD, current_weight)
        next_idx, prob, pred_acc = result
        if next_idx is None:
            logging.warning("⚠️  No more candidates.")
            break        
        next_params = df.iloc[next_idx].to_dict()
        
        logging.info(
            "[SEARCH][bo %d/%d] propose candidate=%d cr=%.4f predicted_accuracy=%.2f feasible_probability=%.3f",
            i + 1, MAX_ITER, next_idx, next_params["cr"], pred_acc, prob,
        )
        
        # Single Point Pruning
        # if best_feasible_cr > 0 and next_params['cr'] <= best_feasible_cr:
        #     logging.info(f"⏭️  SKIPPING (Pruned). Reason: CR <= Best {best_feasible_cr:.4f}")
        #     bo.skipped_indices.add(next_idx)
        #     continue 
        
        logging.info(f"   TT={next_params['transform_type']}, HS={next_params['heads_selection']}, MV={next_params['high_key_max_value']} {next_params['high_value_max_value']} {next_params['low_key_max_value']} {next_params['low_value_max_value']}, AK={next_params['axis_key']}, AV={next_params['axis_value']}")
        real_acc = silent_call(run_acc, acc_evaluator, next_params)
        bo.observed_indices.append(next_idx)
        bo.observed_accs.append(real_acc)
        evaluated_configs.append({**next_params, "accuracy": real_acc, "candidate_index": int(next_idx)})
        
        if real_acc >= TARGET_ACC_THRESHOLD:
            bo.consecutive_fail_count = 0
            # Record feasible config
            record = next_params.copy()
            record['accuracy'] = real_acc
            record['config_id'] = config_id
            config_id += 1
            feasible_configs.append(record)

            logging.info(
                "[SEARCH][bo %d/%d] result candidate=%d accuracy=%.2f status=FEASIBLE",
                i + 1, MAX_ITER, next_idx, real_acc,
            )
            if next_params['cr'] > best_feasible_cr:
                best_feasible_cr = next_params['cr']
                best_config = next_params
                logging.info(f"🎉 NEW BEST! Max CR: {best_feasible_cr:.4f}")
                skipped_indices = smaller_pruning_select_by(df, next_params, next_idx, i, ["axis_key", "axis_value"])
                new_skips = [x for x in skipped_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
                bo.skipped_indices.update(new_skips)
                if new_skips:
                    logging.info(f"🧹 [Batch Pruning] Skipped {len(new_skips)} dominated configs")
            else:
                logging.info(f"⏭️ (CR <= Best)")
                skipped_indices = smaller_pruning_select_by(df, next_params, next_idx, i, ["axis_key", "axis_value"])
                new_skips = [x for x in skipped_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
                bo.skipped_indices.update(new_skips)
                if new_skips:
                    logging.info(f"🧹 [Batch Pruning] Skipped {len(new_skips)} dominated configs")
                # continue
        else:
            if i > MAX_ITER / 2:
                bo.consecutive_fail_count += 1
            logging.info(
                "[SEARCH][bo %d/%d] result candidate=%d accuracy=%.2f status=INFEASIBLE",
                i + 1, MAX_ITER, next_idx, real_acc,
            )
            if real_acc <= TARGET_ACC_THRESHOLD - ACC_TOLERANCE:
                skipped_indices = bigger_pruning_select_by(df, next_params, next_idx, i, ["axis_key", "axis_value"])
                new_skips = [x for x in skipped_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
                bo.skipped_indices.update(new_skips)
                if new_skips:
                    logging.info(f"🧹 [Batch Pruning] Conditionally Skipped {len(new_skips)} configs with CR > {(next_params['cr'] + 2 * PRUNING_EPSILON):.4f}")

        if check_early_stopping(df, bo):
            break

        i += 1

    logging.info("================ Search Finished ================")
    os.makedirs(model_results_dir(), exist_ok=True)
    final_path = os.path.join(model_results_dir(), FINAL_JSON_PATH)
    evaluated_path = os.path.join(model_results_dir(), "evaluated_configs.json")
    summary_path = os.path.join(model_results_dir(), "summary.json")
    if feasible_configs:
        logging.info(f"Found {len(feasible_configs)} feasible configurations.")
        if best_config:
             logging.info(f"Best Config (CR={best_feasible_cr:.4f}):")
             logging.info(json.dumps(best_config, indent=2, default=str))

        with open(final_path, "w") as f:
            json.dump(feasible_configs, f, indent=4, default=str)
        logging.info(f"Saved all feasible configurations to {final_path}")
    elif best_config:
        final = {k: v for k, v in best_config.items()}
        logging.info(json.dumps(final, indent=2, default=str))
        with open(final_path, "w") as f:
            json.dump(final, f, indent=4, default=str)
    else:
        with open(final_path, "w") as f:
            json.dump([], f)

    with open(evaluated_path, "w") as f:
        json.dump(evaluated_configs, f, indent=2, default=str)
    summary = {
        "status": "complete",
        "evaluation_mode": EVALUATION_MODE,
        "measured_profile": EVALUATION_MODE == "model",
        "model": MODEL_NAME,
        "tasks": TASK_TO_SEARCH,
        "dataset_limit": DATASET_LIMIT,
        "candidate_count": int(len(df)),
        "accuracy_evaluations": len(evaluated_configs),
        "feasible_count": len(feasible_configs),
        "accuracy_threshold": TARGET_ACC_THRESHOLD,
        "best_compression_ratio": best_feasible_cr if best_config is not None else None,
        "elapsed_seconds": round(time.monotonic() - started_at, 2),
        "feasible_results": final_path,
        "evaluated_results": evaluated_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(
        "Summary: candidates=%d evaluated=%d feasible=%d best_cr=%s elapsed=%.1fs",
        len(df), len(evaluated_configs), len(feasible_configs),
        f"{best_feasible_cr:.4f}" if best_config is not None else "n/a",
        summary["elapsed_seconds"],
    )
    logging.info(f"Summary JSON: {summary_path}")

if __name__ == "__main__":
    main()
