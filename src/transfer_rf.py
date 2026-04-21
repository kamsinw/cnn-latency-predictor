"""
Adapts a pretrained sklearn RandomForestRegressor to a new target
device/configuration by updating only the leaf prediction values,
while keeping the tree structure (splits + thresholds) unchanged.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_utils import load_and_preprocess
from evaluate import mape


# ---------------------------------------------------------------------------
# Core transfer function
# ---------------------------------------------------------------------------

def retarget_random_forest(rf_source, X_target, y_target, alpha=1.0):
    """
    Adapt a pretrained RandomForestRegressor to target-domain data
    using leaf re-targeting.

    The tree structure (split features and thresholds) is kept unchanged.
    Only the leaf prediction values are updated using the K target samples.

 
    """
    X_target = np.asarray(X_target)
    y_target = np.asarray(y_target)

    rf_adapted = copy.deepcopy(rf_source)

    total_leaves_updated = 0
    total_leaves_unchanged = 0

    for tree in rf_adapted.estimators_:
        # Route every target sample to its leaf
        leaf_ids = tree.apply(X_target)           # shape (K,)
        unique_leaves = np.unique(leaf_ids)

        leaves_updated = 0
        leaves_unchanged = 0

        for leaf in unique_leaves:
            mask = leaf_ids == leaf
            samples_in_leaf = y_target[mask]

            if len(samples_in_leaf) == 0:
                leaves_unchanged += 1
                continue

            old_value = tree.tree_.value[leaf, 0, 0]
            new_mean  = samples_in_leaf.mean()

            # Blend old and new value according to alpha
            blended = alpha * new_mean + (1.0 - alpha) * old_value
            tree.tree_.value[leaf, 0, 0] = blended
            leaves_updated += 1

        # Count leaves that received no target samples
        all_leaves = np.where(tree.tree_.children_left == -1)[0]
        leaves_unchanged += len(all_leaves) - leaves_updated

        total_leaves_updated   += leaves_updated
        total_leaves_unchanged += leaves_unchanged

    n_trees = len(rf_adapted.estimators_)
    print(f"  Leaves updated   : {total_leaves_updated}  "
          f"(avg {total_leaves_updated / n_trees:.1f} per tree)")
    print(f"  Leaves unchanged : {total_leaves_unchanged}  "
          f"(avg {total_leaves_unchanged / n_trees:.1f} per tree)")

    return rf_adapted


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, label="Model"):
    """
    Compute and print MAE, RMSE, and MAPE for a fitted model.


    """
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape_val = mape(np.asarray(y_test), y_pred)

    print(f"  [{label}]  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape_val:.2f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape_val}



def k_sweep(rf_source, X_target_train, y_target_train, X_target_test, y_target_test,
            k_values=None, alpha=1.0, metric="mape"):
    """
    Evaluate transfer quality as a function of K (number of target samples).

    For each K, a fresh adapted model is created using the first K samples
    of X_target_train / y_target_train and evaluated on the held-out test set.

"""
    if k_values is None:
        k_values = [10, 25, 50, 100, 200, 500]

    # Filter to K values that don't exceed the available calibration pool
    k_values = [k for k in k_values if k <= len(X_target_train)]

    source_metrics = evaluate_model(rf_source, X_target_test, y_target_test,
                                    label="Source (no transfer)")
    baseline_error = source_metrics[metric]

    errors = []
    for k in k_values:
        print(f"\n  K = {k}")
        rf_adapted = retarget_random_forest(
            rf_source,
            X_target_train[:k],
            y_target_train[:k],
            alpha=alpha,
        )
        metrics = evaluate_model(rf_adapted, X_target_test, y_target_test,
                                 label=f"Adapted K={k}")
        errors.append(metrics[metric])

    return k_values, errors, baseline_error


def plot_k_sweep(k_values, errors, baseline_error,
                 source_col, target_col, metric="MAPE", save_path=None):
    """Plot error vs K for a transfer learning sweep."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, errors, marker="o", linewidth=2, label="Adapted (leaf re-targeting)")
    ax.axhline(baseline_error, color="red", linestyle="--", linewidth=1.5,
               label=f"Source (no transfer): {baseline_error:.2f}%")
    ax.set_xlabel("K (number of target calibration samples)")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(f"Transfer: {source_col}  →  {target_col}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Plot saved to {save_path}")
    plt.show()



if __name__ == "__main__":
    import os
    import joblib

    SOURCE_COL = "pixel4|1large|quant"
    TARGET_COL = "pixel4|1large1med|quant"
    OP_TYPE    = "CONV_2D"
    MODEL_DIR  = "../models"
    RESULTS_DIR = "../results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading source data ...")
    X_source, y_source = load_and_preprocess(target_col=SOURCE_COL, op_type=OP_TYPE)
    X_source = X_source.values
    y_source = y_source.values

    print("Loading target data ...")
    X_target, y_target = load_and_preprocess(target_col=TARGET_COL, op_type=OP_TYPE)
    X_target = X_target.values
    y_target = y_target.values

   
    source_col_safe = SOURCE_COL.replace("|", "_")
    source_model_path = os.path.join(MODEL_DIR, f"rf_{OP_TYPE}_{source_col_safe}.joblib")

    if os.path.exists(source_model_path):
        print(f"\nLoading pretrained source model from {source_model_path}")
        rf_source = joblib.load(source_model_path)
    else:
        print("\nTraining source model ...")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_source, y_source, test_size=0.2, random_state=42
        )
        rf_source = RandomForestRegressor(
            n_estimators=300, max_depth=None, max_features=0.75,
            random_state=42, n_jobs=-1
        )
        rf_source.fit(X_tr, y_tr)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(rf_source, source_model_path)
        print(f"  Source model saved to {source_model_path}")
        print("  Source test MAPE:", end=" ")
        evaluate_model(rf_source, X_te, y_te, label="Source on source test set")

    # Hold out 20 % of target data as a fixed test set.
    # The remaining 80 % is the calibration pool from which we draw K samples.
    X_cal, X_test_t, y_cal, y_test_t = train_test_split(
        X_target, y_target, test_size=0.2, random_state=42
    )

  
    K = 100
    print(f"\n{'='*60}")
    print(f"Transfer: {SOURCE_COL}  →  {TARGET_COL}  (K={K})")
    print(f"{'='*60}")

    print("\nSource model on target test set (no transfer):")
    evaluate_model(rf_source, X_test_t, y_test_t, label="Source")

    print(f"\nAdapting with K={K} target samples ...")
    rf_adapted = retarget_random_forest(rf_source, X_cal[:K], y_cal[:K], alpha=1.0)

    print("\nAdapted model on target test set:")
    evaluate_model(rf_adapted, X_test_t, y_test_t, label="Adapted")

    # Save adapted model
    target_col_safe = TARGET_COL.replace("|", "_")
    adapted_path = os.path.join(
        MODEL_DIR, f"rf_{OP_TYPE}_{source_col_safe}_to_{target_col_safe}_K{K}.joblib"
    )
    joblib.dump(rf_adapted, adapted_path)
    print(f"\nAdapted model saved to {adapted_path}")


    print(f"\n{'='*60}")
    print("K-sweep experiment")
    print(f"{'='*60}")

    k_values = [10, 25, 50, 100, 200, 500, 1000]
    k_vals, errs, baseline = k_sweep(
        rf_source, X_cal, y_cal, X_test_t, y_test_t,
        k_values=k_values, alpha=1.0, metric="mape",
    )

    plot_path = os.path.join(
        RESULTS_DIR,
        f"transfer_k_sweep_{source_col_safe}_to_{target_col_safe}.png"
    )
    plot_k_sweep(k_vals, errs, baseline,
                 source_col=SOURCE_COL, target_col=TARGET_COL,
                 metric="MAPE", save_path=plot_path)
