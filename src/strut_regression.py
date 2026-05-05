"""
Idea is:
(per tree)
  1. Walk the tree recursively from the root.
  2. At each internal node:
       a. If fewer than min_samples_leaf target samples reach the node → prune
          it to a leaf (keep source prediction value).
       b. Otherwise, search candidate thresholds along the SAME feature as the
          source tree (feature index is frozen). Score each candidate:
            score = (1-λ) × MSE_gain  -  λ × distribution_divergence
          where distribution_divergence measures how far the source child means
          are from the target child means (regression analogue of STRUT's DG).
       c. Update the threshold with the best-scoring candidate.
       d. Split target samples along the new threshold and recurse.
  3. At leaf nodes: replace stored value with mean(y_target).


"""

import copy
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from data_utils import load_and_preprocess
from evaluate import mape
from transfer_rf import retarget_random_forest, evaluate_model


_TREE_LEAF      = -1   # children_left / children_right value for a leaf
_TREE_UNDEFINED = -2   # feature value that marks a node as a leaf



def _make_writeable(dtree):
    """Ensure all mutable sklearn tree arrays are writeable after deepcopy."""
    for attr in ("threshold", "feature", "children_left", "children_right",
                 "value", "n_node_samples", "weighted_n_node_samples", "impurity"):
        arr = getattr(dtree.tree_, attr, None)
        if arr is not None and not arr.flags.writeable:
            arr.flags.writeable = True


def _prune_subtree(tree, node_index):
    """
    Recursively orphan all nodes below node_index and turn node_index into
    a leaf.  The prediction (tree_.value) at node_index is left unchanged so
    the source value acts as a fallback.
    """
    left  = tree.children_left[node_index]
    right = tree.children_right[node_index]
    if left != _TREE_LEAF:
        _prune_subtree(tree, left)
    if right != _TREE_LEAF:
        _prune_subtree(tree, right)
    tree.children_left[node_index]  = _TREE_LEAF
    tree.children_right[node_index] = _TREE_LEAF
    tree.feature[node_index]        = _TREE_UNDEFINED
    tree.threshold[node_index]      = -2.0


def _variance_reduction(y_parent, y_left, y_right):
    """
    MSE / variance reduction from a candidate split.
    Regression analogue of Information Gain.
    """
    n, n_l, n_r = len(y_parent), len(y_left), len(y_right)
    if n == 0:
        return 0.0
    var_total = y_parent.var() if n > 1 else 0.0
    var_l     = y_left.var()   if n_l > 1 else 0.0
    var_r     = y_right.var()  if n_r > 1 else 0.0
    return var_total - (n_l / n) * var_l - (n_r / n) * var_r


def _distribution_divergence(src_mean_l, src_mean_r, y_target_l, y_target_r, scale):
    """
    Weighted normalised distance between source child means and target child
    means.  Regression analogue of STRUT's DG (divergence score).

    Lower is better (small divergence = source and target children are
    compatible — no big distribution shift at this node).
    """
    if scale < 1e-8:
        return 0.0
    n_l = len(y_target_l)
    n_r = len(y_target_r)
    n   = n_l + n_r
    if n == 0:
        return 0.0

    tgt_mean_l = y_target_l.mean() if n_l > 0 else src_mean_l
    tgt_mean_r = y_target_r.mean() if n_r > 0 else src_mean_r

    div_l = abs(src_mean_l - tgt_mean_l) / scale
    div_r = abs(src_mean_r - tgt_mean_r) / scale
    return (n_l / n) * div_l + (n_r / n) * div_r


def _score_threshold(t, x_phi, y_node, src_mean_l, src_mean_r,
                     scale, lambda_div, use_divergence):
    """Score a single threshold candidate. Used by _find_best_threshold."""
    left_mask  = x_phi <= t
    right_mask = ~left_mask
    y_l = y_node[left_mask]
    y_r = y_node[right_mask]
    if len(y_l) == 0 or len(y_r) == 0:
        return -np.inf
    gain = _variance_reduction(y_node, y_l, y_r)
    if use_divergence:
        div = _distribution_divergence(src_mean_l, src_mean_r, y_l, y_r, scale)
        return (1.0 - lambda_div) * gain - lambda_div * div * scale
    return gain


def _find_best_threshold(X_node, y_node, phi,
                         src_mean_l, src_mean_r, src_threshold,
                         lambda_div=0.5, use_divergence=True):
    x_phi       = X_node[:, phi]
    unique_vals = np.unique(x_phi)

    if len(unique_vals) < 2:
        return src_threshold

    scale = y_node.std() if len(y_node) > 1 else 1.0

    # Score the source threshold first — use it as the baseline to beat
    src_score  = _score_threshold(src_threshold, x_phi, y_node,
                                  src_mean_l, src_mean_r,
                                  scale, lambda_div, use_divergence)
    best_score     = src_score
    best_threshold = src_threshold   # fallback: keep source if nothing beats it

    candidates = (unique_vals[:-1] + unique_vals[1:]) / 2.0
    for t in candidates:
        score = _score_threshold(t, x_phi, y_node,
                                 src_mean_l, src_mean_r,
                                 scale, lambda_div, use_divergence)
        if score > best_score:
            best_score     = score
            best_threshold = t

    return best_threshold



def _strut_node(dtree, node_index, X_node, y_node,
                K_total, tau=0.1,
                lambda_div=0.5, use_divergence=True,
                stats=None):
    """
    Recursively apply Regression-STRUT to one node.

    tau controls selectivity: a node is only updated if the fraction of the
    total K target samples reaching it is >= tau.  Nodes below the threshold
    are traversed with the original (frozen) threshold so routing is preserved,
    but their split and leaf values are left unchanged.
    """
    tree    = dtree.tree_
    is_leaf = tree.children_left[node_index] == _TREE_LEAF

    # --- Leaf node ---
    if is_leaf:
        if len(y_node) > 0:
            tree.value[node_index, 0, 0]             = y_node.mean()
            tree.n_node_samples[node_index]          = len(y_node)
            tree.weighted_n_node_samples[node_index] = float(len(y_node))
            if stats is not None:
                stats["leaves_updated"] += 1
        else:
            if stats is not None:
                stats["leaves_unchanged"] += 1
        return

    phi           = tree.feature[node_index]
    src_threshold = tree.threshold[node_index]
    left_child    = tree.children_left[node_index]
    right_child   = tree.children_right[node_index]

    # coverage = fraction of all K samples that reached this node
    coverage = len(y_node) / K_total if K_total > 0 else 0.0

    if coverage < tau or len(y_node) == 0:
        # Not enough target data here — keep this node frozen, but still
        # recurse into children using the ORIGINAL threshold so leaf updates
        # can happen deeper down where coverage may be sufficient.
        if len(y_node) > 0:
            left_mask  = X_node[:, phi] <= src_threshold
            right_mask = ~left_mask
            _strut_node(dtree, left_child,
                        X_node[left_mask],  y_node[left_mask],
                        K_total, tau, lambda_div, use_divergence, stats)
            _strut_node(dtree, right_child,
                        X_node[right_mask], y_node[right_mask],
                        K_total, tau, lambda_div, use_divergence, stats)
        else:
            if stats is not None:
                stats["leaves_unchanged"] += 1
        return

    # --- Coverage met: update this node ---
    src_mean_l = tree.value[left_child,  0, 0]
    src_mean_r = tree.value[right_child, 0, 0]

    new_threshold = _find_best_threshold(
        X_node, y_node, phi,
        src_mean_l, src_mean_r, src_threshold,
        lambda_div=lambda_div, use_divergence=use_divergence,
    )
    tree.threshold[node_index]               = new_threshold
    tree.value[node_index, 0, 0]             = y_node.mean()
    tree.n_node_samples[node_index]          = len(y_node)
    tree.weighted_n_node_samples[node_index] = float(len(y_node))
    if stats is not None:
        stats["nodes_updated"] += 1

    left_mask  = X_node[:, phi] <= new_threshold
    right_mask = ~left_mask

    _strut_node(dtree, left_child,
                X_node[left_mask],  y_node[left_mask],
                K_total, tau, lambda_div, use_divergence, stats)
    _strut_node(dtree, right_child,
                X_node[right_mask], y_node[right_mask],
                K_total, tau, lambda_div, use_divergence, stats)



def global_scale_rf(rf_source, y_source, y_target_small):
    """
    Pre-scale all leaf values by median(y_target) / median(y_source).
    Corrects the systematic offset between source and target before
    any structural adaptation, reducing the damage noisy threshold
    searches can cause at small K.

    Returns a deep copy of rf_source with scaled leaf values.
    """
    ratio = np.median(y_target_small) / np.median(y_source)
    rf_scaled = copy.deepcopy(rf_source)
    for tree in rf_scaled.estimators_:
        if not tree.tree_.value.flags.writeable:
            tree.tree_.value.flags.writeable = True
        tree.tree_.value[:, 0, 0] *= ratio
    print(f"  Global scale ratio: {ratio:.4f}  "
          f"(median src={np.median(y_source):.1f}, "
          f"median tgt={np.median(y_target_small):.1f})", flush=True)
    return rf_scaled


def strut_regression_rf(rf_source, X_target, y_target,
                         y_source=None,
                         tau=0.1, lambda_div=0.5,
                         use_divergence=True, global_scale=True):
    """
    Apply Regression-STRUT with selective path updates to every tree.

    
    """
    X_target = np.asarray(X_target)
    y_target = np.asarray(y_target)
    K_total  = len(y_target)

    # Step 1: global scale correction (Fix 3)
    if global_scale and y_source is not None:
        rf_start = global_scale_rf(rf_source, y_source, y_target)
    else:
        rf_start = rf_source

    rf_adapted = copy.deepcopy(rf_start)

    agg = {"leaves_updated": 0, "leaves_unchanged": 0, "nodes_updated": 0}

    for dtree in rf_adapted.estimators_:
        _make_writeable(dtree)
        stats = {"leaves_updated": 0, "leaves_unchanged": 0, "nodes_updated": 0}
        _strut_node(dtree, 0, X_target, y_target,
                    K_total=K_total, tau=tau,
                    lambda_div=lambda_div,
                    use_divergence=use_divergence,
                    stats=stats)
        for k, v in stats.items():
            agg[k] += v

    n = len(rf_adapted.estimators_)
    print(f"  Leaves updated   : {agg['leaves_updated']:5d}  "
          f"(avg {agg['leaves_updated']  / n:.1f}/tree)", flush=True)
    print(f"  Leaves unchanged : {agg['leaves_unchanged']:5d}  "
          f"(avg {agg['leaves_unchanged'] / n:.1f}/tree)", flush=True)
    print(f"  Internal updated : {agg['nodes_updated']:5d}  "
          f"(avg {agg['nodes_updated']   / n:.1f}/tree)", flush=True)

    return rf_adapted



def k_sweep_comparison(rf_source,
                        X_cal, y_cal, X_test, y_test,
                        y_source=None,
                        k_values=None, alpha=1.0,
                        tau=0.1, lambda_div=0.5):
    """
    Compare leaf re-targeting vs Regression-STRUT (selective) across K values.

    Returns dict with keys 'k_values', 'leaf_errors', 'strut_errors', 'baseline'.
    """
    if k_values is None:
        k_values = [10, 25, 50, 100, 200, 500, 1000]
    k_values = [k for k in k_values if k <= len(X_cal)]

    src_mape = evaluate_model(rf_source, X_test, y_test,
                               label="Source (no transfer)")["mape"]

    leaf_errors  = []
    strut_errors = []

    for k in k_values:
        Xk, yk = X_cal[:k], y_cal[:k]

        print(f"\n── K = {k}  (tau={tau}) ──────────────────────────", flush=True)

        print("  [Leaf re-targeting]", flush=True)
        rf_leaf = retarget_random_forest(rf_source, Xk, yk, alpha=alpha)
        leaf_m  = evaluate_model(rf_leaf, X_test, y_test,
                                  label=f"Leaf K={k}")["mape"]
        leaf_errors.append(leaf_m)

        print("  [Regression-STRUT (selective)]", flush=True)
        rf_strut = strut_regression_rf(rf_source, Xk, yk,
                                        y_source=y_source,
                                        tau=tau, lambda_div=lambda_div)
        strut_m  = evaluate_model(rf_strut, X_test, y_test,
                                   label=f"STRUT K={k}")["mape"]
        strut_errors.append(strut_m)

    return {
        "k_values":    k_values,
        "leaf_errors": leaf_errors,
        "strut_errors": strut_errors,
        "baseline":    src_mape,
    }


def plot_comparison(results, source_col, target_col, tau=None, save_path=None):
    """Plot MAPE vs K for leaf re-targeting vs Regression-STRUT."""
    k   = results["k_values"]
    tau_str = f"  (tau={tau})" if tau is not None else ""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k, results["leaf_errors"],  marker="o", linewidth=2, label="Leaf re-targeting")
    ax.plot(k, results["strut_errors"], marker="s", linewidth=2,
            label=f"Regression-STRUT (selective{tau_str})")
    ax.axhline(results["baseline"], color="red", linestyle="--", linewidth=1.5,
               label=f"Source (no transfer): {results['baseline']:.2f}%")
    ax.set_xlabel("K  (calibration samples from target)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(f"Transfer comparison{tau_str}\n{source_col}  →  {target_col}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Plot saved → {save_path}")
    plt.show()



if __name__ == "__main__":
    SOURCE_COL = "pixel4|1large|quant"
    TARGET_COL = "pixel4|1large1med|quant"
    OP_TYPE    = "CONV_2D"
    MODEL_DIR  = "../models"
    RESULTS_DIR = "../results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading source data ...", flush=True)
    X_source, y_source = load_and_preprocess(target_col=SOURCE_COL, op_type=OP_TYPE)
    X_source, y_source = X_source.values, y_source.values

    print("Loading target data ...", flush=True)
    X_target, y_target = load_and_preprocess(target_col=TARGET_COL, op_type=OP_TYPE)
    X_target, y_target = X_target.values, y_target.values

    src_safe  = SOURCE_COL.replace("|", "_")
    src_path  = os.path.join(MODEL_DIR, f"rf_{OP_TYPE}_{src_safe}.joblib")

    if os.path.exists(src_path):
        print(f"\nLoading source model from {src_path}", flush=True)
        rf_source = joblib.load(src_path)
    else:
        print("\nTraining source model ...", flush=True)
        Xtr, Xte, ytr, yte = train_test_split(
            X_source, y_source, test_size=0.2, random_state=42
        )
        rf_source = RandomForestRegressor(
            n_estimators=50, max_depth=15, max_features=0.75,
            random_state=42, n_jobs=-1
        )
        rf_source.fit(Xtr, ytr)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(rf_source, src_path)
        print(f"  Saved to {src_path}")
        evaluate_model(rf_source, Xte, yte, label="Source on source test set")

    TAU = 0.1   # coverage threshold: update node only if >= 10% of K reach it

    X_cal, X_test_t, y_cal, y_test_t = train_test_split(
        X_target, y_target, test_size=0.2, random_state=42
    )

    K = 100
    print(f"\n{'='*65}", flush=True)
    print(f"Single transfer example  |  K={K}  tau={TAU}", flush=True)
    print(f"  {SOURCE_COL}  →  {TARGET_COL}", flush=True)
    print(f"{'='*65}", flush=True)

    print("\nSource model (no transfer):", flush=True)
    evaluate_model(rf_source, X_test_t, y_test_t, label="Source")

    print(f"\nLeaf re-targeting  (K={K}):", flush=True)
    rf_leaf = retarget_random_forest(rf_source, X_cal[:K], y_cal[:K], alpha=1.0)
    evaluate_model(rf_leaf, X_test_t, y_test_t, label="Leaf-retarget")

    print(f"\nRegression-STRUT (selective, K={K}, tau={TAU}, lambda=0.5):", flush=True)
    rf_strut = strut_regression_rf(
        rf_source, X_cal[:K], y_cal[:K],
        y_source=y_source,
        tau=TAU, lambda_div=0.5,
    )
    evaluate_model(rf_strut, X_test_t, y_test_t, label="Regression-STRUT")

    tgt_safe  = TARGET_COL.replace("|", "_")
    strut_path = os.path.join(
        MODEL_DIR, f"rf_{OP_TYPE}_{src_safe}_to_{tgt_safe}_strut_K{K}.joblib"
    )
    joblib.dump(rf_strut, strut_path)
    print(f"\n  Regression-STRUT model saved → {strut_path}", flush=True)

    print(f"\n{'='*65}", flush=True)
    print(f"K-sweep  |  Leaf re-targeting vs Regression-STRUT (tau={TAU})", flush=True)
    print(f"{'='*65}", flush=True)

    results = k_sweep_comparison(
        rf_source, X_cal, y_cal, X_test_t, y_test_t,
        y_source=y_source,
        k_values=[10, 25, 50, 100, 200, 500, 1000],
        alpha=1.0, tau=TAU, lambda_div=0.5,
    )

    plot_path = os.path.join(
        RESULTS_DIR,
        f"strut_vs_leaf_{src_safe}_to_{tgt_safe}.png"
    )
    plot_comparison(results, SOURCE_COL, TARGET_COL, tau=TAU, save_path=plot_path)
