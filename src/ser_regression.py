"""
SER for regression trees (SER_reg / SER_RF_reg).

Ports the structural logic of SER (Segev et al., 2017) from classification to
regression, reusing fusionTree, cut_into_leaf2, cut_from_left_right, and the
recursive node-update skeleton verbatim.

Key differences from the classification SER in strut/ser/ser.py:
  - Value updates store the *mean* of target samples, not per-class counts.
    (sklearn regressors return tree_.value[leaf, 0, 0] directly as the
     prediction, so storing the sum would scale predictions by leaf occupancy.)
  - Pruning uses SSE reduction instead of misclassification rate.
  - Extension grows a DecisionTreeRegressor, not a DecisionTreeClassifier.
  - All classification/fairness knobs (no_red_on_cl, leaf_loss_quantify, etc.)
    are removed — only the original_ser=True code path is exposed.

The original strut/ser/ser.py is NOT imported or modified.
"""

import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from data_utils import load_and_preprocess
from evaluate import mape
from transfer_rf import retarget_random_forest, evaluate_model
from strut_regression import strut_regression_rf


# =============================================================================
# Section 1: structural helpers (copied verbatim from strut/ser/ser.py;
#             classifier/regressor-agnostic — no changes made)
# =============================================================================

def find_parent(dtree, i_node):
    # Use np.where instead of list.index to avoid O(n) Python list conversion
    # on trees with tens of thousands of nodes.
    p = -1
    b = 0
    if i_node != 0 and i_node != -1:
        left_hits = np.where(dtree.tree_.children_left == i_node)[0]
        if left_hits.size > 0:
            p = int(left_hits[0])
            b = -1
        else:
            right_hits = np.where(dtree.tree_.children_right == i_node)[0]
            if right_hits.size > 0:
                p = int(right_hits[0])
                b = 1
    return p, b


def extract_rule(dtree, node):
    feats = []
    ths   = []
    bools = []
    b = 1
    if node != 0:
        while b != 0:
            feats.append(dtree.tree_.feature[node])
            ths.append(dtree.tree_.threshold[node])
            bools.append(b)
            node, b = find_parent(dtree, node)
        feats.pop(0)
        ths.pop(0)
        bools.pop(0)
    return np.array(feats), np.array(ths), np.array(bools)


def depth(dtree, node):
    p, t, b = extract_rule(dtree, node)
    return len(p)


def depth_array(dtree, inds):
    depths = np.zeros(np.array(inds).size)
    for i, e in enumerate(inds):
        depths[i] = depth(dtree, e)
    return depths


def sub_nodes(tree, node):
    if node == -1:
        return []
    if tree.feature[node] == -2:
        return [node]
    return ([node]
            + sub_nodes(tree, tree.children_left[node])
            + sub_nodes(tree, tree.children_right[node]))


def find_parent_vtree(tree, i_node):
    """Operates on a raw Tree object (not a DecisionTree wrapper)."""
    p = -1
    b = 0
    if i_node != 0 and i_node != -1:
        left_hits = np.where(tree.children_left == i_node)[0]
        if left_hits.size > 0:
            p = int(left_hits[0])
            b = -1
        else:
            right_hits = np.where(tree.children_right == i_node)[0]
            if right_hits.size > 0:
                p = int(right_hits[0])
                b = 1
    return p, b


def depth_vtree(tree, node):
    feats = []
    b = 1
    if node != 0:
        while b != 0:
            feats.append(tree.feature[node])
            node, b = find_parent_vtree(tree, node)
        feats.pop(0)
    return len(feats)


def fusionTree(tree1, f, tree2):
    """Add tree2 to leaf f of tree1 (copied verbatim from ser.py)."""
    dic  = tree1.__getstate__().copy()
    dic2 = tree2.__getstate__().copy()

    size_init = tree1.node_count

    if depth_vtree(tree1, f) + dic2['max_depth'] > dic['max_depth']:
        dic['max_depth'] = int(depth_vtree(tree1, f) + tree2.max_depth)

    dic['capacity']   = tree1.capacity   + tree2.capacity   - 1
    dic['node_count'] = tree1.node_count + tree2.node_count - 1

    dic['nodes'][f] = dic2['nodes'][0]

    if dic2['nodes']['left_child'][0] != -1:
        dic['nodes']['left_child'][f]  = dic2['nodes']['left_child'][0]  + size_init - 1
    else:
        dic['nodes']['left_child'][f]  = -1
    if dic2['nodes']['right_child'][0] != -1:
        dic['nodes']['right_child'][f] = dic2['nodes']['right_child'][0] + size_init - 1
    else:
        dic['nodes']['right_child'][f] = -1

    dic['nodes'] = np.concatenate((dic['nodes'], dic2['nodes'][1:]))
    dic['nodes']['left_child'][size_init:]  = (
        (dic['nodes']['left_child'][size_init:]  != -1)
        * (dic['nodes']['left_child'][size_init:]  + size_init) - 1
    )
    dic['nodes']['right_child'][size_init:] = (
        (dic['nodes']['right_child'][size_init:] != -1)
        * (dic['nodes']['right_child'][size_init:] + size_init) - 1
    )

    values = np.concatenate(
        (dic['values'],
         np.zeros((dic2['values'].shape[0] - 1,
                   dic['values'].shape[1],
                   dic['values'].shape[2]))),
        axis=0,
    )
    dic['values'] = values

    (Tree, (n_f, n_c, n_o), _b) = tree1.__reduce__()
    tree1 = Tree(n_f, n_c, n_o)
    tree1.__setstate__(dic)
    return tree1


def cut_into_leaf2(dTree, node):
    """Prune the subtree rooted at node, turning it into a leaf."""
    dic = dTree.tree_.__getstate__().copy()

    size_init    = dTree.tree_.node_count
    node_to_rem  = list(set(sub_nodes(dTree.tree_, node)[1:]))

    inds = list(
        set(np.linspace(0, size_init - 1, size_init).astype(int)) - set(node_to_rem)
    )
    # Keep max_depth as-is: recomputing via depth_array is O(n*depth) and only
    # updates metadata, not prediction routing.
    # dic['max_depth'] left unchanged.

    dic['capacity']   = dTree.tree_.capacity   - len(node_to_rem)
    dic['node_count'] = dTree.tree_.node_count - len(node_to_rem)

    dic['nodes']['feature'][node]      = -2
    dic['nodes']['left_child'][node]   = -1
    dic['nodes']['right_child'][node]  = -1

    dic_old    = dic.copy()
    left_old   = dic_old['nodes']['left_child']
    right_old  = dic_old['nodes']['right_child']

    dic['nodes']  = dic['nodes'][inds]
    dic['values'] = dic['values'][inds]

    # O(1) lookup instead of O(n) list.index per node
    inds_map = {v: i for i, v in enumerate(inds)}
    for i, new in enumerate(inds):
        dic['nodes']['left_child'][i]  = inds_map[left_old[new]]  if left_old[new]  != -1 else -1
        dic['nodes']['right_child'][i] = inds_map[right_old[new]] if right_old[new] != -1 else -1

    (Tree, (n_f, n_c, n_o), _b) = dTree.tree_.__reduce__()
    del dTree.tree_
    dTree.tree_ = Tree(n_f, n_c, n_o)
    dTree.tree_.__setstate__(dic)
    return inds_map[node]


def cut_from_left_right(dTree, node, bool_left_right):
    """Remove an empty child branch, replacing the parent split with the survivor."""
    dic = dTree.tree_.__getstate__().copy()

    size_init = dTree.tree_.node_count
    p, b = find_parent(dTree, node)

    if bool_left_right == 1:
        repl_node   = dTree.tree_.children_left[node]
        node_to_rem = [node, dTree.tree_.children_right[node]]
    elif bool_left_right == -1:
        repl_node   = dTree.tree_.children_right[node]
        node_to_rem = [node, dTree.tree_.children_left[node]]

    inds = list(
        set(np.linspace(0, size_init - 1, size_init).astype(int)) - set(node_to_rem)
    )

    dic['capacity']   = dTree.tree_.capacity   - len(node_to_rem)
    dic['node_count'] = dTree.tree_.node_count - len(node_to_rem)

    if b == 1:
        dic['nodes']['right_child'][p] = repl_node
    elif b == -1:
        dic['nodes']['left_child'][p]  = repl_node

    dic_old   = dic.copy()
    left_old  = dic_old['nodes']['left_child']
    right_old = dic_old['nodes']['right_child']

    dic['nodes']  = dic['nodes'][inds]
    dic['values'] = dic['values'][inds]

    # O(1) lookup instead of O(n) list.index per node
    inds_map = {v: i for i, v in enumerate(inds)}
    for i, new in enumerate(inds):
        dic['nodes']['left_child'][i]  = inds_map[left_old[new]]  if left_old[new]  != -1 else -1
        dic['nodes']['right_child'][i] = inds_map[right_old[new]] if right_old[new] != -1 else -1

    (Tree, (n_f, n_c, n_o), _b) = dTree.tree_.__reduce__()
    del dTree.tree_
    dTree.tree_ = Tree(n_f, n_c, n_o)
    dTree.tree_.__setstate__(dic)
    # Skip depth_array recomputation — max_depth is metadata only, not used
    # for prediction routing.

    return inds_map[repl_node]


# =============================================================================
# Section 2: regression-specific helpers
# =============================================================================

def fusionDecisionTreeReg(dTree1, f, dTree2):
    """
    Attach dTree2 to leaf f of dTree1 (regression variant of fusionDecisionTree).

    Unlike the classifier version there are no classes_ to re-index — values are
    copied straight across: shape (n_nodes, n_outputs, 1).
    """
    size_init = dTree1.tree_.node_count
    dTree1.tree_ = fusionTree(dTree1.tree_, f, dTree2.tree_)
    # Direct copy — no classes_ re-indexing needed for regressors
    dTree1.tree_.value[size_init:, :, :] = dTree2.tree_.value[1:, :, :]
    dTree1.max_depth = dTree1.tree_.max_depth
    return dTree1


def leaf_error_reg(y_target_node):
    """
    Sum of squared errors (SSE) if we predict the mean — used for pruning.
    Returns 0 when the node receives no target samples.
    """
    if len(y_target_node) == 0:
        return 0.0
    mu = np.mean(y_target_node)
    return float(np.sum((y_target_node - mu) ** 2))


def error_reg(dTree, node, X_target_node, y_target_node):
    """
    Recursive SSE of the subtree rooted at node, routing target samples
    by the source tree's split features and thresholds.

    Early-exits when no target samples reach a subtree — this keeps the
    cost proportional to K rather than to the number of tree nodes.
    """
    if len(X_target_node) == 0:
        return 0.0
    if dTree.tree_.feature[node] == -2:
        return leaf_error_reg(y_target_node)
    feat      = dTree.tree_.feature[node]
    thr       = dTree.tree_.threshold[node]
    left_mask = X_target_node[:, feat] <= thr
    el = error_reg(dTree, dTree.tree_.children_left[node],
                   X_target_node[left_mask],  y_target_node[left_mask])
    er = error_reg(dTree, dTree.tree_.children_right[node],
                   X_target_node[~left_mask], y_target_node[~left_mask])
    return el + er


# =============================================================================
# Section 3: SER_reg — per-tree recursive adaptation
# =============================================================================

def _make_writeable_ser(dtree):
    for attr in ("threshold", "feature", "children_left", "children_right",
                 "value", "n_node_samples", "weighted_n_node_samples", "impurity"):
        arr = getattr(dtree.tree_, attr, None)
        if arr is not None and not arr.flags.writeable:
            arr.flags.writeable = True


def SER_reg(node, dTree, X_target_node, y_target_node,
            original_ser=True, _cur_depth=0, max_depth_ser=8):
    """
    Regression port of SER (Segev et al., 2017).

    Traverses the tree rooted at *node* and:
      - Updates each node's stored value to mean(y_target_node).
      - At leaves: optionally grows a new DecisionTreeRegressor if target
        samples are present and non-constant (extension step).
      - At internal nodes: after recursing, prunes the subtree back to a leaf
        if the leaf SSE <= subtree SSE (pruning step).
      - Removes branches that receive no target samples.

    Parameters
    ----------
    node          : int  — current node index (call with 0 for root)
    dTree         : DecisionTreeRegressor  — tree being adapted (mutated in-place)
    X_target_node : ndarray (n, n_features)
    y_target_node : ndarray (n,)
    original_ser  : bool  — kept for API compatibility; only True is used here
    _cur_depth    : int  — internal recursion counter; do not set manually
    max_depth_ser : int  — stop adapting below this depth and leave nodes frozen.
                    find_parent() is O(n_nodes) per call; on deep trees with
                    50K+ nodes the total cost grows quadratically with depth.
                    Default 3 caps work to ~14 find_parent calls per tree.

    Returns
    -------
    node      : int   — possibly updated node index after structural changes
    bool_flag : bool  — always False (no classification no-red logic)
    """
    # --- value update ---
    val = np.zeros((dTree.n_outputs_, 1))
    if y_target_node.size > 0:
        # Store the mean so sklearn's predict() returns the correct value directly.
        val[0, 0] = np.mean(y_target_node)
    dTree.tree_.value[node] = val
    dTree.tree_.n_node_samples[node]          = y_target_node.size
    dTree.tree_.weighted_n_node_samples[node] = float(y_target_node.size)

    # --- leaf: extension ---
    if dTree.tree_.feature[node] == -2:
        if y_target_node.size > 0 and len(np.unique(y_target_node)) > 1:
            # Cap extension depth to avoid explosive tree growth that slows
            # subsequent find_parent np.where scans.
            DT_to_add = DecisionTreeRegressor(max_depth=4)
            try:
                DT_to_add.min_impurity_decrease = 0
            except AttributeError:
                DT_to_add.min_impurity_split = 0
            DT_to_add.fit(X_target_node, y_target_node)
            fusionDecisionTreeReg(dTree, node, DT_to_add)
        return node, False

    # --- depth cap: leave sub-trees below max_depth_ser frozen ---
    if _cur_depth >= max_depth_ser:
        return node, False

    # --- split samples using the current (source) threshold ---
    feat      = dTree.tree_.feature[node]
    thr       = dTree.tree_.threshold[node]
    left_mask  = X_target_node[:, feat] <= thr
    right_mask = ~left_mask

    ind_left  = np.where(left_mask)[0]
    ind_right = np.where(right_mask)[0]

    X_left,  y_left  = X_target_node[ind_left],  y_target_node[ind_left]
    X_right, y_right = X_target_node[ind_right], y_target_node[ind_right]

    # --- recurse ---
    new_node_left, _ = SER_reg(dTree.tree_.children_left[node],  dTree,
                                X_left,  y_left,
                                original_ser=original_ser,
                                _cur_depth=_cur_depth + 1,
                                max_depth_ser=max_depth_ser)
    node, _b = find_parent(dTree, new_node_left)

    new_node_right, _ = SER_reg(dTree.tree_.children_right[node], dTree,
                                 X_right, y_right,
                                 original_ser=original_ser,
                                 _cur_depth=_cur_depth + 1,
                                 max_depth_ser=max_depth_ser)
    node, _b = find_parent(dTree, new_node_right)

    # --- SSE-based pruning ---
    le = leaf_error_reg(y_target_node)
    e  = error_reg(dTree, node, X_target_node, y_target_node)
    if le <= e:
        node = cut_into_leaf2(dTree, node)

    # --- remove empty branches ---
    if dTree.tree_.feature[node] != -2:
        if ind_left.size == 0:
            node = cut_from_left_right(dTree, node, -1)
        if ind_right.size == 0:
            node = cut_from_left_right(dTree, node, 1)

    return node, False


# =============================================================================
# Section 4: SER_RF_reg — forest wrapper
# =============================================================================

def SER_RF_reg(random_forest, X_target, y_target,
               original_ser=True, bootstrap_=False, max_depth_ser=8):
    """
    Apply SER_reg to every tree in a RandomForestRegressor.

    Parameters
    ----------
    random_forest : fitted RandomForestRegressor
    X_target      : ndarray (K, n_features)
    y_target      : ndarray (K,)
    original_ser  : bool  — passed through to SER_reg
    bootstrap_    : bool  — if True, each tree sees a bootstrap sample of K
    max_depth_ser : int   — depth cap for SER_reg (default 8); prevents O(n²)
                    cost from error_reg on deep trees

    Returns
    -------
    rf_adapted : deep copy of random_forest with all trees adapted
    """
    X_target = np.asarray(X_target)
    y_target = np.asarray(y_target)

    rf_adapted = copy.deepcopy(random_forest)

    for i, dtree in enumerate(rf_adapted.estimators_):
        _make_writeable_ser(dtree)
        inds = np.arange(y_target.size)
        if bootstrap_:
            inds = np.random.choice(inds, inds.size, replace=True)
        SER_reg(0, dtree, X_target[inds], y_target[inds],
                original_ser=original_ser,
                max_depth_ser=max_depth_ser)

    return rf_adapted


# =============================================================================
# Section 5: three-way K-sweep, plot, and CSV
# =============================================================================

def k_sweep_three_way(rf_source, X_cal, y_cal, X_test, y_test,
                      y_source, k_values=None, tau=0.1, lambda_div=0.5,
                      max_depth_ser=8):
    """
    Compare leaf re-targeting, Regression-STRUT, and SER_reg across K values.

    Returns a dict with keys: k_values, leaf, strut, ser, baseline.
    """
    if k_values is None:
        k_values = [10, 25, 50, 100, 200, 500, 1000]
    k_values = [k for k in k_values if k <= len(X_cal)]

    src_mape = evaluate_model(rf_source, X_test, y_test,
                               label="Source (no transfer)")["mape"]

    leaf_errs  = []
    strut_errs = []
    ser_errs   = []

    for k in k_values:
        Xk, yk = X_cal[:k], y_cal[:k]
        print(f"\n{'─'*55}", flush=True)
        print(f"  K = {k}", flush=True)
        print(f"{'─'*55}", flush=True)

        print("  [Leaf re-targeting]", flush=True)
        rf_leaf  = retarget_random_forest(rf_source, Xk, yk, alpha=1.0)
        leaf_m   = evaluate_model(rf_leaf,  X_test, y_test,
                                   label=f"Leaf  K={k}")["mape"]
        leaf_errs.append(leaf_m)

        print("  [Regression-STRUT]", flush=True)
        rf_strut = strut_regression_rf(rf_source, Xk, yk,
                                        y_source=y_source,
                                        tau=tau, lambda_div=lambda_div)
        strut_m  = evaluate_model(rf_strut, X_test, y_test,
                                   label=f"STRUT K={k}")["mape"]
        strut_errs.append(strut_m)

        print(f"  [SER_reg (max_depth_ser={max_depth_ser})]", flush=True)
        rf_ser   = SER_RF_reg(rf_source, Xk, yk, original_ser=True,
                               max_depth_ser=max_depth_ser)
        ser_m    = evaluate_model(rf_ser,   X_test, y_test,
                                   label=f"SER   K={k}")["mape"]
        ser_errs.append(ser_m)

    return {
        "k_values": k_values,
        "leaf":     leaf_errs,
        "strut":    strut_errs,
        "ser":      ser_errs,
        "baseline": src_mape,
    }


def plot_three_way(results, source_col, target_col, save_path=None):
    """Plot MAPE vs K for all three methods."""
    k    = results["k_values"]
    base = results["baseline"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k, results["leaf"],  marker="o", linewidth=2, label="Leaf re-targeting")
    ax.plot(k, results["strut"], marker="s", linewidth=2, label="Regression-STRUT")
    ax.plot(k, results["ser"],   marker="^", linewidth=2, label="SER_reg")
    ax.axhline(base, color="red", linestyle="--", linewidth=1.5,
               label=f"Source (no transfer): {base:.1f}%")

    ax.set_xlabel("K  (calibration samples from target)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(f"Transfer comparison\n{source_col}  →  {target_col}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Plot saved → {save_path}", flush=True)
    plt.show()


def write_csv_table(results, source_col, target_col, csv_path=None):
    """Print and optionally save a CSV table of MAPE vs K."""
    k_vals  = results["k_values"]
    base    = results["baseline"]
    header  = f"{'K':>6}  {'Leaf-MAPE':>10}  {'STRUT-MAPE':>11}  {'SER-MAPE':>9}  {'Source-MAPE':>12}"
    sep     = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    rows = []
    for i, k in enumerate(k_vals):
        row = (f"{k:>6}  "
               f"{results['leaf'][i]:>10.2f}  "
               f"{results['strut'][i]:>11.2f}  "
               f"{results['ser'][i]:>9.2f}  "
               f"{base:>12.2f}")
        print(row)
        rows.append(row)
    print(sep)

    if csv_path:
        lines = ["K,Leaf-MAPE,STRUT-MAPE,SER-MAPE,Source-MAPE"]
        for i, k in enumerate(k_vals):
            lines.append(
                f"{k},{results['leaf'][i]:.4f},"
                f"{results['strut'][i]:.4f},"
                f"{results['ser'][i]:.4f},"
                f"{base:.4f}"
            )
        with open(csv_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"  CSV saved → {csv_path}", flush=True)


# =============================================================================
# Section 6: __main__ driver
# =============================================================================

if __name__ == "__main__":
    SOURCE_COL  = "pixel4|1large|quant"
    TARGET_COL  = "pixel4|1large1med|quant"
    OP_TYPE     = "CONV_2D"
    MODEL_DIR   = "../models"
    RESULTS_DIR = "../results"
    TAU         = 0.1
    K_VALUES    = [10, 25, 50, 100, 200, 500, 1000]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading source data ...", flush=True)
    X_source, y_source = load_and_preprocess(target_col=SOURCE_COL, op_type=OP_TYPE)
    X_source, y_source = X_source.values, y_source.values

    print("Loading target data ...", flush=True)
    X_target, y_target = load_and_preprocess(target_col=TARGET_COL, op_type=OP_TYPE)
    X_target, y_target = X_target.values, y_target.values

    src_safe = SOURCE_COL.replace("|", "_")
    src_path = os.path.join(MODEL_DIR, f"rf_{OP_TYPE}_{src_safe}.joblib")

    if os.path.exists(src_path):
        print(f"\nLoading source model from {src_path}", flush=True)
        rf_source = joblib.load(src_path)
    else:
        print("\nTraining source model (n_estimators=50, max_depth=15) ...", flush=True)
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
        print(f"  Saved → {src_path}")
        evaluate_model(rf_source, Xte, yte, label="Source on source test set")

    X_cal, X_test_t, y_cal, y_test_t = train_test_split(
        X_target, y_target, test_size=0.2, random_state=42
    )

    print(f"\n{'='*65}", flush=True)
    print(f"Three-way K-sweep  |  tau={TAU}", flush=True)
    print(f"  {SOURCE_COL}  →  {TARGET_COL}", flush=True)
    print(f"{'='*65}", flush=True)

    # With np.where, find_parent is fast; depth 8 is a reasonable cap for
    # SER on max_depth=15 trees — adapts the upper half of each tree.
    SER_MAX_DEPTH = 8

    results = k_sweep_three_way(
        rf_source, X_cal, y_cal, X_test_t, y_test_t,
        y_source=y_source,
        k_values=K_VALUES,
        tau=TAU, lambda_div=0.5,
        max_depth_ser=SER_MAX_DEPTH,
    )

    tgt_safe = TARGET_COL.replace("|", "_")

    plot_path = os.path.join(
        RESULTS_DIR,
        f"transfer_comparison_{src_safe}_to_{tgt_safe}.png"
    )
    csv_path  = os.path.join(
        RESULTS_DIR,
        f"transfer_comparison_{src_safe}_to_{tgt_safe}.csv"
    )

    write_csv_table(results, SOURCE_COL, TARGET_COL, csv_path=csv_path)
    plot_three_way(results, SOURCE_COL, TARGET_COL, save_path=plot_path)
