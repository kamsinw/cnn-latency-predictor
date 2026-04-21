import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMRegressor

from data_utils import load_and_preprocess
from evaluate import mape
from tune_utils import sensitivity_analysis

import time

MODELS_DIR = "../models"

PIXEL4_COLUMNS = [
    "pixel4|1large|float",
    "pixel4|1med|float",
    "pixel4|2med|float",
    "pixel4|3med|float",
    "pixel4|1large1med|float",
    "pixel4|1small|float",
    "pixel4|2small|float",
    "pixel4|3small|float",
    "pixel4|4small|float",
    "pixel4|1large|quant",
    "pixel4|1med|quant",
    "pixel4|2med|quant",
    "pixel4|3med|quant",
    "pixel4|1large1med|quant",
    "pixel4|1small|quant",
    "pixel4|2small|quant",
    "pixel4|3small|quant",
    "pixel4|4small|quant",
]

OP_TYPES = ["CONV_2D", "FULLY_CONNECTED"]

RF_FIXED = {"random_state": 42, "n_jobs": -1}
RF_HYPERPARAMS = {
     "n_estimators": [50,100, 300, 500],#no more then 40
    "max_depth": [10, 50, None],# no more then 800
    "max_features": [0.5, 0.75, 1.0],
}

GBDT_FIXED = {"random_state": 42, "n_jobs": -1, "verbosity": -1}
GBDT_HYPERPARAMS = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, 50,100-1],#look for deeper depthsm since GBDT can overfit less than RF(50 or 100)
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 63, 255],
    #add num_iterations to a higher valiue, "learing rate" %error doesnt make sense
    "num_iterations": [100, 200, 400, 500],
}

MODELS = {
    "RF": (RandomForestRegressor, RF_FIXED, RF_HYPERPARAMS),
    "GBDT": (LGBMRegressor, GBDT_FIXED, GBDT_HYPERPARAMS),
}


def classify_sensitivity(spread):
    if spread < 1.0:
        return "low"
    elif spread < 5.0:
        return "medium"
    return "high"


def run_sensitivity(model_class, fixed_params, hyperparams, X_train, y_train):
    """Run per-hyperparameter sensitivity analysis using GridSearchCV.

    Returns a dict mapping param_name -> {best_val, best_err, worst_err, sensitivity, results}.
    """
    analysis = {}
    for param_name, param_values in hyperparams.items():
        results = sensitivity_analysis(
            model_class, param_name, param_values, fixed_params,
            X_train, y_train, cv=4,
        )

        best_val, best_err = min(results, key=lambda r: r[1])
        worst_err = max(r[1] for r in results)
        spread = worst_err - best_err

        analysis[param_name] = {
            "best_val": best_val,
            "best_err": best_err,
            "worst_err": worst_err,
            "sensitivity": classify_sensitivity(spread),
            "results": results,
        }
    return analysis


def print_sensitivity(analysis):
    for param_name, info in analysis.items():
        print(f"\n  === {param_name} sensitivity ===")
        for val, err in info["results"]:
            print(f"    {str(val):>6s} -> MAPE: {err:.2f}%")
        print(f"    Best: {info['best_val']} ({info['best_err']:.2f}%)"
              f"  |  Range: {info['best_err']:.2f}% - {info['worst_err']:.2f}%"
              f"  |  Sensitivity: {info['sensitivity']}")


def main():
    summary_rows = []

    for op_type in OP_TYPES:
        for col in PIXEL4_COLUMNS:
            print("\n" + "#" * 70)
            print(f"# Column: {col}  |  Op: {op_type}")
            print("#" * 70)

            try:
                X, y = load_and_preprocess(target_col=col, op_type=op_type)
            except Exception as e:
                print(f"  SKIPPED — {e}")
                continue

            if len(X) < 20:
                print(f"  SKIPPED — only {len(X)} samples")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            for model_name, (model_class, fixed_params, hyperparams) in MODELS.items():
                print(f"\n  [{model_name}]")

                t0 = time.time()
                analysis = run_sensitivity(
                    model_class, fixed_params, hyperparams,
                    X_train, y_train,
                )
                sensitivity_time = time.time() - t0

                print_sensitivity(analysis)

                grid = GridSearchCV(
                    model_class(**fixed_params),
                    param_grid=hyperparams,
                    scoring="neg_mean_absolute_percentage_error",
                    cv=4,
                    n_jobs=-1,
                )

                t0 = time.time()
                grid.fit(X_train, y_train)
                grid_time = time.time() - t0

                best_model = grid.best_estimator_
                test_pred = best_model.predict(X_test)
                test_mape = mape(y_test, test_pred)

                print(f"\n  Best hyperparameters: {grid.best_params_}")
                print(f"  Best CV MAPE: {-grid.best_score_ * 100:.2f}%")
                print(f"  Test MAPE: {test_mape:.2f}%")
                print(f"  Sensitivity time: {sensitivity_time:.2f}s  |  GridSearchCV time: {grid_time:.2f}s")

                os.makedirs(MODELS_DIR, exist_ok=True)
                col_safe = col.replace("|", "_")
                model_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_{op_type}_{col_safe}.joblib")
                joblib.dump(best_model, model_path)
                print(f"  Model saved -> {model_path}")

                most_sensitive = max(analysis, key=lambda p: analysis[p]["worst_err"] - analysis[p]["best_err"])

                summary_rows.append({
                    "column": col,
                    "op_type": op_type,
                    "model": model_name,
                    "test_mape": test_mape,
                    "best_params": grid.best_params_,
                    "grid_time": grid_time,
                    "most_sensitive_param": most_sensitive,
                })

    print("\n\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Column':<30s} {'Op':<20s} {'Model':<6s} {'Test MAPE':>10s} {'Time':>8s} {'Most Sensitive Param'}")
    print("-" * 90)
    for row in summary_rows:
        print(f"{row['column']:<30s} {row['op_type']:<20s} {row['model']:<6s} "
              f"{row['test_mape']:>9.2f}% {row['grid_time']:>7.1f}s {row['most_sensitive_param']}")


if __name__ == "__main__":
    output_path = "../results/experiments.txt"
    import os
    os.makedirs("../results", exist_ok=True)

    import builtins
    _real_print = builtins.print

    with open(output_path, "w") as f:
        def _file_print(*args, **kwargs):
            kwargs.pop("file", None)
            _real_print(*args, **kwargs, file=f, flush=True)

        builtins.print = _file_print
        try:
            main()
        finally:
            builtins.print = _real_print

    _real_print("Done! Results saved to", output_path)
