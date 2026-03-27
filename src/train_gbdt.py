from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

from data_utils import load_and_preprocess
from evaluate import mape
from tune_utils import sensitivity_analysis

import time

X, y = load_and_preprocess()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gbdt_fixed_params = {"random_state": 42, "n_jobs": -1, "verbosity": -1}

gbdt_hyperparams = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, -1],#look for deeper depthsm since GBDT can overfit less than RF(50 or 100)
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 63, 255],
    #add num_iterations to a higher valiue, "learing rate" %error doesnt make sense
}

print("=" * 60)
print("LightGBM GBDT — Hyperparameter Sensitivity Analysis")
print("=" * 60)

start = time.time()

for param_name, param_values in gbdt_hyperparams.items():
    results = sensitivity_analysis(
        LGBMRegressor,
        param_name,
        param_values,
        gbdt_fixed_params,
        X_train,
        y_train,
        cv=4,
    )

    print(f"\n=== {param_name} sensitivity ===")
    best_val, best_err = None, float("inf")
    worst_err = 0.0
    for val, err in results:
        print(f"  {str(val):>6s} -> MAPE: {err:.2f}%")
        if err < best_err:
            best_val, best_err = val, err
        worst_err = max(worst_err, err)

    spread = worst_err - best_err
    if spread < 1.0:
        sensitivity = "low"
    elif spread < 5.0:
        sensitivity = "medium"
    else:
        sensitivity = "high"
    print(f"  Best: {best_val} ({best_err:.2f}%)  |  Range: {best_err:.2f}% - {worst_err:.2f}%  |  Sensitivity: {sensitivity}")

sensitivity_time = time.time() - start

print("\n" + "=" * 60)
print("LightGBM GBDT — Combined GridSearchCV")
print("=" * 60)

gbdt = LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)

grid = GridSearchCV(
    gbdt,
    param_grid=gbdt_hyperparams,
    scoring="neg_mean_absolute_percentage_error",
    cv=4,
    n_jobs=-1,
)

start = time.time()
grid.fit(X_train, y_train)
grid_time = time.time() - start

best_model = grid.best_estimator_
test_pred = best_model.predict(X_test)

print(f"\nBest hyperparameters: {grid.best_params_}")
print(f"Best CV MAPE: {-grid.best_score_ * 100:.2f}%")
print(f"Test MAPE: {mape(y_test, test_pred):.2f}%")
print(f"\nSensitivity analysis time: {sensitivity_time:.2f} seconds")
print(f"Combined GridSearchCV time: {grid_time:.2f} seconds")
