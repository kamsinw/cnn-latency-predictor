import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

from data_utils import load_and_preprocess, TARGET_COL
from evaluate import mape
from tune_utils import sensitivity_analysis

import time

OP_TYPE = "CONV_2D"
X, y = load_and_preprocess(op_type=OP_TYPE)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_fixed_params = {"random_state": 42, "n_jobs": -1}

rf_hyperparams = {
    "n_estimators": [50,100, 300, 500],#no more then 40
    "max_depth": [10, 50, None],# no more then 800
    "max_features": [0.5, 0.75, 1.0],
}

print("=" * 60)
print("Random Forest — Hyperparameter Sensitivity Analysis")
print("=" * 60)

start = time.time()

for param_name, param_values in rf_hyperparams.items():
    results = sensitivity_analysis(
        RandomForestRegressor,
        param_name,
        param_values,
        rf_fixed_params,
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
print("Random Forest — Combined GridSearchCV")
print("=" * 60)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

grid = GridSearchCV(
    rf,
    param_grid=rf_hyperparams,
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

col_safe = TARGET_COL.replace("|", "_")
models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, f"rf_{OP_TYPE}_{col_safe}.joblib")
joblib.dump(best_model, model_path)
print(f"\nModel saved to {model_path}")
