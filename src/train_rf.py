from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

from data_utils import load_and_preprocess
from evaluate import mape

import time

X, y = load_and_preprocess()


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

param_grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 50, 100],
    "max_features": [0.5, 0.6, 0.75]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

grid = GridSearchCV(
    rf,
    param_grid,
    scoring="neg_mean_absolute_percentage_error",
    cv=3,
    n_jobs=-1
)

start = time.time()
grid.fit(X_train, y_train)
elapsed = time.time() - start

best_model = grid.best_estimator_

val_pred = best_model.predict(X_val)
test_pred = best_model.predict(X_test)

print("Best hyperparameters:", grid.best_params_)
print(f"Validation mean absolute percentage error: {mape(y_val, val_pred):.2f}%")
print(f"Test mean absolute percentage error: {mape(y_test, test_pred):.2f}%")

print(f"Training time: {elapsed:.2f} seconds")
