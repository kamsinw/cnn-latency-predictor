from sklearn.model_selection import GridSearchCV


def sensitivity_analysis(model_class, param_name, param_values, fixed_params, X_train, y_train, cv=4):
    """Tune one hyperparameter at a time using GridSearchCV, keeping all others fixed.

    Returns a list of (param_value, mean_cv_mape) tuples for every
    candidate value, so callers can inspect the full error landscape.
    """
    model = model_class(**fixed_params)

    grid = GridSearchCV(
        model,
        param_grid={param_name: param_values},
        scoring="neg_mean_absolute_percentage_error",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    results = []
    for params, mean_score in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"]):
        results.append((params[param_name], -mean_score * 100))

    return results
