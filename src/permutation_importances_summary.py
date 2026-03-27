import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from data_utils import load_and_preprocess, FEATURE_COLS

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

OUTPUT_DIR = "permutation_features"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "permutation_importance_summary.txt")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        f.write("Permutation Feature Importances — All Pixel4 Columns\n")
        f.write("=" * 60 + "\n\n")

        for target in PIXEL4_COLUMNS:
            print(f"Processing {target}...")

            X, y = load_and_preprocess(target_col=target)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)

            result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

            indices = np.argsort(result.importances_mean)[::-1]

            f.write(f"[ {target} ]\n")
            f.write("-" * 40 + "\n")
            for i in indices:
                f.write(f"  {FEATURE_COLS[i]:<20s}  {result.importances_mean[i]:.4f}  (+/- {result.importances_std[i]:.4f})\n")
            f.write("\n")

    print(f"\nDone! Summary saved to {OUTPUT_FILE}")
