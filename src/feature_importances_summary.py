import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor

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

OUTPUT_FILE = os.path.join("feature_importancesRF", "feature_importance_summary.txt")


if __name__ == "__main__":
    os.makedirs("feature_importancesRF", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        f.write("Feature Importances (MDI) — All Pixel4 Columns\n")
        f.write("=" * 60 + "\n\n")

        for target in PIXEL4_COLUMNS:
            print(f"Processing {target}...")

            X, y = load_and_preprocess(target_col=target)

            rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            importances = rf.feature_importances_
            std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

            # Sort descending so most important is first
            indices = np.argsort(importances)[::-1]

            f.write(f"[ {target} ]\n")
            f.write("-" * 40 + "\n")
            for i in indices:
                f.write(f"  {FEATURE_COLS[i]:<20s}  {importances[i]:.4f}  (+/- {std[i]:.4f})\n")
            f.write("\n")

    print(f"\nDone! Summary saved to {OUTPUT_FILE}")
