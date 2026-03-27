import numpy as np
import matplotlib.pyplot as plt
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


import os

OUTPUT_DIR = "feature_importancesRF"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for target in PIXEL4_COLUMNS:
        print(f"Processing {target}...")

        X, y = load_and_preprocess(target_col=target)

        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        importances = rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

        indices = np.argsort(importances)
        sorted_features = [FEATURE_COLS[i] for i in indices]
        sorted_importances = importances[indices]
        sorted_std = std[indices]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(sorted_features, sorted_importances, xerr=sorted_std, align="center")
        ax.set_title(f"Feature importances using MDI\n{target}")
        ax.set_xlabel("Mean decrease in impurity")

        plt.tight_layout()

        filename = os.path.join(OUTPUT_DIR, f"{target.replace('|', '_')}_featureimportance.png")
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  Saved {filename}")

    print("\nAll done!")
