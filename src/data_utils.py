import ast
import pandas as pd

FEATURE_COLS = [
    "Input_shape",
    "Input_channel",
    "Output_shape",
    "Stride",
    "Kernel",
    "Filter",
    "Group",
    "FLOPS"
]

TARGET_COL = "pixel4|1large|float"


def load_and_preprocess(path="../data/tflite_cnn_synthetic_ops_cpu.csv", target_col=None):
    if target_col is None:
        target_col = TARGET_COL

    df = pd.read_csv(path)

    df = df[df["operation"] == "CONV_2D"]

    features = df["feature"].apply(ast.literal_eval).apply(pd.Series)

    X = features[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")

    data = pd.concat([X, y.rename(target_col)], axis=1).dropna()

    return data[FEATURE_COLS], data[target_col]