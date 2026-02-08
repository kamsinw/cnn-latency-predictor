import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data_utils import load_and_preprocess


X, y = load_and_preprocess()

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

x = pd.DataFrame([{
    "Input_shape": 224,
    "Input_channel": 3,
    "Output_shape": 112,
    "Stride": 2,
    "Kernel": 5,
    "Filter": 8,
    "Group": 1,
    "FLOPS": 15052800
}])

prediction = model.predict(x)
print(f"Predicted latency: {prediction[0]:.2f}")