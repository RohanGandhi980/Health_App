import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_dummy_model():
    train_X = pd.DataFrame({
        "temp": np.random.uniform(20, 35, 200),
        "humidity": np.random.uniform(40, 90, 200),
        "rainfall": np.random.uniform(0, 50, 200),
        "ph": np.random.uniform(5, 9, 200),
        "Turbidity": np.random.uniform(1, 10, 200),
        "Conductivity": np.random.uniform(100, 500, 200),
        "reported_cases": np.random.randint(0, 20, 200)
    })
    train_y = np.random.choice([0, 1], size=200, p=[0.7, 0.3])

    model = RandomForestClassifier()
    model.fit(train_X, train_y)
    joblib.dump(model, "model.pkl")
    return model
