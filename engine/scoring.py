import xgboost as xgb
import numpy as np

def score_pitch(df, models, features):
    if not models or "A" not in models or "B" not in models:
        return 100.0, 100.0
    X = df[features].copy()
    for col in X.columns:
        if X[col].dtype.name != 'category':
            X[col] = X[col].fillna(0.0)
    try:
        dmatrix = xgb.DMatrix(X, enable_categorical=True)
        raw_q = models["A"].predict(dmatrix)[0]
        raw_c = models["B"].predict(dmatrix)[0]
        q_plus = np.clip((raw_q / 0.30) * 100, 20.0, 200.0)
        c_plus = np.clip((raw_c / 0.50) * 100, 20.0, 200.0)
        return float(q_plus), float(c_plus)
    except Exception:
        return 100.0, 100.0
