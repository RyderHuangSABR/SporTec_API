import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ==========================================
# SCORING CONFIGURATION (PLUS METRICS)
# ==========================================
# Normalization constants to scale raw probabilities into 
# standard baseball "Plus" metrics (where 100 is league average).

BASELINE_WHIFF_RATE = 0.30
BASELINE_CONTACT_RATE = 0.50
PLUS_SCALE_MULTIPLIER = 100.0
MIN_PLUS_SCORE = 20.0
MAX_PLUS_SCORE = 200.0

def score_pitch(df: pd.DataFrame, models: Dict[str, xgb.Booster], features: List[str]) -> Tuple[float, float]:
    """
    Evaluates a pitch using the dual-engine XGBoost models to generate 
    normalized metrics (Whiff+ and Contact+).
    """
    if not models or "A" not in models or "B" not in models:
        logger.warning("Models missing or incomplete. Defaulting to baseline scores (100.0).")
        return 100.0, 100.0
        
    X = df[features].copy()
    
    # Safely impute missing numerical data without breaking categorical columns
    for col in X.columns:
        if X[col].dtype.name != 'category':
            X[col] = X[col].fillna(0.0)
            
    try:
        dmatrix = xgb.DMatrix(X, enable_categorical=True)
        
        # Model A: Whiff Engine | Model B: Contact Damage Engine
        raw_whiff = models["A"].predict(dmatrix)[0]
        raw_contact = models["B"].predict(dmatrix)[0]
        
        # Normalize to the "Plus" scale
        whiff_plus = np.clip((raw_whiff / BASELINE_WHIFF_RATE) * PLUS_SCALE_MULTIPLIER, MIN_PLUS_SCORE, MAX_PLUS_SCORE)
        contact_plus = np.clip((raw_contact / BASELINE_CONTACT_RATE) * PLUS_SCALE_MULTIPLIER, MIN_PLUS_SCORE, MAX_PLUS_SCORE)
        
        return float(whiff_plus), float(contact_plus)
        
    except Exception as e:
        logger.error(f"Scoring engine fault: {e}. Defaulting to baseline scores (100.0).")
        return 100.0, 100.0
