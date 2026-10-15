"""Optional gradient-boosting backends (XGBoost, LightGBM)."""

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    xgb = None
    HAS_XGB = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    lgb = None
    HAS_LGB = False
    print("Warning: LightGBM not available")
