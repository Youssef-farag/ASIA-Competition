"""Per-target regression model (random forest, extra trees, ridge, XGBoost, LightGBM)."""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from isncsci._optional import HAS_LGB, HAS_XGB, lgb, xgb


class SimpleModel:
    """Per-target regression model with optional per-target hyperparameters"""
    
    def __init__(self, model_type='rf', custom_params=None, per_target_params=None):
        self.model_type = model_type.lower()
        self.custom_params = custom_params or {}
        self.per_target_params = per_target_params or {}  # Dict of {target: {params: {...}}}
        self.models = {}
        self.target_columns = None
        
        # Validate model type
        if self.model_type in ['xgb', 'xgboost'] and not HAS_XGB:
            raise ImportError("XGBoost not available")
        if self.model_type in ['lgb', 'lightgbm', 'lgbm'] and not HAS_LGB:
            raise ImportError("LightGBM not available")
        
        # Print info about loaded params
        if self.per_target_params:
            valid_params = sum(1 for v in self.per_target_params.values() if v.get('cv_rmse') is not None)
            print(f"  Loaded per-target parameters for {valid_params} targets")
    
    def _get_model(self, target=None):
        """Get model instance based on model_type, optionally using target-specific params"""
        # Start with custom params or empty dict
        params = self.custom_params.copy() if self.custom_params else {}
        
        # Override with per-target params if available
        if target and target in self.per_target_params:
            target_info = self.per_target_params[target]
            if 'params' in target_info and target_info.get('cv_rmse') is not None:
                params.update(target_info['params'])
        
        if self.model_type in ['xgb', 'xgboost']:
            defaults = dict(n_estimators=200, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=1.0,
                            random_state=42, n_jobs=-1, tree_method='hist')
            defaults.update(params)
            return xgb.XGBRegressor(**defaults)
        elif self.model_type in ['lgb', 'lightgbm', 'lgbm']:
            defaults = dict(n_estimators=200, max_depth=6, learning_rate=0.05,
                            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=1.0,
                            random_state=42, n_jobs=-1, verbose=-1)
            defaults.update(params)
            return lgb.LGBMRegressor(**defaults)
        elif self.model_type == 'ridge':
            defaults = dict(alpha=1.0, random_state=42)
            defaults.update(params)
            return Ridge(**defaults)
        elif self.model_type in ['et', 'extratrees']:
            defaults = dict(n_estimators=200, max_depth=10,
                            min_samples_split=10, min_samples_leaf=4,
                            max_features='sqrt',
                            random_state=42, n_jobs=-1)
            defaults.update(params)
            return ExtraTreesRegressor(**defaults)
        else:  # Default RF
            defaults = dict(n_estimators=200, max_depth=10,
                            min_samples_split=10, min_samples_leaf=4,
                            max_features='sqrt',
                            random_state=42, n_jobs=-1)
            defaults.update(params)
            return RandomForestRegressor(**defaults)
    
    def train(self, X_train, y_train):
        print(f"\nTraining {self.model_type.upper()}...")
        
        self.target_columns = [c for c in y_train.columns if c != 'ID']
        feature_cols = [c for c in X_train.columns if c != 'ID']
        X = X_train[feature_cols].values
        
        for idx, target in enumerate(self.target_columns):
            y = y_train[target].values
            mask = ~np.isnan(y)
            
            if mask.sum() < 10:
                continue
            
            X_sel = X[mask]
            model = self._get_model(target=target)  # Pass target for per-target params
            
            if self.model_type in ['xgb', 'xgboost']:
                model.fit(X_sel, y[mask], verbose=False)
            else:
                model.fit(X_sel, y[mask])
            
            self.models[target] = model
            
            if (idx + 1) % 30 == 0:
                print(f"  Trained {idx + 1}/{len(self.target_columns)} targets...")
        
        print(f"Trained {len(self.models)} models")
    
    def predict(self, X_test):
        feature_cols = [c for c in X_test.columns if c != 'ID']
        X = X_test[feature_cols].values
        
        predictions = {'ID': X_test['ID'].values}
        for target in self.target_columns:
            if target in self.models:
                predictions[target] = self.models[target].predict(X)
            else:
                predictions[target] = np.zeros(len(X))
        
        return pd.DataFrame(predictions)

    def fit_target(self, target, X_train, y_train):
        """Fit the model for one target (used in cross-validation); returns a predict function."""
        model = self._get_model(target=target)
        if self.model_type in ['xgb', 'xgboost']:
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)
        return model.predict
