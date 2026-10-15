"""Ordinal classifier model: cumulative binary classifiers for {0, 1, 2} targets."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from isncsci._optional import HAS_LGB, HAS_XGB, lgb, xgb


class OrdinalClassifierModel:
    """
    Ordinal regression via cumulative binary thresholding.

    For a 3-class ordinal target {0, 1, 2} trains two binary classifiers:
      clf0: P(y > 0)  -- separates 0 from {1,2}
      clf1: P(y > 1)  -- separates {0,1} from 2

    Prediction: E[y] = P(y>0) + min(P(y>1), P(y>0)), in [0,2], directly evaluated by RMSE.

    Hyperparameters are the fixed defaults in _get_clf. Per-target params loaded from regression
    tuning (--load-params) are applied to the classifiers as follows:
      rf, et, lgb   tuned params are applied (keys the classifier does not accept are ignored)
      xgb           fixed defaults; tuned params are applied only with xgb_tuned=True
                    (--ordinal-xgb-tuned)
      ridge         fixed defaults (ridge has no classifier equivalent; an RF classifier is used)
    """

    def __init__(self, model_type='lgb', custom_params=None, per_target_params=None, xgb_tuned=False):
        self.model_type = model_type.lower()
        self.custom_params = custom_params or {}
        self.per_target_params = per_target_params or {}
        self.xgb_tuned = xgb_tuned
        self.models = {}
        self.target_columns = None

        if self.model_type in ['lgb', 'lightgbm', 'lgbm'] and not HAS_LGB:
            print("Warning: LightGBM not available for ordinal model, falling back to RF")
            self.model_type = 'rf'
        if self.model_type in ['xgb', 'xgboost'] and not HAS_XGB:
            print("Warning: XGBoost not available for ordinal model, falling back to RF")
            self.model_type = 'rf'

        if self.per_target_params:
            valid = sum(1 for v in self.per_target_params.values() if v.get('cv_rmse') is not None)
            print(f"  Loaded per-target parameters for {valid} targets")

    def _get_clf(self, target=None):
        raw_params = self.custom_params.copy()
        if target and target in self.per_target_params:
            info = self.per_target_params[target]
            if 'params' in info and info.get('cv_rmse') is not None:
                raw_params.update(info['params'])

        def _accepted(cls, merged):
            # Keep only hyperparameters the estimator accepts. get_params() lists them for
            # sklearn, XGBoost and LightGBM alike (signature inspection misses XGBoost's).
            valid = set(cls().get_params())
            return {k: v for k, v in merged.items() if k in valid}

        if self.model_type in ['lgb', 'lightgbm', 'lgbm']:
            defaults = dict(n_estimators=200, max_depth=6, learning_rate=0.05,
                            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=1.0,
                            random_state=42, n_jobs=-1, verbose=-1)
            defaults.update(_accepted(lgb.LGBMClassifier, raw_params))
            return lgb.LGBMClassifier(**defaults)
        elif self.model_type in ['xgb', 'xgboost']:
            defaults = dict(n_estimators=200, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=1.0,
                            random_state=42, n_jobs=-1, tree_method='hist',
                            eval_metric='logloss', use_label_encoder=False)
            if self.xgb_tuned:
                defaults.update(_accepted(xgb.XGBClassifier, raw_params))
            return xgb.XGBClassifier(**defaults)
        else:  # rf / et / ridge all use the RF classifier
            defaults = dict(n_estimators=200, max_depth=10,
                            min_samples_split=10, min_samples_leaf=4,
                            max_features='sqrt', random_state=42, n_jobs=-1)
            if self.model_type != 'ridge':
                defaults.update(_accepted(RandomForestClassifier, raw_params))
            return RandomForestClassifier(**defaults)

    def _predict_expected(self, clf0, clf1, X):
        p_gt0 = clf0.predict_proba(X)[:, 1]
        p_gt1 = clf1.predict_proba(X)[:, 1]
        return p_gt0 + np.minimum(p_gt1, p_gt0)

    def train(self, X_train, y_train):
        print(f"\nTraining ORDINAL ({self.model_type.upper()})...")
        self.target_columns = [c for c in y_train.columns if c != 'ID']
        feature_cols = [c for c in X_train.columns if c != 'ID']
        X = X_train[feature_cols].values

        for idx, target in enumerate(self.target_columns):
            y = y_train[target].values
            mask = ~np.isnan(y)
            if mask.sum() < 10:
                continue
            X_sel = X[mask]
            y_int = y[mask].astype(int)
            y_gt0, y_gt1 = (y_int > 0).astype(int), (y_int > 1).astype(int)

            clf0 = self._get_clf(target) if y_gt0.sum() > 0 and y_gt0.sum() < len(y_gt0) else None
            clf1 = self._get_clf(target) if y_gt1.sum() > 0 and y_gt1.sum() < len(y_gt1) else None
            if clf0 is not None:
                clf0.fit(X_sel, y_gt0)
            if clf1 is not None:
                clf1.fit(X_sel, y_gt1)
            self.models[target] = (clf0, clf1, float(y_int.mean()))

            if (idx + 1) % 30 == 0:
                print(f"  Trained {idx + 1}/{len(self.target_columns)} targets...")
        print(f"Trained ordinal models for {len(self.models)} targets")

    def predict(self, X_test):
        feature_cols = [c for c in X_test.columns if c != 'ID']
        X = X_test[feature_cols].values
        predictions = {'ID': X_test['ID'].values}
        for target in self.target_columns:
            if target not in self.models:
                predictions[target] = np.zeros(len(X))
                continue
            clf0, clf1, fallback = self.models[target]
            if clf0 is None and clf1 is None:
                predictions[target] = np.full(len(X), fallback)
            elif clf0 is None:
                predictions[target] = 1.0 + clf1.predict_proba(X)[:, 1]
            elif clf1 is None:
                predictions[target] = clf0.predict_proba(X)[:, 1]
            else:
                predictions[target] = self._predict_expected(clf0, clf1, X)
        return pd.DataFrame(predictions)

    def _fit_threshold(self, target, X_train, y_binary):
        """Fit one threshold classifier (cross-validation); returns a function giving P(y > threshold)."""
        if 0 < y_binary.sum() < len(y_binary):
            clf = self._get_clf(target)
            clf.fit(X_train, y_binary)
            return lambda X: clf.predict_proba(X)[:, 1]
        prob = float(y_binary.mean())
        return lambda X: np.full(len(X), prob)

    def fit_target(self, target, X_train, y_train):
        """Fit both threshold classifiers for one target (used in cross-validation); returns a predict function."""
        y_int = y_train.astype(int)
        p_gt0 = self._fit_threshold(target, X_train, (y_int > 0).astype(int))
        p_gt1 = self._fit_threshold(target, X_train, (y_int > 1).astype(int))
        return lambda X: p_gt0(X) + np.minimum(p_gt1(X), p_gt0(X))
