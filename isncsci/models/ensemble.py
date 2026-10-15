"""Ensemble of per-target models, each with its own feature set."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from isncsci.features.observation import add_observation_mask_features
from isncsci.metrics import compute_ordinal_metrics, print_ordinal_metrics
from isncsci.models.ordinal import OrdinalClassifierModel
from isncsci.models.regression import SimpleModel


def parse_model_spec(spec, default_features):
    """'rf:large' -> ('rf', 'large'); a plain 'rf' uses `default_features`."""
    if ':' in spec:
        parts = spec.split(':')
        return parts[0], parts[1]
    return spec, default_features


class EnsembleModel:
    """
    Average of per-target models. Every member has its own feature set, so this class covers

      - a single model                   ["rf"]
      - an ensemble on one feature set   ["rf", "xgb", "et", "ridge"]  (plain names use default_features)
      - a hybrid ensemble                ["rf:large", "xgb:medium", "ridge:baseline"]

    Options (all off by default):
      use_ordinal        members are cumulative ordinal classifiers instead of regressors
      ordinal_xgb_tuned  apply tuned XGBoost params to the ordinal classifiers
      obs_mask_features  append obs_<col> indicator columns to every feature matrix
      two_pass           second prediction pass with the first-pass predictions substituted for the
                         label columns of the feature matrix. The models are trained once, on the
                         un-enriched features, and whole columns are replaced, including observed cells.
                         In cross-validation only the current target's column is substituted.
    """

    def __init__(self, model_specs, get_engineer, default_features, per_target_params_list=None,
                 use_ordinal=False, ordinal_xgb_tuned=False, obs_mask_features=False, two_pass=False):
        """
        Args:
            model_specs: "model" or "model:features" strings
            get_engineer: callable mapping a feature-set name to a fresh feature engineer
            default_features: feature set for specs given without ":features"
            per_target_params_list: one per-target params dict (or None) per model, in order
        """
        self.use_ordinal = use_ordinal
        self.obs_mask_features = obs_mask_features
        self.two_pass = two_pass
        self.members = []
        self.engineers = {}      # one feature engineer per feature set
        self.target_columns = None
        self.overlapping = []    # label columns that are also raw feature columns (two-pass)
        self.X_trains = {}

        for idx, spec in enumerate(model_specs):
            model_type, features = parse_model_spec(spec, default_features)
            if features not in self.engineers:
                self.engineers[features] = get_engineer(features)

            params = per_target_params_list[idx] if per_target_params_list and idx < len(per_target_params_list) else None
            if use_ordinal:
                model = OrdinalClassifierModel(model_type=model_type, per_target_params=params,
                                               xgb_tuned=ordinal_xgb_tuned)
            else:
                model = SimpleModel(model_type=model_type, per_target_params=params)
            self.members.append({'model_type': model_type, 'features': features, 'model': model})

        print(f"Created ensemble{' (ORDINAL)' if use_ordinal else ''} with {len(self.members)} model(s):")
        for m in self.members:
            print(f"  - {m['model_type'].upper()}: {m['features']} features")

    def _describe(self):
        return ', '.join(f"{m['model_type'].upper()}:{m['features']}" for m in self.members)

    def _set_targets(self, features_train, y_train):
        self.target_columns = [c for c in y_train.columns if c != 'ID']
        self.overlapping = [c for c in self.target_columns if c in features_train.columns]

    def _engineer(self, features_df, metadata_df, is_training):
        """Engineer every feature set (plus optional obs-mask columns) for one split."""
        matrices = {}
        for features_type, engineer in self.engineers.items():
            if is_training:
                print(f"  Engineering {features_type} features...")
            X = engineer.engineer_features(features_df, metadata_df, is_training=is_training)
            if self.obs_mask_features:
                X = add_observation_mask_features(X, features_df, self.target_columns)
            if is_training:
                print(f"    {features_type}: {X.shape[1] - 1} features")
            matrices[features_type] = X
        return matrices

    def _enrich(self, X_df, predictions_df):
        """Two-pass: replace the label columns of a feature matrix by first-pass predictions (clipped to [0, 2])."""
        X_enriched = X_df.copy()
        pred_indexed = predictions_df.set_index('ID')
        for col in self.overlapping:
            if col in X_enriched.columns and col in pred_indexed.columns:
                vals = pred_indexed[col].reindex(X_enriched['ID'].values).values
                X_enriched[col] = np.clip(vals, 0, 2)
        return X_enriched

    def _average_predictions(self, X_by_features):
        all_predictions = []
        for m in self.members:
            print(f"  Getting predictions from {m['model_type'].upper()} ({m['features']})...")
            all_predictions.append(m['model'].predict(X_by_features[m['features']]))
        averaged = {'ID': next(iter(X_by_features.values()))['ID'].values}
        for target in self.target_columns:
            averaged[target] = np.mean([p[target].values for p in all_predictions], axis=0)
        print(f"Averaged predictions from {len(self.members)} model(s)")
        return pd.DataFrame(averaged)

    def train(self, features_train, metadata_train, y_train):
        print(f"\n{'=' * 60}\nENSEMBLE TRAINING ({len(self.members)} models)\n{'=' * 60}")
        self._set_targets(features_train, y_train)
        print("\nPreparing feature sets...")
        self.X_trains = self._engineer(features_train, metadata_train, is_training=True)
        for idx, m in enumerate(self.members):
            print(f"\nTraining model {idx + 1}/{len(self.members)}: {m['model_type'].upper()} ({m['features']} features)")
            m['model'].train(self.X_trains[m['features']], y_train)
        print(f"\n{'=' * 60}\nEnsemble training complete!\n{'=' * 60}")

    def predict(self, features_test, metadata_test):
        print("\nGenerating ensemble predictions...")
        X_tests = self._engineer(features_test, metadata_test, is_training=False)
        predictions = self._average_predictions(X_tests)
        if self.two_pass:
            print("\nTwo-pass: re-predicting with first-pass predictions substituted for the label columns...")
            X_enriched = {ftype: self._enrich(X, predictions) for ftype, X in X_tests.items()}
            predictions = self._average_predictions(X_enriched)
        return predictions

    def cross_validate(self, features_train, metadata_train, y_train, cv=5):
        """Per-target k-fold CV of the averaged ensemble; returns the mean per-target RMSE (Pass 2 if two_pass)."""
        print(f"\n{'=' * 60}")
        print(f"{'TWO-PASS ' if self.two_pass else ''}ENSEMBLE CV ({len(self.members)} models, {cv} folds)")
        print(f"  {self._describe()}")
        print('=' * 60)

        self._set_targets(features_train, y_train)
        print("\nPreparing feature sets...")
        self.X_trains = self._engineer(features_train, metadata_train, is_training=True)
        views = []  # per member: (feature column names, feature matrix)
        for m in self.members:
            X_df = self.X_trains[m['features']]
            feature_cols = [c for c in X_df.columns if c != 'ID']
            views.append((feature_cols, X_df[feature_cols].values))
        ids = next(iter(self.X_trains.values()))['ID'].values

        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        rmses1, rmses2 = [], []
        all_true, all_pred1, all_pred2 = [], [], []

        for idx, target in enumerate(self.target_columns):
            y = y_train[target].values
            mask = ~np.isnan(y)
            if mask.sum() < 20:
                continue
            mask_idx = np.where(mask)[0]

            fold1, fold2 = [], []
            for train_idx, val_idx in kfold.split(np.arange(mask.sum())):
                tr, val = mask_idx[train_idx], mask_idx[val_idx]
                predictors = [m['model'].fit_target(target, X[tr], y[tr])
                              for m, (_, X) in zip(self.members, views)]

                p1 = np.mean([f(X[val]) for f, (_, X) in zip(predictors, views)], axis=0)
                fold1.append(np.sqrt(mean_squared_error(y[val], p1)))
                all_true.extend(y[val].tolist())
                all_pred1.extend(p1.tolist())

                if self.two_pass:
                    p1_df = pd.DataFrame({'ID': ids[val], target: p1})
                    X_enriched = []
                    for feature_cols, X in views:
                        X_val_df = pd.DataFrame(X[val], columns=feature_cols)
                        X_val_df['ID'] = ids[val]
                        X_enriched.append(self._enrich(X_val_df, p1_df)[feature_cols].values)
                    p2 = np.mean([f(Xe) for f, Xe in zip(predictors, X_enriched)], axis=0)
                    fold2.append(np.sqrt(mean_squared_error(y[val], p2)))
                    all_pred2.extend(p2.tolist())

            rmses1.append(np.mean(fold1))
            if self.two_pass:
                rmses2.append(np.mean(fold2))
            if (idx + 1) % 30 == 0:
                if self.two_pass:
                    print(f"  CV {idx + 1}/{len(self.target_columns)}: "
                          f"Pass1={np.mean(rmses1):.4f}  Pass2={np.mean(rmses2):.4f}")
                else:
                    print(f"  CV on {idx + 1}/{len(self.target_columns)} targets... Avg RMSE: {np.mean(rmses1):.4f}")

        final_rmse = np.mean(rmses2 if self.two_pass else rmses1)
        print(f"\n{'=' * 60}")
        print(f"ENSEMBLE CV RESULTS ({self._describe()}):")
        print_ordinal_metrics(compute_ordinal_metrics(all_true, all_pred1), label='Pass1' if self.two_pass else '')
        if self.two_pass:
            print_ordinal_metrics(compute_ordinal_metrics(all_true, all_pred2), label='Pass2')
        print('=' * 60)
        return final_rmse
