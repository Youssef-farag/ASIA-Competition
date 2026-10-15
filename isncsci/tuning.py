"""Per-target hyperparameter search (RandomizedSearchCV, one search per target)."""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss, make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm

from isncsci._optional import HAS_LGB, HAS_XGB, lgb, xgb
from isncsci.models.ordinal import OrdinalClassifierModel

# Parameter grids for each model
PARAM_GRIDS = {
    'rf': {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [6, 8, 10, 12, 15, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
    },
    'xgb': {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0]
    },
    'lgb': {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [3, 4, 5, 6, 7, 8, -1],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'num_leaves': [15, 31, 63, 127],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0]
    },
    'et': {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [6, 8, 10, 12, 15, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
    },
    'ridge': {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    }
}


def get_base_model(model_type):
    """Get base model instance"""
    if model_type in ['xgb', 'xgboost']:
        if not HAS_XGB:
            raise ImportError("XGBoost not available")
        return xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist')
    elif model_type in ['lgb', 'lightgbm', 'lgbm']:
        if not HAS_LGB:
            raise ImportError("LightGBM not available")
        return lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    elif model_type == 'ridge':
        return Ridge(random_state=42)
    elif model_type in ['et', 'extratrees']:
        return ExtraTreesRegressor(random_state=42, n_jobs=-1)
    else:  # rf
        return RandomForestRegressor(random_state=42, n_jobs=-1)


def get_base_classifier(model_type):
    """Base binary classifier for ordinal tuning: the same estimator OrdinalClassifierModel uses for `model_type`."""
    if model_type == 'ridge':
        raise ValueError("ridge has no ordinal classifier; tune rf, et or xgb with --use-ordinal")
    return OrdinalClassifierModel(model_type=model_type)._get_clf()


def tune_per_target(model_type, X_train, y_train, param_grid, n_iter=20, cv=3, output_file='tuned_params.json',
                    use_ordinal=False):
    """
    Tune hyperparameters individually for each target with incremental saving.

    Args:
        model_type: Model type string
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid for search
        n_iter: Number of random search iterations
        cv: Number of CV folds
        output_file: Path to save results (supports resume)
        use_ordinal: If True, tune the binary classifiers of the ordinal model (for use with
                     --use-ordinal in train.py). Scores by neg_log_loss on the P(y>0) threshold.
                     The stored cv_rmse field contains the log-loss (lower = better).

    Returns:
        dict: Per-target best parameters {target: {params: {...}, cv_rmse: float}}
    """
    mode_label = 'ORDINAL' if use_ordinal else 'REGRESSION'
    print(f"\n{'='*60}")
    print(f"PER-TARGET TUNING: {model_type.upper()} [{mode_label}]")
    print(f"Iterations: {n_iter}, CV: {cv}")
    print(f"{'='*60}")
    
    feature_cols = [c for c in X_train.columns if c != 'ID']
    X = X_train[feature_cols].values
    target_cols = [c for c in y_train.columns if c != 'ID']
    
    # Load existing results if file exists (for resume support)
    output_path = Path(output_file)
    if output_path.exists():
        with open(output_path, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {output_file}")
        print(f"Already tuned: {len(all_results)} targets")
    else:
        all_results = {}
    
    # Get base model / classifier
    if use_ordinal:
        base_model = get_base_classifier(model_type)
        # Score with neg_log_loss on the P(y>0) binary threshold (most informative
        # threshold for {0,1,2} targets where label 0 is the dominant class).
        scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    else:
        base_model = get_base_model(model_type)
        # Custom scorer (negative RMSE for minimization)
        def rmse_scorer(y_true, y_pred):
            return -np.sqrt(mean_squared_error(y_true, y_pred))
        scorer = make_scorer(rmse_scorer, greater_is_better=True)
    
    # Track progress
    total_targets = len(target_cols)
    tuned_count = len(all_results)
    skipped = 0
    
    # Tune each target with progress bar
    pbar = tqdm(target_cols, desc="Tuning targets", unit="target")
    
    for target in pbar:
        # Skip if already tuned
        if target in all_results:
            skipped += 1
            pbar.set_postfix({'skipped': skipped, 'avg_rmse': f"{np.mean([r['cv_rmse'] for r in all_results.values()]):.4f}"})
            continue
        
        y = y_train[target].values
        mask = ~np.isnan(y)
        
        if mask.sum() < 20:
            # Skip targets with insufficient data
            all_results[target] = {'params': {}, 'cv_rmse': None, 'note': 'insufficient_data'}
            skipped += 1
            continue
        
        X_t, y_t = X[mask], y[mask]

        if use_ordinal:
            # Binarise: tune on the P(y>0) threshold (0 vs {1,2}).
            # This is the most informative split given ~78% label-0 prevalence.
            y_t_fit = (y_t.astype(int) > 0).astype(int)
            # Skip if degenerate (only one class present)
            if y_t_fit.sum() == 0 or y_t_fit.sum() == len(y_t_fit):
                all_results[target] = {'params': {}, 'cv_rmse': None, 'note': 'degenerate_binary'}
                skipped += 1
                continue
        else:
            y_t_fit = y_t

        # Run RandomizedSearchCV for this target
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scorer,
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        random_search.fit(X_t, y_t_fit)

        # Store results
        # For ordinal: best_score_ is neg_log_loss (already negative), store as positive log-loss.
        # For regression: best_score_ is neg_RMSE, store as positive RMSE.
        cv_score = float(-random_search.best_score_)
        all_results[target] = {
            'params': random_search.best_params_,
            'cv_rmse': cv_score
        }
        
        tuned_count += 1
        
        # Update progress bar with current stats
        valid_scores = [r['cv_rmse'] for r in all_results.values() if r['cv_rmse'] is not None]
        avg_score = np.mean(valid_scores) if valid_scores else 0
        score_label = 'avg_logloss' if use_ordinal else 'avg_rmse'
        pbar.set_postfix({'tuned': tuned_count, score_label: f"{avg_score:.4f}"})
        
        # Save incrementally after each target
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    pbar.close()
    
    # Final summary
    valid_results = {k: v for k, v in all_results.items() if v['cv_rmse'] is not None}
    avg_score = np.mean([r['cv_rmse'] for r in valid_results.values()])
    score_name = 'Average CV log-loss (P(y>0) threshold)' if use_ordinal else 'Average CV RMSE'

    print(f"\n{'='*60}")
    print(f"PER-TARGET TUNING COMPLETE FOR {model_type.upper()} [{mode_label}]")
    print(f"Total targets: {total_targets}")
    print(f"Successfully tuned: {len(valid_results)}")
    print(f"Skipped (insufficient data / degenerate): {total_targets - len(valid_results)}")
    print(f"{score_name}: {avg_score:.4f}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    return all_results
