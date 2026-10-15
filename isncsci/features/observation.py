"""Observation-mask features."""

import numpy as np


def add_observation_mask_features(X_engineered, raw_features_df, label_cols):
    """
    Append binary indicator columns obs_<col> to an already-engineered feature
    DataFrame, one per label column that appears in the raw feature file.

    obs_<col> = 1 where the raw feature was directly measured (non-NaN),
                0 where it was missing and later median-imputed.

    This lets the model distinguish 'c5ltl = 2 (observed)' from
    'c5ltl = 2 (median-imputed)' — currently indistinguishable.

    Args:
        X_engineered    : Output of engineer.engineer_features() — has 'ID' column.
        raw_features_df : The original features CSV (features_train/test_2.csv) — has 'ID'.
        label_cols      : List of target column names to create indicators for.

    Returns:
        Copy of X_engineered with extra obs_* columns appended (before 'ID').
    """
    df = X_engineered.copy()
    ids = df['ID'].values
    df = df.drop('ID', axis=1)

    obs_indexed = raw_features_df.set_index('ID')

    for col in label_cols:
        if col in obs_indexed.columns:
            obs_vals = obs_indexed[col].reindex(ids).values
            df[f'obs_{col}'] = (~np.isnan(obs_vals.astype(float))).astype(np.int8)

    df['ID'] = ids
    return df
