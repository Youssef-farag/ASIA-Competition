"""Task definitions, data loading and submission file names."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Task:
    number: int
    default_features: str
    feature_sets: tuple

    @property
    def default_dir(self):
        return Path(f'data/track{self.number}')


TASKS = {
    1: Task(1, default_features='medium', feature_sets=('baseline', 'medium', 'large')),
    2: Task(2, default_features='large', feature_sets=('baseline', 'medium', 'large', 'xl')),
}


def load_train(task, data_dir=None):
    """Return (features, labels, metadata) training DataFrames."""
    d = Path(data_dir) if data_dir else task.default_dir
    n = task.number
    return (pd.read_csv(d / f'features_train_{n}.csv'),
            pd.read_csv(d / f'labels_train_{n}.csv'),
            pd.read_csv(d / f'metadata_train_{n}.csv'))


def load_test(task, data_dir=None):
    """Return (features, metadata) test DataFrames."""
    d = Path(data_dir) if data_dir else task.default_dir
    n = task.number
    return pd.read_csv(d / f'features_test_{n}.csv'), pd.read_csv(d / f'metadata_test_{n}.csv')


def submission_name(task, model_name, features, observe_override, hybrid):
    """File name of a submission, matching the names used for the Kaggle uploads."""
    obs = '_obs' if observe_override else ''
    if hybrid:
        return f'submission_task{task.number}_{model_name}{obs}.csv'
    if task.number == 1:
        return f'submission_task1_{model_name}_{features}{obs}.csv'
    return f'submission_task2_v2_{model_name}_{features}{obs}.csv'
