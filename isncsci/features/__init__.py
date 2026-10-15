"""Feature engineers for both tasks."""

from isncsci.features import task1, task2
from isncsci.features.observation import add_observation_mask_features  # noqa: F401


def get_feature_engineer(task, features_type):
    """Return a fresh feature engineer for `task` (1 or 2) and `features_type`."""
    if task == 1:
        engineers = {'baseline': task1.BaselineFeatureEngineer,
                     'medium': task1.MediumFeatureEngineer,
                     'large': task1.LargeFeatureEngineer}
    elif task == 2:
        engineers = {'baseline': task2.BaselineFeatureEngineer,
                     'medium': task2.MediumFeatureEngineer,
                     'large': task2.LargeFeatureEngineer,
                     'xl': task2.XLFeatureEngineer}
    else:
        raise ValueError(f"unknown task {task}")
    if features_type not in engineers:
        raise ValueError(f"task {task} has no '{features_type}' feature set (choose from {sorted(engineers)})")
    return engineers[features_type]()
