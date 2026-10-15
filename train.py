"""Train / cross-validate the ensembles and write Kaggle submissions.

Run from the repository root; see README.md for the exact command of each submission.

  python3 train.py --task 1 --models rf:large xgb:medium ridge:baseline \
      --use-ordinal --observe-override --skip-cv --submit
"""

import argparse
import functools
import warnings
from pathlib import Path

from isncsci.data import TASKS, load_test, load_train, submission_name
from isncsci.features import get_feature_engineer
from isncsci.models import EnsembleModel
from isncsci.params import load_per_target_params
from isncsci.postprocess import post_process, post_process_with_observed

warnings.filterwarnings('ignore')

EXAMPLES = """
Examples:
  # Single model, 5-fold CV
  python3 train.py --task 1 --model rf --features large

  # Ensemble on one feature set, with per-target tuned parameters (one JSON per model)
  python3 train.py --task 2 --models rf xgb et ridge --features large \\
      --load-params params/tuned_task2_{rf,xgb,et,ridge}_large_per_target.json

  # Hybrid ensemble: every model has its own feature set
  python3 train.py --task 1 --models rf:large xgb:medium ridge:baseline --skip-cv --submit

  # Optional extras (all off by default)
  python3 train.py --task 2 --models rf et --features xl --use-ordinal --observe-override --skip-cv --submit
  python3 train.py --task 1 --models rf xgb --features large --two-pass --obs-mask-features --cv 3
"""


def build_parser():
    p = argparse.ArgumentParser(description='ISNCSCI sensory score imputation: training and submission',
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EXAMPLES)
    p.add_argument('--task', type=int, choices=sorted(TASKS), required=True,
                   help='1 = single snapshot, 2 = longitudinal')

    models = p.add_argument_group('models and features')
    models.add_argument('--models', nargs='+', default=None,
                        help='Ensemble members, each "model" or "model:features", e.g. rf:large xgb:medium ridge:baseline. '
                             'Models: rf, xgb, lgb, ridge, et')
    models.add_argument('--model', default=None, choices=['rf', 'xgb', 'xgboost', 'lgb', 'lightgbm', 'lgbm', 'ridge', 'et', 'extratrees'],
                        help='Single model (same as --models with one entry)')
    models.add_argument('--features', default=None, choices=['baseline', 'medium', 'large', 'xl'],
                        help='Feature set for members without ":features" '
                             '(default: medium for task 1, large for task 2; xl exists for task 2 only)')
    models.add_argument('--use-xl-features', action='store_true', help='Same as --features xl (task 2)')
    models.add_argument('--load-params', nargs='+', default=None,
                        help='Tuned per-target hyperparameters: one JSON (from tune.py) per model, in model order')

    opts = p.add_argument_group('optional extras (all off by default)')
    opts.add_argument('--use-ordinal', action='store_true',
                      help='Cumulative ordinal classifiers instead of regression')
    opts.add_argument('--ordinal-xgb-tuned', action='store_true',
                      help='Also apply tuned XGBoost parameters to ordinal classifiers (by default they use fixed defaults)')
    opts.add_argument('--observe-override', action='store_true',
                      help='Copy cells already observed in the test features into the submission')
    opts.add_argument('--obs-mask-features', action='store_true',
                      help='Add obs_<col> indicator columns marking directly measured feature cells')
    opts.add_argument('--two-pass', action='store_true',
                      help='Second prediction pass with first-pass predictions substituted for the label columns')

    run = p.add_argument_group('run')
    run.add_argument('--submit', action='store_true', help='Train on the full training set and write a submission CSV')
    run.add_argument('--cv', type=int, default=5, help='CV folds (default: 5)')
    run.add_argument('--skip-cv', action='store_true', help='Skip cross-validation')
    run.add_argument('--data-dir', default=None, help='Data directory (default: data/track<task>)')
    run.add_argument('--out-dir', default='submissions', help='Submission directory (default: submissions)')
    return p


def main():
    args = build_parser().parse_args()
    task = TASKS[args.task]

    if args.models and args.model:
        raise SystemExit("use either --model or --models, not both")
    model_specs = args.models or ([args.model] if args.model else ['rf'])
    features = 'xl' if args.use_xl_features else (args.features or task.default_features)
    if features not in task.feature_sets:
        raise SystemExit(f"task {task.number} has no '{features}' feature set (choose from {', '.join(task.feature_sets)})")
    if args.ordinal_xgb_tuned and not args.use_ordinal:
        print("Note: --ordinal-xgb-tuned only affects --use-ordinal runs")
    hybrid = any(':' in spec for spec in model_specs)

    per_target_params_list = load_per_target_params(args.load_params, model_specs) if args.load_params else None

    # Model name used in the submission file name
    if hybrid:
        model_name = 'hybrid_' + '_'.join(spec.replace(':', '-') for spec in model_specs)
    elif len(model_specs) > 1:
        model_name = 'ensemble_' + '_'.join(model_specs)
    else:
        model_name = model_specs[0]
    model_name = ('ordinal_' if args.use_ordinal else '') + model_name
    if args.use_ordinal and args.ordinal_xgb_tuned:
        model_name += '_xgbtuned'
    if args.two_pass:
        model_name += '_2pass'
    if args.obs_mask_features:
        model_name += '_obsmask'

    print('=' * 80)
    print(f"TASK {task.number}: " + ', '.join(model_specs))
    print(f"Default features : {features}")
    print(f"Mode             : {'ORDINAL' if args.use_ordinal else 'regression'}"
          f"{' + two-pass' if args.two_pass else ''}{' + obs-mask features' if args.obs_mask_features else ''}"
          f"{' + observe override' if args.observe_override else ''}")
    print(f"CV folds         : {'skipped' if args.skip_cv else args.cv}")
    print('=' * 80)

    features_train, labels_train, metadata_train = load_train(task, args.data_dir)
    label_cols = [c for c in labels_train.columns if c != 'ID']
    print(f"\nTrain: {len(features_train)} samples, {len(label_cols)} targets")

    model = EnsembleModel(
        model_specs,
        get_engineer=functools.partial(get_feature_engineer, task.number),
        default_features=features,
        per_target_params_list=per_target_params_list,
        use_ordinal=args.use_ordinal,
        ordinal_xgb_tuned=args.ordinal_xgb_tuned,
        obs_mask_features=args.obs_mask_features,
        two_pass=args.two_pass,
    )

    cv_rmse = None
    if args.skip_cv:
        print("Skipping CV (--skip-cv)")
    else:
        cv_rmse = model.cross_validate(features_train, metadata_train, labels_train, cv=args.cv)

    if args.submit:
        print("\nGenerating submission...")
        features_test, metadata_test = load_test(task, args.data_dir)
        model.train(features_train, metadata_train, labels_train)
        predictions = post_process(model.predict(features_test, metadata_test))
        if args.observe_override:
            predictions = post_process_with_observed(predictions, features_test, label_cols)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / submission_name(task, model_name, features, args.observe_override, hybrid)
        predictions.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")

    if cv_rmse is not None:
        print(f"\nFINAL CV RMSE: {cv_rmse:.4f}")
    return cv_rmse


if __name__ == '__main__':
    main()
