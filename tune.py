"""Per-target hyperparameter tuning (RandomizedSearchCV); writes one JSON per model.

Run from the repository root; pass the result to train.py with --load-params.

  python3 tune.py --task 2 --model ridge --features large
"""

import argparse
import warnings

from isncsci.data import TASKS, load_train
from isncsci.features import get_feature_engineer
from isncsci.tuning import PARAM_GRIDS, tune_per_target

warnings.filterwarnings('ignore')

EXAMPLES = """
Examples:
  python3 tune.py --task 1 --model rf --features large --iter 30
  python3 tune.py --task 2 --model ridge --features large
  python3 tune.py --task 2 --model xgb --features large --use-ordinal --iter 50
  python3 tune.py --task 1 --model xgb --features medium --output my_params.json
"""


def main():
    parser = argparse.ArgumentParser(description='Per-target hyperparameter tuning',
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EXAMPLES)
    parser.add_argument('--task', type=int, choices=sorted(TASKS), required=True, help='1 = single snapshot, 2 = longitudinal')
    parser.add_argument('--model', required=True,
                        choices=['rf', 'xgb', 'xgboost', 'lgb', 'lightgbm', 'lgbm', 'ridge', 'et', 'extratrees'],
                        help='Model type to tune')
    parser.add_argument('--features', default='medium', choices=['baseline', 'medium', 'large', 'xl'],
                        help='Feature set (default: medium; xl exists for task 2 only)')
    parser.add_argument('--iter', type=int, default=20, help='Random search iterations per target (default: 20)')
    parser.add_argument('--cv', type=int, default=3, help='CV folds (default: 3)')
    parser.add_argument('--output', default=None,
                        help='Output JSON (default: tuned_task<task>_<model>[_ordinal]_<features>_per_target.json)')
    parser.add_argument('--use-ordinal', action='store_true',
                        help='Tune the binary classifiers of the ordinal model (neg log-loss on the P(y>0) threshold) '
                             'for use with train.py --use-ordinal')
    args = parser.parse_args()

    task = TASKS[args.task]
    if args.features not in task.feature_sets:
        raise SystemExit(f"task {task.number} has no '{args.features}' feature set (choose from {', '.join(task.feature_sets)})")

    model_type = {'xgboost': 'xgb', 'lightgbm': 'lgb', 'lgbm': 'lgb', 'extratrees': 'et'}.get(args.model.lower(), args.model.lower())
    if args.output is None:
        ordinal_tag = '_ordinal' if args.use_ordinal else ''
        args.output = f'tuned_task{task.number}_{model_type}{ordinal_tag}_{args.features}_per_target.json'

    print('=' * 80)
    print(f"TASK {task.number} PER-TARGET HYPERPARAMETER TUNING")
    print(f"Model: {model_type.upper()}   Features: {args.features}")
    print(f"Mode: {'ORDINAL (binary classifier, neg_log_loss)' if args.use_ordinal else 'REGRESSION (RMSE)'}")
    print(f"Iterations: {args.iter}   CV folds: {args.cv}   Output: {args.output}")
    print('=' * 80)

    features_train, labels_train, metadata_train = load_train(task)
    print(f"\nTrain: {len(features_train)} samples")
    print(f"Engineering {args.features} features...")
    X_train = get_feature_engineer(task.number, args.features).engineer_features(features_train, metadata_train, is_training=True)
    print(f"Features: {X_train.shape[1] - 1}")

    param_grid = PARAM_GRIDS[model_type]
    print("\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    tune_per_target(model_type=model_type, X_train=X_train, y_train=labels_train, param_grid=param_grid,
                    n_iter=args.iter, cv=args.cv, output_file=args.output, use_ordinal=args.use_ordinal)

    print(f"\nDone! Use these parameters with:")
    print(f"  python3 train.py --task {task.number} --models {model_type} --features {args.features} "
          f"{'--use-ordinal ' if args.use_ordinal else ''}"
          f"{'--ordinal-xgb-tuned ' if args.use_ordinal and model_type == 'xgb' else ''}--load-params {args.output}")


if __name__ == '__main__':
    main()
