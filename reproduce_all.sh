#!/usr/bin/env bash
# Regenerate the four final Kaggle submissions into submissions/.
# Run from the repo root with the competition data in data/track1 and data/track2.
# Each fit takes a few minutes; all seeds are fixed.
set -euo pipefail
mkdir -p submissions

T2_PARAMS="params/tuned_task2_rf_large_per_target.json
           params/tuned_task2_xgb_large_per_target.json
           params/tuned_task2_et_large_per_target.json
           params/tuned_task2_ridge_large_per_target.json"

# Task 1: ordinal hybrid ensemble + observed override (best)
python3 train.py --task 1 --models rf:large xgb:medium ridge:baseline \
    --use-ordinal --observe-override --skip-cv --submit

# Task 1: regression hybrid ensemble + observed override
python3 train.py --task 1 --models rf:large xgb:medium ridge:baseline \
    --observe-override --skip-cv --submit

# Task 2: regression ensemble + tuned params + observed override
python3 train.py --task 2 --models rf xgb et ridge --features large \
    --load-params $T2_PARAMS --observe-override --skip-cv --submit

# Task 2: ordinal ensemble + tuned params + observed override
python3 train.py --task 2 --models rf xgb et ridge --features large --use-ordinal \
    --load-params $T2_PARAMS --observe-override --skip-cv --submit
