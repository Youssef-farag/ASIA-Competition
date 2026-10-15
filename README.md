# Imputing full ISNCSCI sensory scores from the expedited exam

Code for our submissions to the ASIA Challenge 2026 (Sygen clinical trial data), Task 1 (single snapshot) and Task 2 (longitudinal).

Paper: _link to be added_

## Abstract

The International Standards for Neurological Classification of Spinal Cord Injury (ISNCSCI) examination is the standard
measure of injury severity, but the full assessment is time-consuming. The expedited E-ISNCSCI examination
captures key muscle function only, leaving most sensory scores unobserved. The ASIA Challenge 2026 asks
participants to impute the 112 missing sensory scores (light touch and pin prick, each 0/1/2) from E-ISNCSCI
features, metadata and demographics, either from a single time point (Task 1) or with an earlier full
ISNCSCI exam available (Task 2). We average per-target ensembles of random forest, gradient-boosted tree, extra-trees
and ridge models over engineered neurological features, treat each score as an ordinal outcome through cumulative
binary classifiers, and copy through scores already observed in the test features. The best submissions reach a private
RMSE of 0.41291 (Task 1) and 0.40832 (Task 2).

## Results

Kaggle RMSE (lower is better).

| Task | Submission | Private | Public |
|---|---|---|---|
| 1 | Ordinal hybrid ensemble + observed override | 0.41291 | 0.40487 |
| 1 | Regression hybrid ensemble + observed override | 0.41623 | 0.40666 |
| 2 | Ordinal ensemble + observed override | 0.40832 | 0.38601 |
| 2 | Regression ensemble + observed override | 0.41321 | 0.38980 |

## Method

- **Models:** one model per target (112), predictions averaged across members.
  Task 1 hybrid: Random Forest on `large` features, XGBoost on `medium`, Ridge on `baseline`.
  Task 2: Random Forest, XGBoost, ExtraTrees and Ridge, all on `large` features.
- **Ordinal classifiers** (`--use-ordinal`): two binary classifiers per target, P(y>0) and P(y>1),
  combined as `P(y>0) + min(P(y>1), P(y>0))`.
- **Observed override** (`--observe-override`): cells already present in the test features replace the prediction.
- Predictions are clipped to [0, 2] and rounded to 4 decimals. All seeds are fixed (`random_state=42`).

## Installation

```bash
pip install -r requirements.txt
```

## Data

The data is provided by the challenge organizers on request and is not included. Place the files in:

```
data/track1/   features_{train,test}_1.csv  labels_train_1.csv  metadata_{train,test}_1.csv
data/track2/   features_{train,test}_2.csv  labels_train_2.csv  metadata_{train,test}_2.csv
```

## Usage

Run from the repository root. Submissions are written to `submissions/`. Drop `--skip-cv` to also run
cross-validation (`--cv` sets the number of folds, default 5).

```bash
./reproduce_all.sh      # regenerates all four submissions below
```

Re-running reproduces the submitted files up to rounding: a handful of cells can differ by 0.0001 between runs,
because scikit-learn sums random forest predictions across threads in a nondeterministic order.

**Task 1, ordinal hybrid ensemble + observed override**
```bash
python3 train.py --task 1 --models rf:large xgb:medium ridge:baseline \
    --use-ordinal --observe-override --skip-cv --submit
```

**Task 1, regression hybrid ensemble + observed override**
```bash
python3 train.py --task 1 --models rf:large xgb:medium ridge:baseline \
    --observe-override --skip-cv --submit
```

**Task 2, ordinal ensemble + observed override**
```bash
python3 train.py --task 2 --models rf xgb et ridge --features large --use-ordinal \
    --load-params params/tuned_task2_{rf,xgb,et,ridge}_large_per_target.json \
    --observe-override --skip-cv --submit
```

**Task 2, regression ensemble + observed override**
```bash
python3 train.py --task 2 --models rf xgb et ridge --features large \
    --load-params params/tuned_task2_{rf,xgb,et,ridge}_large_per_target.json \
    --observe-override --skip-cv --submit
```

### Options

| Option | Description | Used in final submissions |
|---|---|---|
| `--models` | Ensemble members, `model` or `model:features` (`rf`, `xgb`, `lgb`, `ridge`, `et`) | yes |
| `--features` | Feature set for members without `:features`: `baseline`, `medium`, `large`, `xl` (task 2 only) | yes |
| `--load-params` | Tuned per-target hyperparameters, one JSON per model in model order | Task 2 |
| `--use-ordinal` | Cumulative ordinal classifiers instead of regression | yes |
| `--observe-override` | Copy cells already observed in the test features into the submission | yes |
| `--obs-mask-features` | Add `obs_<col>` indicator columns marking directly measured feature cells | no |
| `--two-pass` | Second prediction pass with first-pass predictions substituted for the label columns | no |
| `--ordinal-xgb-tuned` | Also apply tuned XGBoost parameters to ordinal classifiers | no |
| `--use-xl-features` | Same as `--features xl` | no |
| `--submit` / `--skip-cv` / `--cv N` | Write a submission / skip cross-validation / number of CV folds | |
| `--data-dir` / `--out-dir` | Data directory (default `data/track<task>`) / output directory (default `submissions`) | |

### Configuration

| | Hyperparameters |
|---|---|
| Task 1 | Defaults. Tuning gave no consistent improvement on the leaderboard (tuned vs. default, both with the override: private 0.41478 vs. 0.41623, public 0.40866 vs. 0.40666), so tuned Task 1 parameters are not provided and are not used in reproduction. |
| Task 2 | Per-target tuned parameters in `params/`, loaded with `--load-params` (one file per model, in model order). |

In ordinal runs, tuned parameters apply to the `rf`, `et` and `lgb` classifiers. XGBoost classifiers use fixed
defaults unless `--ordinal-xgb-tuned` is given.

### Tuning

```bash
python3 tune.py --task 2 --model ridge --features large
```

Per-target `RandomizedSearchCV` (defaults: `--cv 3 --iter 20`), writing one JSON per model
(`--use-ordinal` tunes the classifiers of the ordinal model). Pass the result to `train.py` with `--load-params`.

## Repository structure

```
train.py                      Training, cross-validation and submission (both tasks)
tune.py                       Per-target hyperparameter search
reproduce_all.sh              Regenerates the four final submissions
params/                 Tuned per-target parameters (rf, xgb, et, ridge)
isncsci/
  data.py                     Task definitions, data loading, submission names
  features/                   Feature engineers: task1.py, task2.py, observation.py (obs-mask)
  models/                     regression.py, ordinal.py, ensemble.py (hybrid / two-pass ensemble)
  metrics.py  postprocess.py  Metrics; clipping and observed-value override
  params.py   tuning.py       Loading tuned parameters; search grids and search loop
```

## Citation

_To be added._
