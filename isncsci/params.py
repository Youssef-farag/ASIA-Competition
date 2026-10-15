"""Loading of per-target tuned hyperparameters (JSON files written by tune.py)."""

import json
from pathlib import Path


def load_per_target_params(param_files, model_specs):
    """
    Load one per-target params dict per model, in model order.

    Files are matched to models 1:1; with fewer files than models, one file per unique model
    type is accepted. Returns a list with a dict (or None for defaults) per model.
    """
    n_models = len(model_specs)
    model_types = [spec.split(':')[0] for spec in model_specs]

    print(f"\n{'=' * 60}\nLOADING PER-TARGET PARAMETERS\n{'=' * 60}")
    if len(param_files) == n_models:
        print(f"Mapping {len(param_files)} param files to {n_models} models (1:1)")
        mapping = list(param_files)
    elif len(param_files) < n_models:
        unique_types = list(dict.fromkeys(model_types))
        if len(param_files) == len(unique_types):
            print(f"Mapping {len(param_files)} param files to {len(unique_types)} unique model types: {unique_types}")
            by_type = dict(zip(unique_types, param_files))
            mapping = [by_type[t] for t in model_types]
        else:
            print(f"Warning: {len(param_files)} param files for {n_models} models; "
                  f"the first {len(param_files)} models use them, the rest use defaults")
            mapping = list(param_files) + [None] * (n_models - len(param_files))
    else:
        print(f"Warning: {len(param_files)} param files for {n_models} models; using the first {n_models}")
        mapping = list(param_files)[:n_models]

    params_list = []
    for idx, pf in enumerate(mapping):
        if pf is None:
            params_list.append(None)
            print(f"  Model {idx + 1} ({model_specs[idx]}): no params file, using defaults")
            continue
        path = Path(pf)
        if not path.exists():
            raise SystemExit(f"Params file not found: {pf}")
        with open(path) as f:
            params = json.load(f)
        n_valid = sum(1 for v in params.values() if v.get('cv_rmse') is not None)
        params_list.append(params)
        print(f"  Model {idx + 1} ({model_specs[idx]}): loaded {n_valid} targets from {pf}")
    print('=' * 60 + '\n')
    return params_list
