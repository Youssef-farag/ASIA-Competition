"""Feature engineering for Task 2 (longitudinal): baseline, medium, large and XL feature sets."""

import numpy as np
import pandas as pd

from isncsci.constants import SPINAL_LEVELS


class BaselineFeatureEngineer:
    def __init__(self):
        self.feature_names = None
        self.train_medians = None

    def engineer_features(self, df, metadata, is_training=True):
        df = df.copy()
        metadata = metadata.copy()
        df = df.merge(metadata, on='ID', how='left')
        ids = df['ID'].copy()
        df = df.drop('ID', axis=1)
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        if is_training:
            self.train_medians = df.median()
            self.feature_names = df.columns.tolist()
        df = df.fillna(self.train_medians).fillna(0)
        if not is_training:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        df['ID'] = ids.values
        return df


class MediumFeatureEngineer:
    def __init__(self):
        self.feature_names = None
        self.train_medians = None

    def engineer_features(self, df, metadata, is_training=True):
        df = df.copy()
        metadata = metadata.copy()
        df = df.merge(metadata, on='ID', how='left')
        ids = df['ID'].copy()
        df = df.drop('ID', axis=1)
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        motor_left  = ['elbfll', 'wrextl', 'elbexl', 'finfll', 'finabl',
                       'hipfll', 'kneexl', 'ankdol', 'gretol', 'ankpll']
        motor_right = ['elbflr', 'wrextr', 'elbexr', 'finflr', 'finabr',
                       'hipflr', 'kneetr', 'ankdor', 'gretor', 'ankplr']

        ml_present = [c for c in motor_left  if c in df.columns]
        mr_present = [c for c in motor_right if c in df.columns]
        if ml_present:
            df['motor_left_total'] = df[ml_present].sum(axis=1)
            df['motor_left_mean']  = df[ml_present].mean(axis=1)
        if mr_present:
            df['motor_right_total'] = df[mr_present].sum(axis=1)
            df['motor_right_mean']  = df[mr_present].mean(axis=1)

        w1_ml = ['w1_' + c for c in motor_left  if 'w1_' + c in df.columns]
        w1_mr = ['w1_' + c for c in motor_right if 'w1_' + c in df.columns]
        if w1_ml:
            df['w1_motor_left_total'] = df[w1_ml].sum(axis=1)
            df['w1_motor_left_mean']  = df[w1_ml].mean(axis=1)
        if w1_mr:
            df['w1_motor_right_total'] = df[w1_mr].sum(axis=1)
            df['w1_motor_right_mean']  = df[w1_mr].mean(axis=1)

        if ml_present and w1_ml:
            df['motor_left_change'] = df['motor_left_total'] - df['w1_motor_left_total']
        if mr_present and w1_mr:
            df['motor_right_change'] = df['motor_right_total'] - df['w1_motor_right_total']

        lt_cols    = [c for c in df.columns if 'lt' in c and any(x in c for x in SPINAL_LEVELS) and not c.startswith('w1_')]
        pp_cols    = [c for c in df.columns if 'pp' in c and any(x in c for x in SPINAL_LEVELS) and not c.startswith('w1_')]
        w1_lt_cols = [c for c in df.columns if c.startswith('w1_') and 'lt' in c and any(x in c for x in SPINAL_LEVELS)]
        w1_pp_cols = [c for c in df.columns if c.startswith('w1_') and 'pp' in c and any(x in c for x in SPINAL_LEVELS)]

        if lt_cols:
            df['lt_total'] = df[lt_cols].sum(axis=1)
            df['lt_mean']  = df[lt_cols].mean(axis=1)
        if pp_cols:
            df['pp_total'] = df[pp_cols].sum(axis=1)
            df['pp_mean']  = df[pp_cols].mean(axis=1)
        if w1_lt_cols:
            df['w1_lt_total'] = df[w1_lt_cols].sum(axis=1)
            df['w1_lt_mean']  = df[w1_lt_cols].mean(axis=1)
        if w1_pp_cols:
            df['w1_pp_total'] = df[w1_pp_cols].sum(axis=1)
            df['w1_pp_mean']  = df[w1_pp_cols].mean(axis=1)

        if lt_cols and w1_lt_cols:
            df['lt_change'] = df['lt_total'] - df['w1_lt_total']
        if pp_cols and w1_pp_cols:
            df['pp_change'] = df['pp_total'] - df['w1_pp_total']

        if is_training:
            self.train_medians = df.median()
            self.feature_names = df.columns.tolist()
        df = df.fillna(self.train_medians).fillna(0)
        if not is_training:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        df['ID'] = ids.values
        return df


class LargeFeatureEngineer(MediumFeatureEngineer):
    def engineer_features(self, df, metadata, is_training=True):
        df = super().engineer_features(df, metadata, is_training)
        ids = df['ID'].copy()
        df  = df.drop('ID', axis=1)

        lt_left  = [c for c in df.columns if c.startswith(tuple(SPINAL_LEVELS)) and 'ltl' in c and not c.startswith('w1_')]
        lt_right = [c for c in df.columns if c.startswith(tuple(SPINAL_LEVELS)) and 'ltr' in c and not c.startswith('w1_')]

        regions = {
            'cervical': ['c2','c3','c4','c5','c6','c7','c8'],
            'thoracic': ['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12'],
            'lumbar':   ['l1','l2','l3','l4','l5'],
            'sacral':   ['s1','s2','s3','s45'],
        }
        for region_name, levels in regions.items():
            for modality in ['lt', 'pp']:
                for side in ['l', 'r']:
                    cols = [f'{lvl}{modality}{side}' for lvl in levels if f'{lvl}{modality}{side}' in df.columns]
                    if cols:
                        df[f'{region_name}_{modality}{side}_sum']  = df[cols].sum(axis=1)
                        df[f'{region_name}_{modality}{side}_mean'] = df[cols].mean(axis=1)
                    w1_cols = [f'w1_{lvl}{modality}{side}' for lvl in levels if f'w1_{lvl}{modality}{side}' in df.columns]
                    if w1_cols:
                        df[f'w1_{region_name}_{modality}{side}_sum']  = df[w1_cols].sum(axis=1)
                        df[f'w1_{region_name}_{modality}{side}_mean'] = df[w1_cols].mean(axis=1)
                    if cols and w1_cols:
                        df[f'{region_name}_{modality}{side}_change'] = (
                            df[f'{region_name}_{modality}{side}_sum'] - df[f'w1_{region_name}_{modality}{side}_sum']
                        )

        if 'pp_total' in df.columns and 'lt_total' in df.columns:
            df['modality_dissociation'] = df['pp_total'] - df['lt_total']
        if 'w1_pp_total' in df.columns and 'w1_lt_total' in df.columns:
            df['w1_modality_dissociation'] = df['w1_pp_total'] - df['w1_lt_total']

        if 'motor_left_total' in df.columns and 'motor_right_total' in df.columns:
            df['motor_asymmetry'] = np.abs(df['motor_left_total'] - df['motor_right_total'])

        lt_left_sum  = df[lt_left].sum(axis=1)  if lt_left  else 0
        lt_right_sum = df[lt_right].sum(axis=1) if lt_right else 0
        if lt_left and lt_right:
            df['lt_asymmetry'] = np.abs(lt_left_sum - lt_right_sum)

        s45_cols = [c for c in df.columns if 's45' in c and ('lt' in c or 'pp' in c) and not c.startswith('w1_')]
        if s45_cols:
            df['sacral_sparing'] = (df[s45_cols].sum(axis=1) > 0).astype(int)

        if is_training:
            self.train_medians = df.median()
            self.feature_names = df.columns.tolist()
        df = df.fillna(self.train_medians).fillna(0)
        if not is_training:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        df['ID'] = ids.values
        return df


class XLFeatureEngineer(LargeFeatureEngineer):
    """
    Extends LargeFeatureEngineer with:
      - NLI estimate from w1 sensory data (scalar, 0-27)
      - Recovery rate features: (current - w1) / time for motor and sensory totals
      - Per-timepoint interactions: time * nli_estimate, time * recovery rates
      - Injury-zone category (integer encoding of how far above/below NLI
        the aggregate score lies) — NOTE: this is a *patient-level* feature,
        not per-target. Per-target NLI-relative features require knowing which
        target is being predicted at feature-engineering time, which is awkward
        in this pipeline architecture; instead we encode the NLI index + the
        full w1 sensory profile so the tree models can learn the relationship.
    """

    def engineer_features(self, df, metadata, is_training=True):
        # Start from Large features (already includes w1 regional sums etc.)
        df = super().engineer_features(df, metadata, is_training)
        ids = df['ID'].copy()
        df  = df.drop('ID', axis=1)

        # ── NLI estimation from w1 sensory columns ──────────────────────────
        # w1 sensory is 100% observed so this is always available
        w1_lt_levels = [c for c in df.columns if c.startswith('w1_') and 'lt' in c
                        and any(c.startswith('w1_' + lvl) for lvl in SPINAL_LEVELS)]
        w1_pp_levels = [c for c in df.columns if c.startswith('w1_') and 'pp' in c
                        and any(c.startswith('w1_' + lvl) for lvl in SPINAL_LEVELS)]

        # Vectorised NLI: for each level find whether both modalities have ≥1 side == 2
        # then take the most-caudal such level
        nli_estimates = np.full(len(df), -1, dtype=float)
        for i, lvl in enumerate(SPINAL_LEVELS):
            ltl = f'w1_{lvl}ltl'
            ltr = f'w1_{lvl}ltr'
            ppl = f'w1_{lvl}ppl'
            ppr = f'w1_{lvl}ppr'
            lt_intact = np.zeros(len(df), dtype=bool)
            pp_intact = np.zeros(len(df), dtype=bool)
            for col in [ltl, ltr]:
                if col in df.columns:
                    lt_intact |= (df[col].values == 2)
            for col in [ppl, ppr]:
                if col in df.columns:
                    pp_intact |= (df[col].values == 2)
            both_intact = lt_intact & pp_intact
            nli_estimates[both_intact] = i  # keep updating → takes most-caudal

        df['nli_estimate'] = nli_estimates  # -1 = complete C2, 27 = intact through S45

        # ── Recovery rate features ────────────────────────────────────────────
        # time is how many weeks post-injury the current assessment is
        time_col = 'time' if 'time' in df.columns else None

        if time_col:
            time_safe = df[time_col].replace(0, np.nan)  # avoid div-by-zero

            for motor_side in ['left', 'right']:
                curr_col = f'motor_{motor_side}_total'
                w1_col   = f'w1_motor_{motor_side}_total'
                if curr_col in df.columns and w1_col in df.columns:
                    df[f'motor_{motor_side}_rate'] = (
                        (df[curr_col] - df[w1_col]) / time_safe
                    ).fillna(0)

            for modality in ['lt', 'pp']:
                curr_col = f'{modality}_total'
                w1_col   = f'w1_{modality}_total'
                if curr_col in df.columns and w1_col in df.columns:
                    df[f'{modality}_rate'] = (
                        (df[curr_col] - df[w1_col]) / time_safe
                    ).fillna(0)

            # Regional rates (per-region recovery speed)
            regions = ['cervical', 'thoracic', 'lumbar', 'sacral']
            for region in regions:
                for modality in ['lt', 'pp']:
                    for side in ['l', 'r']:
                        curr_col = f'{region}_{modality}{side}_sum'
                        w1_col   = f'w1_{region}_{modality}{side}_sum'
                        if curr_col in df.columns and w1_col in df.columns:
                            df[f'{region}_{modality}{side}_rate'] = (
                                (df[curr_col] - df[w1_col]) / time_safe
                            ).fillna(0)

            # Time × NLI interaction (injury level × how far into recovery)
            df['time_x_nli'] = df['time'] * (df['nli_estimate'] + 1)  # +1 so -1→0

            # Time × motor recovery rate
            if 'motor_left_rate' in df.columns:
                df['time_x_motor_left_rate']  = df['time'] * df['motor_left_rate']
            if 'motor_right_rate' in df.columns:
                df['time_x_motor_right_rate'] = df['time'] * df['motor_right_rate']

        # ── Per-level bilateral difference and mean features ─────────────────
        # Level-specific left-right asymmetry (finer than global LT/motor asymmetry).
        # Also per-level mean = denoised bilateral signal.
        # Applied to both w1_ (always observed) and current (partially observed) prefixes.
        for pfx in ['w1_', '']:
            for lvl in SPINAL_LEVELS:
                for mod in ['lt', 'pp']:
                    l_col = f'{pfx}{lvl}{mod}l'
                    r_col = f'{pfx}{lvl}{mod}r'
                    if l_col in df.columns and r_col in df.columns:
                        df[f'{pfx}{lvl}_{mod}_diff'] = df[l_col] - df[r_col]
                        df[f'{pfx}{lvl}_{mod}_mean'] = (df[l_col] + df[r_col]) / 2.0

        # ── Regional × time interaction features ─────────────────────────────
        # Each regional w1 sum scaled by time-point captures "how much of this
        # region was intact at the previous assessment and how far into recovery
        # are we now?" — complementary to NLI × time already computed above.
        if time_col:
            regions_for_interaction = ['cervical', 'thoracic', 'lumbar', 'sacral']
            for region in regions_for_interaction:
                for modality in ['lt', 'pp']:
                    for side in ['l', 'r']:
                        w1_sum_col = f'w1_{region}_{modality}{side}_sum'
                        if w1_sum_col in df.columns:
                            df[f'time_x_{w1_sum_col}'] = df['time'] * df[w1_sum_col]

        # ── Injury completeness indicator ────────────────────────────────────
        # NLI == -1 means no intact level found → complete injury
        df['complete_injury'] = (df['nli_estimate'] < 0).astype(int)

        # Broad NLI zone (cervical/thoracic/lumbar/sacral/intact)
        nli_arr = df['nli_estimate'].values
        df['nli_zone_cervical'] = ((nli_arr >= 0)  & (nli_arr <= 6)).astype(int)
        df['nli_zone_thoracic'] = ((nli_arr >= 7)  & (nli_arr <= 18)).astype(int)
        df['nli_zone_lumbar']   = ((nli_arr >= 19) & (nli_arr <= 23)).astype(int)
        df['nli_zone_sacral']   = (nli_arr >= 24).astype(int)

        # ── Align/impute ─────────────────────────────────────────────────────
        if is_training:
            self.train_medians = df.median()
            self.feature_names = df.columns.tolist()

        df = df.fillna(self.train_medians).fillna(0)

        if not is_training:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]

        df['ID'] = ids.values
        return df
