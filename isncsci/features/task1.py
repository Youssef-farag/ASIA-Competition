"""Feature engineering for Task 1 (single snapshot): baseline, medium and large feature sets."""

import numpy as np
import pandas as pd

from isncsci.constants import SPINAL_LEVELS


class BaselineFeatureEngineer:
    """Baseline feature engineering (from train_domain_v2.py)"""
    
    def __init__(self):
        self.feature_names = None
        self.train_medians = None
        
    def engineer_features(self, df, metadata, is_training=True):
        """Create baseline features"""
        df = df.copy()
        metadata = metadata.copy()
        
        # Merge metadata
        df = df.merge(metadata, on='ID', how='left')
        ids = df['ID'].copy()
        df = df.drop('ID', axis=1)
        
        # One-hot encode categorical
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
        # Motor features
        motor_left = ['elbfll', 'wrextl', 'elbexl', 'finfll', 'finabl', 
                      'hipfll', 'kneexl', 'ankdol', 'gretol', 'ankpll']
        motor_right = ['elbflr', 'wrextr', 'elbexr', 'finflr', 'finabr',
                       'hipflr', 'kneetr', 'ankdor', 'gretor', 'ankplr']
        
        ml_present = [c for c in motor_left if c in df.columns]
        mr_present = [c for c in motor_right if c in df.columns]
        
        if ml_present:
            df['motor_left_total'] = df[ml_present].sum(axis=1)
            df['motor_left_mean'] = df[ml_present].mean(axis=1)
            df['motor_left_max'] = df[ml_present].max(axis=1)
            df['motor_left_min'] = df[ml_present].min(axis=1)
        
        if mr_present:
            df['motor_right_total'] = df[mr_present].sum(axis=1)
            df['motor_right_mean'] = df[mr_present].mean(axis=1)
            df['motor_right_max'] = df[mr_present].max(axis=1)
            df['motor_right_min'] = df[mr_present].min(axis=1)
        
        # Bilateral symmetry
        if ml_present and mr_present:
            for l, r in zip(ml_present, mr_present):
                df[f'diff_{l}_{r}'] = df[l] - df[r]
                df[f'asym_{l}_{r}'] = np.abs(df[l] - df[r])
        
        # Sensory features
        lt_cols = [c for c in df.columns if 'lt' in c and any(x in c for x in SPINAL_LEVELS)]
        pp_cols = [c for c in df.columns if 'pp' in c and any(x in c for x in SPINAL_LEVELS)]
        
        if lt_cols:
            df['lt_total'] = df[lt_cols].sum(axis=1)
            df['lt_mean'] = df[lt_cols].mean(axis=1)
            df['lt_std'] = df[lt_cols].std(axis=1)
            df['lt_max'] = df[lt_cols].max(axis=1)
            
        if pp_cols:
            df['pp_total'] = df[pp_cols].sum(axis=1)
            df['pp_mean'] = df[pp_cols].mean(axis=1)
            df['pp_std'] = df[pp_cols].std(axis=1)
            df['pp_max'] = df[pp_cols].max(axis=1)
        
        # Interaction features
        if ml_present and lt_cols:
            df['motor_left_x_lt_mean'] = df['motor_left_mean'] * df['lt_mean']
        if mr_present and pp_cols:
            df['motor_right_x_pp_mean'] = df['motor_right_mean'] * df['pp_mean']
        
        # Store medians for imputation
        if is_training:
            self.train_medians = df.median()
            self.feature_names = df.columns.tolist()
        
        # Impute with training medians
        df = df.fillna(self.train_medians).fillna(0)
        
        # Align test features with training
        if not is_training:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        
        df['ID'] = ids.values
        return df


class MediumFeatureEngineer(BaselineFeatureEngineer):
    """Enhanced feature engineering with additional domain knowledge (from train_domain_v2.py)"""
    
    def _get_sensory_cols(self, df, modality='lt', side='l'):
        cols = []
        for level in SPINAL_LEVELS:
            col_name = f'{level}{modality}{side}'
            if col_name in df.columns:
                cols.append(col_name)
        return cols
    
    def _estimate_nli(self, row, sensory_cols):
        for i, col in enumerate(sensory_cols):
            if pd.notna(row[col]) and row[col] < 2:
                return i
        return len(sensory_cols)
    
    def engineer_features(self, df, metadata, is_training=True):
        """Create baseline features PLUS enhanced domain features"""
        # Start with baseline features (includes motor/sensory aggregates, bilateral diffs, interactions)
        df = super().engineer_features(df, metadata, is_training)
        
        # Remove ID temporarily for additional features
        ids = df['ID'].copy()
        df = df.drop('ID', axis=1)
        
        # Get sensory columns
        lt_left = self._get_sensory_cols(df, 'lt', 'l')
        lt_right = self._get_sensory_cols(df, 'lt', 'r')
        pp_left = self._get_sensory_cols(df, 'pp', 'l')
        pp_right = self._get_sensory_cols(df, 'pp', 'r')
        
        # === ENHANCED FEATURES ===
        
        # 1. Caudal gradient features
        for modality_cols, prefix in [(lt_left, 'ltl'), (lt_right, 'ltr'), 
                                       (pp_left, 'ppl'), (pp_right, 'ppr')]:
            if not modality_cols:
                continue
            for i in range(1, min(len(modality_cols), 10)):  # Limit to avoid too many features
                df[f'{prefix}_cumsum_{i}'] = df[modality_cols[:i+1]].sum(axis=1)
        
        # 2. NLI estimation
        for modality_cols, prefix in [(lt_left, 'nli_ltl'), (lt_right, 'nli_ltr'),
                                       (pp_left, 'nli_ppl'), (pp_right, 'nli_ppr')]:
            if not modality_cols:
                continue
            df[prefix] = df.apply(lambda row: self._estimate_nli(row, modality_cols), axis=1)
        
        # 3. Zone of partial preservation
        for modality_cols, nli_col, prefix in [
            (lt_left, 'nli_ltl', 'zpp_ltl'), (lt_right, 'nli_ltr', 'zpp_ltr'),
            (pp_left, 'nli_ppl', 'zpp_ppl'), (pp_right, 'nli_ppr', 'zpp_ppr')]:
            if not modality_cols or nli_col not in df.columns:
                continue
            def calc_zpp(row):
                nli = int(row[nli_col])
                if nli < len(modality_cols):
                    below = [row[c] for c in modality_cols[nli:] if pd.notna(row[c])]
                    return sum(1 for v in below if v > 0)
                return 0
            df[prefix] = df.apply(calc_zpp, axis=1)
        
        # 4. Regional aggregations
        regions = {
            'cervical': ['c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'],
            'thoracic': ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12'],
            'lumbar': ['l1', 'l2', 'l3', 'l4', 'l5'],
            'sacral': ['s1', 's2', 's3', 's45']
        }
        for region_name, levels in regions.items():
            for modality in ['lt', 'pp']:
                for side in ['l', 'r']:
                    cols = [f'{lvl}{modality}{side}' for lvl in levels if f'{lvl}{modality}{side}' in df.columns]
                    if cols:
                        df[f'{region_name}_{modality}{side}_sum'] = df[cols].sum(axis=1)
                        df[f'{region_name}_{modality}{side}_mean'] = df[cols].mean(axis=1)
        
        # 5. Modality dissociation (PP - LT)
        for level in SPINAL_LEVELS:
            for side in ['l', 'r']:
                lt_col = f'{level}lt{side}'
                pp_col = f'{level}pp{side}'
                if lt_col in df.columns and pp_col in df.columns:
                    df[f'{level}_{side}_modality_diff'] = df[pp_col] - df[lt_col]
        
        # Total modality dissociation
        if 'pp_total' in df.columns and 'lt_total' in df.columns:
            df['total_modality_dissociation'] = df['pp_total'] - df['lt_total']
        
        # 6. Injury pattern indicators
        upper_motor_cols = [c for c in df.columns if any(x in c for x in ['elb', 'wr', 'fin']) and c.startswith(('elbfll', 'elbflr', 'wrextl', 'wrextr', 'elbexl', 'elbexr', 'finfll', 'finflr', 'finabl', 'finabr'))]
        lower_motor_cols = [c for c in df.columns if any(x in c for x in ['hip', 'knee', 'ank', 'gre']) and c.startswith(('hipfll', 'hipflr', 'kneexl', 'kneetr', 'ankdol', 'ankdor', 'gretol', 'gretor', 'ankpll', 'ankplr'))]
        
        if upper_motor_cols and lower_motor_cols:
            df['upper_motor_score'] = df[upper_motor_cols].sum(axis=1)
            df['lower_motor_score'] = df[lower_motor_cols].sum(axis=1)
            df['central_cord_index'] = df['lower_motor_score'] - df['upper_motor_score']
        
        # Brown-Séquard indicators (asymmetry)
        if 'motor_left_total' in df.columns and 'motor_right_total' in df.columns:
            df['brown_sequard_motor'] = np.abs(df['motor_left_total'] - df['motor_right_total'])
        
        lt_left_sum = df[lt_left].sum(axis=1) if lt_left else 0
        lt_right_sum = df[lt_right].sum(axis=1) if lt_right else 0
        pp_left_sum = df[pp_left].sum(axis=1) if pp_left else 0
        pp_right_sum = df[pp_right].sum(axis=1) if pp_right else 0
        
        if lt_left and lt_right:
            df['brown_sequard_lt'] = np.abs(lt_left_sum - lt_right_sum)
        if pp_left and pp_right:
            df['brown_sequard_pp'] = np.abs(pp_left_sum - pp_right_sum)
        
        # 7. Sacral sparing
        s45_cols = [c for c in df.columns if 's45' in c and ('lt' in c or 'pp' in c)]
        if s45_cols:
            df['sacral_sparing'] = (df[s45_cols].sum(axis=1) > 0).astype(int)
            df['sacral_sparing_score'] = df[s45_cols].sum(axis=1)
        
        # Update feature names and medians if training
        if is_training:
            self.train_medians = df.median()
            self.feature_names = df.columns.tolist()
        
        # Fill missing values
        df = df.fillna(self.train_medians).fillna(0)
        
        # Align test features
        if not is_training:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        
        df['ID'] = ids.values
        return df


class LargeFeatureEngineer:
    """Domain feature engineering with comprehensive spinal cord injury domain knowledge (from train_domain.py)"""
    
    def __init__(self):
        self.feature_names = None
        self.train_medians = None
    
    def _get_sensory_cols(self, df, modality='lt', side='l'):
        """Get sensory columns for a modality (lt/pp) and side (l/r)"""
        suffix = side
        cols = []
        for level in SPINAL_LEVELS:
            col_name = f'{level}{modality}{suffix}'
            if col_name in df.columns:
                cols.append(col_name)
        return cols
    
    def _estimate_nli(self, row, sensory_cols):
        """Estimate neurological level of injury from sensory scores"""
        # NLI is the most caudal level with normal (2) sensation
        for i, col in enumerate(sensory_cols):
            if pd.notna(row[col]) and row[col] < 2:
                # Found first abnormal level
                return i
        return len(sensory_cols)  # No injury detected
    
    def engineer_features(self, df, metadata, is_training=True):
        """Create domain-informed features"""
        df = df.copy()
        metadata = metadata.copy()
        
        # Merge metadata
        df = df.merge(metadata, on='ID', how='left')
        ids = df['ID'].copy()
        df = df.drop('ID', axis=1)
        
        # One-hot encode categorical
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
        # === DOMAIN-SPECIFIC FEATURE ENGINEERING ===
        
        # Get sensory columns by modality and side
        lt_left = self._get_sensory_cols(df, 'lt', 'l')
        lt_right = self._get_sensory_cols(df, 'lt', 'r')
        pp_left = self._get_sensory_cols(df, 'pp', 'l')
        pp_right = self._get_sensory_cols(df, 'pp', 'r')
        
        # 1. CAUDAL GRADIENT FEATURES
        # Key insight: If level N is impaired, levels N+1, N+2, ... should also be impaired
        for modality_cols, prefix in [(lt_left, 'ltl'), (lt_right, 'ltr'), 
                                       (pp_left, 'ppl'), (pp_right, 'ppr')]:
            if not modality_cols:
                continue
                
            # Running sum from top (should plateau after injury)
            for i in range(1, len(modality_cols)):
                df[f'{prefix}_cumsum_{i}'] = df[modality_cols[:i+1]].sum(axis=1)
            
            # Running mean (should decrease after injury)
            for i in range(1, len(modality_cols)):
                df[f'{prefix}_cummean_{i}'] = df[modality_cols[:i+1]].mean(axis=1)
        
        # 2. ESTIMATED NEUROLOGICAL LEVEL OF INJURY (NLI)
        # Most caudal level with normal (score=2) function
        for modality_cols, prefix in [(lt_left, 'nli_ltl'), (lt_right, 'nli_ltr'),
                                       (pp_left, 'nli_ppl'), (pp_right, 'nli_ppr')]:
            if not modality_cols:
                continue
            nli_vals = []
            for idx, row in df.iterrows():
                nli = self._estimate_nli(row, modality_cols)
                nli_vals.append(nli)
            df[prefix] = nli_vals
        
        # 3. ZONE OF PARTIAL PRESERVATION (ZPP)
        # Count of levels below NLI that still have some function
        for modality_cols, nli_col, prefix in [
            (lt_left, 'nli_ltl', 'zpp_ltl'),
            (lt_right, 'nli_ltr', 'zpp_ltr'),
            (pp_left, 'nli_ppl', 'zpp_ppl'),
            (pp_right, 'nli_ppr', 'zpp_ppr')]:
            if not modality_cols or nli_col not in df.columns:
                continue
            zpp_vals = []
            for idx, row in df.iterrows():
                nli = int(row[nli_col])
                if nli < len(modality_cols):
                    below_nli = [row[c] for c in modality_cols[nli:] if pd.notna(row[c])]
                    zpp = sum(1 for v in below_nli if v > 0)  # Count non-zero scores below NLI
                else:
                    zpp = 0
                zpp_vals.append(zpp)
            df[prefix] = zpp_vals
        
        # 4. BILATERAL SYMMETRY FEATURES
        # Spinal injuries often (but not always) affect both sides
        for lt_l, lt_r in zip(lt_left, lt_right):
            if lt_l in df.columns and lt_r in df.columns:
                level = lt_l[:-3]  # Remove 'ltl' suffix
                df[f'{level}_lt_diff'] = df[lt_l] - df[lt_r]
                df[f'{level}_lt_asym'] = np.abs(df[lt_l] - df[lt_r])
                df[f'{level}_lt_min'] = df[[lt_l, lt_r]].min(axis=1)
                df[f'{level}_lt_max'] = df[[lt_l, lt_r]].max(axis=1)
        
        for pp_l, pp_r in zip(pp_left, pp_right):
            if pp_l in df.columns and pp_r in df.columns:
                level = pp_l[:-3]  # Remove 'ppl' suffix
                df[f'{level}_pp_diff'] = df[pp_l] - df[pp_r]
                df[f'{level}_pp_asym'] = np.abs(df[pp_l] - df[pp_r])
                df[f'{level}_pp_min'] = df[[pp_l, pp_r]].min(axis=1)
                df[f'{level}_pp_max'] = df[[pp_l, pp_r]].max(axis=1)
        
        # 5. ADJACENT LEVEL CORRELATIONS
        # Strong correlation between adjacent spinal levels
        all_sensory = lt_left + lt_right + pp_left + pp_right
        for i in range(len(all_sensory) - 1):
            if all_sensory[i] in df.columns and all_sensory[i+1] in df.columns:
                df[f'adj_diff_{i}'] = df[all_sensory[i]] - df[all_sensory[i+1]]
        
        # 6. REGION-SPECIFIC AGGREGATIONS
        # Cervical (C2-C8), Thoracic (T1-T12), Lumbar (L1-L5), Sacral (S1-S45)
        regions = {
            'cervical': ['c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'],
            'thoracic': ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12'],
            'lumbar': ['l1', 'l2', 'l3', 'l4', 'l5'],
            'sacral': ['s1', 's2', 's3', 's45']
        }
        
        for region_name, levels in regions.items():
            for modality in ['lt', 'pp']:
                for side in ['l', 'r']:
                    cols = [f'{lvl}{modality}{side}' for lvl in levels if f'{lvl}{modality}{side}' in df.columns]
                    if cols:
                        df[f'{region_name}_{modality}{side}_sum'] = df[cols].sum(axis=1)
                        df[f'{region_name}_{modality}{side}_mean'] = df[cols].mean(axis=1)
                        df[f'{region_name}_{modality}{side}_std'] = df[cols].std(axis=1)
                        df[f'{region_name}_{modality}{side}_max'] = df[cols].max(axis=1)
                        df[f'{region_name}_{modality}{side}_min'] = df[cols].min(axis=1)
        
        # 7. MOTOR SCORE FEATURES (if available)
        motor_left = ['elbfll', 'wrextl', 'elbexl', 'finfll', 'finabl', 
                      'hipfll', 'kneexl', 'ankdol', 'gretol', 'ankpll']
        motor_right = ['elbflr', 'wrextr', 'elbexr', 'finflr', 'finabr',
                       'hipflr', 'kneetr', 'ankdor', 'gretor', 'ankplr']
        
        ml_present = [c for c in motor_left if c in df.columns]
        mr_present = [c for c in motor_right if c in df.columns]
        
        if ml_present:
            df['motor_left_total'] = df[ml_present].sum(axis=1)
            df['motor_left_mean'] = df[ml_present].mean(axis=1)
            df['motor_left_max'] = df[ml_present].max(axis=1)
            df['motor_left_min'] = df[ml_present].min(axis=1)
            # Upper vs lower motor
            upper_motor_l = [c for c in ml_present if any(x in c for x in ['elb', 'wr', 'fin'])]
            lower_motor_l = [c for c in ml_present if any(x in c for x in ['hip', 'knee', 'ank', 'gre'])]
            if upper_motor_l:
                df['motor_upper_left'] = df[upper_motor_l].sum(axis=1)
            if lower_motor_l:
                df['motor_lower_left'] = df[lower_motor_l].sum(axis=1)
        
        if mr_present:
            df['motor_right_total'] = df[mr_present].sum(axis=1)
            df['motor_right_mean'] = df[mr_present].mean(axis=1)
            df['motor_right_max'] = df[mr_present].max(axis=1)
            df['motor_right_min'] = df[mr_present].min(axis=1)
            upper_motor_r = [c for c in mr_present if any(x in c for x in ['elb', 'wr', 'fin'])]
            lower_motor_r = [c for c in mr_present if any(x in c for x in ['hip', 'knee', 'ank', 'gre'])]
            if upper_motor_r:
                df['motor_upper_right'] = df[upper_motor_r].sum(axis=1)
            if lower_motor_r:
                df['motor_lower_right'] = df[lower_motor_r].sum(axis=1)
        
        # Motor symmetry
        if ml_present and mr_present:
            df['motor_total_diff'] = df['motor_left_total'] - df['motor_right_total']
            df['motor_total_asym'] = np.abs(df['motor_total_diff'])
        
        # 8. SENSORY-MOTOR INTERACTIONS
        # Motor function correlates with sensory at similar levels
        if ml_present and lt_left:
            df['motor_x_lt_left'] = df['motor_left_total'] * df[lt_left].mean(axis=1)
        if mr_present and lt_right:
            df['motor_x_lt_right'] = df['motor_right_total'] * df[lt_right].mean(axis=1)
        if ml_present and pp_left:
            df['motor_x_pp_left'] = df['motor_left_total'] * df[pp_left].mean(axis=1)
        if mr_present and pp_right:
            df['motor_x_pp_right'] = df['motor_right_total'] * df[pp_right].mean(axis=1)
        
        # 9. COMPLETE vs INCOMPLETE INJURY INDICATOR
        # Based on sacral sparing (S4-5 sensation indicates incomplete)
        s45_cols = [c for c in df.columns if 's45' in c and ('lt' in c or 'pp' in c)]
        if s45_cols:
            df['sacral_sparing'] = (df[s45_cols].sum(axis=1) > 0).astype(int)
        
        # 10. OVERALL SENSORY TOTALS
        all_lt = lt_left + lt_right
        all_pp = pp_left + pp_right
        if all_lt:
            df['lt_total'] = df[all_lt].sum(axis=1)
            df['lt_mean'] = df[all_lt].mean(axis=1)
            df['lt_std'] = df[all_lt].std(axis=1)
        if all_pp:
            df['pp_total'] = df[all_pp].sum(axis=1)
            df['pp_mean'] = df[all_pp].mean(axis=1)
            df['pp_std'] = df[all_pp].std(axis=1)
        
        # Store for test set alignment
        if is_training:
            self.train_medians = df.median()
            self.feature_names = df.columns.tolist()
        
        # Impute missing
        df = df.fillna(self.train_medians).fillna(0)
        
        # Align test features
        if not is_training:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        
        df['ID'] = ids.values
        return df
