from duct import Competition, Pipeline
from survival import KMFTransformer, NAFTransformer
import pandas as pd
import numpy as np

class HCTCompetition(Competition):
    """CIBMTR competition implementation"""
    
    def __init__(self):
        super().__init__("hct", "../input/equity-post-HCT-survival-predictions")
        self.race_weights = {
            'American Indian or Alaska Native': 0.68,
            'Asian': 0.7,
            'Black or African-American': 0.67,
            'More than one race': 0.68,
            'Native Hawaiian or other Pacific Islander': 0.66,
            'White': 0.64
        }

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Competition-specific feature engineering"""
        df = df.copy()
        # Age grouping
        df['age_group'] = df['age_at_hct'] // 10
        # Interaction features
        df['comorbidity_karnofsky'] = df['comorbidity_score'] * df['karnofsky_score']
        # Handle outliers
        df['year_hct'] = df['year_hct'].replace(2020, 2019)
        # Add NaN count
        df['nan_count'] = df.isna().sum(axis=1)
        return df

    def transform_target(self, df: pd.DataFrame) -> pd.Series:
        """Competition-specific target transformation"""
        pass
