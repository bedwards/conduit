from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from duct import Competition, Pipeline
from survival import KMFTransformer, NAFTransformer


class HCTCompetition(Competition):
    """CIBMTR competition implementation"""

    def __init__(self):
        super().__init__("hct", "../input/equity-post-HCT-survival-predictions")

    def read_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read and preprocess train/test data"""
        train = self.read_csv("train.csv").set_index("ID")
        test = self.read_csv("test.csv").set_index("ID")
        train.index = train.index.astype("int32")
        test.index = test.index.astype("int32")
        return train, test

    def identify_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify categorical and numerical features"""
        RMV = ["ID", "efs", "efs_time", "y"]
        all_features = [c for c in df.columns if c not in RMV]
        cats = [c for c in all_features if c not in ["age_at_hct", "donor_age"]]
        nums = ["age_at_hct", "donor_age"]
        return {"all": all_features, "categorical": cats, "numerical": nums}

    def prepare_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features for both train and test sets"""
        df = pd.concat([train_df, test_df])

        # Handle outliers
        df["year_hct"] = df["year_hct"].replace(2020, 2019)
        df["karnofsky_score"] = df["karnofsky_score"].replace(40, 50)
        df["hla_high_res_8"] = df["hla_high_res_8"].replace(2, 3)
        df["hla_high_res_6"] = df["hla_high_res_6"].replace(0, 2)
        df["hla_high_res_10"] = df["hla_high_res_10"].replace(3, 4)
        df["hla_low_res_8"] = df["hla_low_res_8"].replace(2, 3)

        # Feature engineering
        df["nan_value_each_row"] = df.isnull().sum(axis=1)
        df["age_group"] = df["age_at_hct"] // 10
        df["donor_age-age_at_hct"] = df["donor_age"] - df["age_at_hct"]
        df["comorbidity_score+karnofsky_score"] = (
            df["comorbidity_score"] + df["karnofsky_score"]
        )
        df["comorbidity_score-karnofsky_score"] = (
            df["comorbidity_score"] - df["karnofsky_score"]
        )
        df["comorbidity_score*karnofsky_score"] = (
            df["comorbidity_score"] * df["karnofsky_score"]
        )
        df["comorbidity_score/karnofsky_score"] = (
            df["comorbidity_score"] / df["karnofsky_score"]
        )

        # Handle categorical features
        df["dri_score"] = df["dri_score"].replace(
            "Missing disease status", "N/A - disease not classifiable"
        )
        df["dri_score_NA"] = df["dri_score"].apply(lambda x: int("N/A" in str(x)))
        for col in ["diabetes", "pulm_moderate", "cardiac"]:
            df.loc[df[col].isna(), col] = "Not done"

        # Split back into train and test
        train = df[: len(train_df)].copy()
        test = df[len(train_df) :].reset_index(drop=True).copy()

        return train, test

    @staticmethod
    def get_categorical_features(df: pd.DataFrame) -> List[str]:
        """Identify categorical features"""
        return [c for c in df.columns if df[c].nunique() < 100]

    def transform_target(self, df: pd.DataFrame) -> pd.Series:
        """Competition-specific target transformation"""
        pass
