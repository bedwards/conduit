import warnings

warnings.simplefilter("ignore")
import os
import sys
import csv
import shutil
import contextlib
from glob import glob
from functools import cache
from collections import defaultdict
from itertools import combinations, starmap
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.utils import concordance_index
import joblib
from conduit.duct import Duct


class Hct:
    def __init__(
        self,
        name,
        pipes,
        submit_full_ensemble=False,
        include_fit_on_kaggle=False,
    ):
        self.name = name
        self.duct = Duct(
            name,
            pipes,
            "ID",
            ["efs", "efs_time"],
            calc_score=self.calc_score,
            submit_full_ensemble=submit_full_ensemble,
            include_fit_on_kaggle=include_fit_on_kaggle,
            data_dir="equity-post-HCT-survival-predictions",
            lb_dir="hct-leaderboard",
            custom_get_y=self.get_y,
        )
        self.running_on_kaggle = self.duct.running_on_kaggle
        self.train = self.duct.train

    def plot_y_transformation(self, Y_name, Y):
        """Visualization function for target transformations"""
        if self.running_on_kaggle:
            fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharey=True)
            plt.subplots_adjust(hspace=0.3)
            fig.suptitle(f"Y transformation: {Y_name}", fontsize="medium")
            sns.histplot(Y, y="y", hue="efs", ax=axes[0, 0])
            sns.scatterplot(Y, x="efs_time", y="y", hue="efs", ax=axes[0, 1])
            sns.scatterplot(
                Y[Y["efs"] == 0], x="efs_time", y="y", label="efs=0", ax=axes[1, 0]
            )
            axes[1, 0].legend()
            sns.scatterplot(
                Y[Y["efs"] == 1],
                x="efs_time",
                y="y",
                label="efs=1",
                color=sns.color_palette()[1],
                ax=axes[1, 1],
            )
            axes[1, 1].legend()
            plt.tight_layout()
            plt.show()

    @cache
    def get_y(self, encoding_type):
        """Target transformation function with caching"""
        # Different survival analysis transformations
        if encoding_type == "nach":
            naf = NelsonAalenFitter(label="y")
            naf.fit(self.train["efs_time"], event_observed=self.train["efs"])
            Y = self.train[["efs", "efs_time"]].join(
                -naf.cumulative_hazard_, on="efs_time"
            )
            title = "Nelson Aalen cumulative hazard"

        elif encoding_type == "km":
            km = KaplanMeierFitter(label="y")
            km.fit(self.train["efs_time"], event_observed=self.train["efs"])
            Y = self.train[["efs", "efs_time"]].join(
                km.survival_function_, on="efs_time"
            )
            title = "Kaplan Meier survival"

        elif encoding_type == "cox":
            Y = self.train[["efs", "efs_time"]]
            Y["y"] = self.train["efs_time"]
            Y.loc[Y["efs"] == 0, "y"] *= -1
            title = "XGB survival:cox, LGBM Cox"

        elif encoding_type == "kmrace":
            Y = self.train[["efs", "efs_time", "race_group"]].copy()
            Y["y"] = 0
            for race in Y["race_group"].unique():
                mask = Y["race_group"] == race
                kmf = KaplanMeierFitter()
                kmf.fit(Y.loc[mask, "efs_time"], Y.loc[mask, "efs"])
                Y.loc[mask, "y"] = kmf.survival_function_at_times(
                    Y.loc[mask, "efs_time"]
                ).values
                gap = (
                    0.7
                    * (
                        Y.loc[(mask) & (Y["efs"] == 0), "y"].max()
                        - Y.loc[(mask) & (Y["efs"] == 1), "y"].min()
                    )
                    / 2
                )
                Y.loc[(mask) & (Y["efs"] == 0), "y"] -= gap
            title = "Kaplan Meier survival by race"

        else:
            raise

        self.plot_y_transformation(title, Y)
        return Y["y"]

    def calc_score(self, y_pred_oof):
        """Calculate competition metric"""
        merged_df = self.train[["race_group", "efs_time", "efs"]].assign(
            prediction=y_pred_oof
        )
        merged_df = merged_df.reset_index()
        merged_df_race_dict = dict(merged_df.groupby(["race_group"]).groups)
        metric_list = []

        # Calculate concordance index for each race group
        for race in merged_df_race_dict.keys():
            indices = sorted(merged_df_race_dict[race])
            merged_df_race = merged_df.iloc[indices]

            c_index_race = concordance_index(
                merged_df_race["efs_time"],
                -merged_df_race["prediction"],
                merged_df_race["efs"],
            )

            metric_list.append(c_index_race)

        # Return mean - std to encourage consistent performance across groups
        return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
