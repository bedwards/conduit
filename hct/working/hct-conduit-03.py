#!/usr/bin/env python

import warnings

warnings.simplefilter("ignore")
import os
import sys
import csv
import shutil
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.utils import concordance_index
import joblib
from conduit.hct import Hct
import conduit.duct as duct

PIPES = {
    "xgb_narace": {
        "m": "xgb",
        "X": "label",
        "y": "narace",
    },
    "cb_narace": {
        "m": "cb",
        "X": "label",
        "y": "narace",
    },
    "lgb_narace": {
        "m": "cb",
        "X": "label",
        "y": "narace",
    },
    # "xgb_kmrace": {
    #     "m": "xgb",
    #     "X": "label",
    #     "y": "kmrace",
    # },
    # "lgb_kmrace": {
    #     "m": "lgb",
    #     "X": "label",
    #     "y": "kmrace",
    # },
    "xgb_na": {
        "m": "xgb",
        "X": "label",
        "y": "na",
    },
    "cb_kmrace": {
        "m": "cb",
        "X": "label",
        "y": "kmrace",
    },
    "cb_na": {
        "m": "cb",
        "X": "label",
        "y": "na",
    },
    # "lgb_na": {
    #     "m": "lgb",
    #     "X": "label",
    #     "y": "narace",
    # },
    "xgb_cox": {
        "m": "xgb",
        "X": "label",
        "y": "cox",
    },
    # "cb_cox": {
    #     "m": "cb",
    #     "X": "label",
    #     "y": "cox",
    # },
}

if __name__ == "__main__":
    running_on_kaggle = duct.running_on_kaggle()
    if not running_on_kaggle and sys.argv[1:]:
        if sys.argv[1] in PIPES:
            PIPES = {sys.argv[1]: PIPES[sys.argv[1]]}
        if sys.argv[1] not in ["cv_score", "predict", "clean"]:
            raise

    hct = Hct(
        "hct-conduit-03", PIPES, submit_full_ensemble=False, include_fit_on_kaggle=False
    )
    duct = hct.duct

    if not running_on_kaggle:
        if sys.argv[1:] and sys.argv[1] == "cv_score":
            duct.cv_score()
            sys.exit()

        if sys.argv[1:] and sys.argv[1] == "predict":
            duct.predict()
            sys.exit()

        if sys.argv[1:] and sys.argv[1] == "clean":
            for fn in glob(f"{duct.csv_path}/*.csv"):
                os.remove(fn)

            for fn in glob(f"{duct.model_path}/*.joblib"):
                os.remove(fn)

        duct.fit()
        duct.cv_score()
        duct.predict()
        sys.exit()

    # Running on kaggle
    if duct.include_fit_on_kaggle:
        duct.fit()
    duct.cv_score()
    duct.predict()

    shutil.rmtree(duct.csv_path)

    if duct.include_fit_on_kaggle:
        shutil.rmtree("catboost_info")
        shutil.rmtree(duct.model_path)
