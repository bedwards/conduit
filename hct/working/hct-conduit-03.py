#!/usr/bin/env python

import warnings

warnings.simplefilter("ignore")
import os
import sys
import shutil
from glob import glob
from conduit.hct import Hct
import conduit.duct as duct

SUBMIT_FULL_ENSEMBLE = True

INCLUDE_FIT_ON_KAGGLE = False

PIPES = {
    "xgb_kmrace": {
        "m": "xgb",
        "X": "onehot",
        "y": "kmrace",
    },
    "xgbcat_km": {
        "m": "xgbcat",
        "X": "label",
        "y": "km",
    },
    "cb_km": {
        "m": "cb",
        "X": "label",
        "y": "km",
    },
    "lgb_km": {
        "m": "lgb",
        "X": "label",
        "y": "km",
    },
    "xgbcat_coxph": {
        "m": "xgbcat",
        "X": "label",
        "y": "coxph",
    },
    "cb_coxph": {
        "m": "cb",
        "X": "label",
        "y": "coxph",
    },
    "lgb_coxph": {
        "m": "lgb",
        "X": "label",
        "y": "coxph",
    },
    "xgbcat_narace": {
        "m": "xgbcat",
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
    "xgbcat_kmrace": {
        "m": "xgbcat",
        "X": "label",
        "y": "kmrace",
    },
    "lgb_kmrace": {
        "m": "lgb",
        "X": "label",
        "y": "kmrace",
    },
    "xgbcat_nach": {
        "m": "xgbcat",
        "X": "label",
        "y": "nach",
    },
    "cb_kmrace": {
        "m": "cb",
        "X": "label",
        "y": "kmrace",
    },
    "cb_nach": {
        "m": "cb",
        "X": "label",
        "y": "nach",
    },
    "lgb_na": {
        "m": "lgb",
        "X": "label",
        "y": "narace",
    },
    "xgbcat_cox": {
        "m": "xgbcat",
        "X": "label",
        "y": "cox",
    },
    "cb_cox": {
        "m": "cb",
        "X": "label",
        "y": "cox",
    },
}

if __name__ == "__main__":
    running_on_kaggle = duct.running_on_kaggle()
    if not running_on_kaggle and sys.argv[1:]:
        if sys.argv[1] in PIPES:
            PIPES = {sys.argv[1]: PIPES[sys.argv[1]]}
        elif sys.argv[1] not in [
            "cv_score",
            "optimize_weights",
            "weighted_predict",
            "predict",
            "clean",
        ]:
            raise

    hct = Hct(
        "hct-conduit-03",
        PIPES,
        submit_full_ensemble=SUBMIT_FULL_ENSEMBLE,
        include_fit_on_kaggle=INCLUDE_FIT_ON_KAGGLE,
    )

    if not running_on_kaggle:
        if sys.argv[1:] and sys.argv[1] == "cv_score":
            hct.cv_score()
            sys.exit()

        if sys.argv[1:] and sys.argv[1] == "optimize_weights":
            hct.optimize_weights()
            sys.exit()

        if sys.argv[1:] and sys.argv[1] == "weighted_predict":
            hct.weighted_predict()
            sys.exit()

        if sys.argv[1:] and sys.argv[1] == "predict":
            hct.predict()
            sys.exit()

        if sys.argv[1:] and sys.argv[1] == "clean":
            for fn in glob(f"{hct.csv_path}/*.csv"):
                os.remove(fn)

            for fn in glob(f"{hct.model_path}/*.joblib"):
                os.remove(fn)

        hct.fit()
        hct.cv_score()
        hct.optimize_weights()
        hct.weighted_predict()
        sys.exit()

    # Running on kaggle
    if INCLUDE_FIT_ON_KAGGLE:
        hct.fit()
        hct.cv_score()
        hct.optimize_weights()
    hct.weighted_predict()

    shutil.rmtree(hct.csv_path)

    if INCLUDE_FIT_ON_KAGGLE:
        shutil.rmtree("catboost_info")
        shutil.rmtree(hct.model_path)
