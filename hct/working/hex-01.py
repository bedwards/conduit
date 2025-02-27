#!/usr/bin/env python

import warnings

warnings.simplefilter("ignore")

import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from scipy.stats import rankdata
from lifelines import NelsonAalenFitter
from lifelines.utils import concordance_index


data_dir = "equity-post-HCT-survival-predictions"

train = pd.read_csv(f"../input/{data_dir}/train.csv").set_index("ID")
train.index = train.index.astype("int32")

test = pd.read_csv(f"../input/{data_dir}/test.csv").set_index("ID")
test.index = test.index.astype("int32")


def calc_score(y_pred_oof):
    df = train.assign(prediction=y_pred_oof).reset_index()
    indices_by_race = {
        race: sorted(indices)
        for race, indices in df.groupby(["race_group"]).groups.items()
    }
    score_by_race = {}
    for race, indices in indices_by_race.items():
        df_race = df.iloc[indices]
        score_by_race[race] = concordance_index(
            df_race["efs_time"], -df_race["prediction"], df_race["efs"]
        )
    for race, score in score_by_race.items():
        print(f"{score:.4f} {race}")
    scores = list(score_by_race.values())
    scores_mean = np.mean(scores)
    scores_stddev = np.sqrt(np.var(scores))
    score = float(scores_mean - scores_stddev)
    print(f"{score:.4f}: mean={scores_mean:.4f} stddev={scores_stddev:.4f}")
    return score


def main():
    X = pd.concat([train.drop(columns=["efs", "efs_time"]), test])
    Xf = X.select_dtypes(["int", "float"]).astype("float32")
    Xo = X.select_dtypes("object").astype("category")
    for col in Xo:
        Xo[col], _ = Xo[col].factorize(use_na_sentinel=False)
        Xo[col] = Xo[col].astype("int32").astype("category")
    X = pd.concat([Xf, Xo], axis=1)
    X = X[: len(train)]

    naf = NelsonAalenFitter(label="y")
    naf.fit(train["efs_time"], event_observed=train["efs"])
    y_nach = train[["efs", "efs_time"]].join(-naf.cumulative_hazard_, on="efs_time")[
        "y"
    ]

    Y_cox = train[["efs", "efs_time"]]
    Y_cox["y"] = train["efs_time"]
    Y_cox.loc[Y_cox["efs"] == 0, "y"] *= -1
    y_cox = Y_cox["y"]

    xgb_kwargs = dict(enable_categorical=True, verbosity=0)

    models = {
        "xgb": {
            "m": XGBRegressor(**xgb_kwargs),
            "y": y_nach,
        },
        "xgb_cox": {
            "m": XGBRegressor(
                objective="survival:cox", eval_metric="cox-nloglik", **xgb_kwargs
            ),
            "y": y_cox,
        },
    }

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred_oof_by_m = defaultdict(lambda: np.zeros(len(train)))

    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(train.index)):
        print(f"fold {fold_n}")
        for m_name, m_config in models.items():
            print(f"  {m_name:<7} fit", end=" ", flush=True)
            m = m_config["m"]
            y = m_config["y"]
            m.fit(
                X.iloc[i_fold],
                y.iloc[i_fold],
                eval_set=[(X.iloc[i_oof], y.iloc[i_oof])],
                verbose=False,
            )
            print("predict")
            y_pred_oof_by_m[m_name][i_oof] = m.predict(X.iloc[i_oof])

    print()
    calc_score(sum(rankdata(y_pred) for y_pred in y_pred_oof_by_m.values()))


if __name__ == "__main__":
    main()
