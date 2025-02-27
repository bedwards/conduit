#!/usr/bin/env python

import warnings

warnings.simplefilter("ignore")

from pprint import pprint
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
y = train[["efs", "efs_time"]].join(-naf.cumulative_hazard_, on="efs_time")["y"]

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
m = XGBRegressor(enable_categorical=True, verbosity=0)
y_pred_oof = np.zeros(len(train))

for fold_n, (i_fold, i_oof) in enumerate(kfold.split(train.index)):
    print(f"{fold_n}", end=" ", flush=True)
    m.fit(
        X.iloc[i_fold],
        y.iloc[i_fold],
        eval_set=[(X.iloc[i_oof], y.iloc[i_oof])],
        verbose=False,
    )
    y_pred_oof[i_oof] = m.predict(X.iloc[i_oof])

print()
calc_score(rankdata(y_pred_oof))
