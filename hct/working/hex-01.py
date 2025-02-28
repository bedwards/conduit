#!/usr/bin/env python

import warnings

warnings.simplefilter("ignore")

import os
import sys
import json
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import rankdata
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.utils import concordance_index
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)

BEST_WEIGHTS_FILENAME = "hex-01.json"

X_TRANSFORMATION = {
    "cat_threshold": 0,
}

Y_TRANSFORMATION = {
    "plot": False,
    "by_race": True,
    "gap": 0.35,
    # kms
    "name": "kms",
    "fitter": KaplanMeierFitter,
    "estimate": "survival_function_",
    "sign": 1,
    # nach
    # "name": "nach",
    # "fitter": NelsonAalenFitter,
    # "estimate": "cumulative_hazard_",
    # "sign": -1,
}

data_dir = "equity-post-HCT-survival-predictions"

train = pd.read_csv(f"../input/{data_dir}/train.csv").set_index("ID").sort_index()
train.index = train.index.astype("int32")

test = pd.read_csv(f"../input/{data_dir}/test.csv").set_index("ID").sort_index()
test.index = test.index.astype("int32")


def calc_score(y_pred_oof, details=False, debug=False, indent=0):
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

    scores = list(score_by_race.values())
    scores_mean = np.mean(scores)
    scores_stddev = np.sqrt(np.var(scores))
    score = float(scores_mean - scores_stddev)

    if debug:
        for race, score in score_by_race.items():
            print(f"{' '*indent}{score:.4f} {race}")

        print(
            f"{' '*indent}{score:.4f}: mean={scores_mean:.4f} stddev={scores_stddev:.4f}"
        )

    if details:
        return score, score_by_race, scores_mean, scores_stddev

    return score


def plot_y_transformation(Y):
    if not Y_TRANSFORMATION["plot"]:
        return
    title = f"{Y_TRANSFORMATION['name']} {'by race' if Y_TRANSFORMATION['by_race'] else ""} {Y_TRANSFORMATION['gap']} gap"
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharey=True)
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle(f"Y transformation: {title}", fontsize="medium")
    sns.histplot(Y, y="y", hue="efs", ax=axes[0, 0])
    sns.scatterplot(Y, x="efs_time", y="y", hue="efs", ax=axes[0, 1])
    sns.scatterplot(Y[Y["efs"] == 0], x="efs_time", y="y", label="efs=0", ax=axes[1, 0])
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


def preprocess_X(debug=False):
    X = pd.concat([train.drop(columns=["efs", "efs_time"]), test])
    Xn = X.select_dtypes(["int", "float"])
    for col in Xn:
        if Xn[col].nunique() < X_TRANSFORMATION["cat_threshold"]:
            Xn[col] = Xn[col].fillna(-1).astype("int32")
            Xn[col] = pd.Categorical(
                Xn[col], categories=sorted(Xn[col].unique()), ordered=True
            )
    Xo = X.select_dtypes("object").astype("category")
    for col in Xo:
        Xo[col], _ = Xo[col].factorize(use_na_sentinel=False)
        Xo[col] = Xo[col].astype("int32").astype("category")
    X = pd.concat([Xn, Xo], axis=1)
    X = X[: len(train)]
    cat_features = X.select_dtypes("category").columns.to_list()
    if debug:
        X.info()
        for col in X.select_dtypes("category"):
            print(f"{X[col].cat.categories} {col}")
    assert X.shape == (28800, 57)
    return X, cat_features


def preprocess_y_kms_race(debug=False):
    Y = pd.DataFrame()
    race_groups = train["race_group"].unique()
    if not Y_TRANSFORMATION["by_race"]:
        race_groups = [race_groups]
    else:
        race_groups = [[g] for g in race_groups]
    for race_group in race_groups:
        f = Y_TRANSFORMATION["fitter"](label="y")
        Y_race = train[train["race_group"].isin(race_group)]
        f.fit(Y_race["efs_time"], Y_race["efs"])
        Y_race = Y_race.join(
            Y_TRANSFORMATION["sign"] * getattr(f, Y_TRANSFORMATION["estimate"]),
            on="efs_time",
        )
        if Y_TRANSFORMATION["gap"] != 0:
            gap = Y_TRANSFORMATION["gap"] * (
                Y_race.loc[train["efs"] == 0, "y"].max()
                - Y_race.loc[train["efs"] == 1, "y"].min()
            )
            if debug:
                print(f"{gap:.4f} {race_group}")
            Y_race.loc[train["efs"] == 0, "y"] -= gap
        Y = pd.concat([Y, Y_race])
    Y = Y.sort_index()
    plot_y_transformation(Y)
    return Y["y"]


def preprocess_y_cox():
    Y_cox = train[["efs", "efs_time"]]
    Y_cox["y"] = train["efs_time"]
    Y_cox.loc[Y_cox["efs"] == 0, "y"] *= -1
    return Y_cox["y"]


def as_xgb_aft(X):
    dmatrix = xgb.DMatrix(X, enable_categorical=True)
    efs_time = train["efs_time"].copy().values
    dmatrix.set_float_info("label_lower_bound", efs_time)
    y_upper = efs_time
    y_upper[train["efs"] == 0] = np.inf
    dmatrix.set_float_info("label_upper_bound", y_upper)
    return dmatrix


def as_cb_aft(X, cat_features):
    y_upper = train["efs_time"].rename("y_upper")
    y_upper[train["efs"] == 0] = -1
    label = pd.concat(
        [
            train["efs_time"].rename("y_lower"),
            y_upper,
        ],
        axis=1,
    )
    pool = cb.Pool(X, label=label, cat_features=cat_features)
    return pool


def fit_fold_model(m_name, m_config, fold_n, i_fold, i_oof):
    print(f"{m_name:<7} {fold_n} fit")

    if train[train["efs_time"] < 0].any().any():
        raise ValueError("negative efs_time")

    if np.isinf(train["efs_time"]).values.any():
        raise ValueError("inf in efs_time")

    if np.isinf(train.select_dtypes("float")).values.any():
        raise ValueError("inf in train")

    m = m_config["m"]
    X, cat_features = preprocess_X()

    if m_config["y"] == "aft":
        if m_name.startswith("xgb"):
            X = as_xgb_aft(X)
            X_oof = X.slice(i_oof)
            m = m(X.slice(i_fold), evals=[(X_oof, "eval")], **m_config["fit"])

        elif m_name.startswith("cb"):
            X = as_cb_aft(X, cat_features)
            X_oof = X.slice(i_oof)
            m.fit(X.slice(i_fold), eval_set=X_oof)

        else:
            raise ValueError(f"{m_name} does not support aft")

    else:
        if m_config["y"] == "kms_race":
            y = preprocess_y_kms_race(m_name == "xgb" and fold_n == 0)

        elif m_config["y"] == "cox":
            y = preprocess_y_cox()

        else:
            raise ValueError(f'unknown y transformation: {m_config["y"]}')

        m = m_config["m"]
        X_oof = X.iloc[i_oof]

        m.fit(
            X.iloc[i_fold],
            y.iloc[i_fold],
            eval_set=[(X_oof, y.iloc[i_oof])],
            **m_config["fit"],
        )

    print(f"{m_name:<7} {fold_n} predict")
    y_pred_oof = m.predict(X_oof, **m_config["predict"])

    if m_config["y"] == "aft":
        y_pred_oof *= -1

    return m_name, fold_n, y_pred_oof


def optimize_weights(optimize_n, y_pred_oof_by_m):
    trial_n = [0]

    def objective(trial):
        if trial_n[0] % 10 == 0:
            print(f"{optimize_n} trial {trial_n[0]}")

        trial_n[0] += 1
        y_pred_oof_w = np.zeros(len(train))

        for m_name, y_pred_oof in y_pred_oof_by_m.items():
            y_pred_oof_w += y_pred_oof * trial.suggest_float(m_name, 0, 1)

        score, by_race, mean, stddev = calc_score(y_pred_oof_w, details=True)
        trial.set_user_attr("by_race", by_race)
        trial.set_user_attr("mean", mean)
        trial.set_user_attr("stddev", stddev)
        return score

    study = optuna.create_study(direction="maximize")

    if optimize_n == 0:
        with open(BEST_WEIGHTS_FILENAME) as f:
            study.enqueue_trial((json.load(f)))

    study.optimize(objective, n_trials=100)
    return study


def main():
    _, cat_features = preprocess_X(debug=True)
    xgb_kwargs = dict(enable_categorical=True, max_depth=3, verbosity=0)
    xgb_fit_kwargs = dict(verbose=False)
    cb_kwargs = dict(
        iterations=100,
        learning_rate=0.1,
        bootstrap_type="Bernoulli",
        grow_policy="Depthwise",
        boosting_type="Plain",
        cat_features=cat_features,
        silent=True,
    )

    models = {
        "xgb": {
            "m": xgb.XGBRegressor(**xgb_kwargs),
            "y": "kms_race",
            "fit": xgb_fit_kwargs,
            "predict": {},
        },
        "xgb_cox": {
            "m": xgb.XGBRegressor(
                objective="survival:cox", eval_metric="cox-nloglik", **xgb_kwargs
            ),
            "y": "cox",
            "fit": xgb_fit_kwargs,
            "predict": {},
        },
        "xgb_aft": {
            "m": partial(
                xgb.train,
                dict(
                    objective="survival:aft",
                    eval_metric="aft-nloglik",
                    aft_loss_distribution="normal",
                    aft_loss_distribution_scale=1.0,
                    **xgb_kwargs,
                ),
            ),
            "y": "aft",
            "fit": dict(
                num_boost_round=100,
                verbose_eval=False,
            ),
            "predict": {},
        },
        "cb": {
            "m": cb.CatBoostRegressor(**cb_kwargs),
            "y": "kms_race",
            "fit": {},
            "predict": {},
        },
        "cb_cox": {
            "m": cb.CatBoostRegressor(loss_function="Cox", **cb_kwargs),
            "y": "cox",
            "fit": {},
            "predict": dict(prediction_type="Exponent"),
        },
        "cb_aft": {
            "m": cb.CatBoostRegressor(
                loss_function="SurvivalAft:dist=Normal",
                eval_metric="SurvivalAft",
                **cb_kwargs,
            ),
            "y": "aft",
            "fit": {},
            "predict": {},
        },
        "lgb": {
            "m": lgb.LGBMRegressor(
                categorical_feature=cat_features,
                verbose=-1,
                verbosity=-1,
            ),
            "y": "kms_race",
            "fit": {},
            "predict": {},
        },
    }

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    i_oofs = []
    args = []

    for fold_n, (i_fold, i_oof) in enumerate(
        kfold.split(train.index, train["race_group"])
    ):
        i_oofs.append(i_oof)
        for m_name, m_config in models.items():
            args.append((m_name, m_config, fold_n, i_fold, i_oof))

    with Pool(min(os.cpu_count(), len(args))) as pool:
        m_fold_y_pred_oofs = pool.starmap(fit_fold_model, args)

    y_pred_oof_by_m = defaultdict(lambda: np.zeros(len(train)))

    for m_name, fold_n, y_pred_oof in m_fold_y_pred_oofs:
        y_pred_oof_by_m[m_name][i_oofs[fold_n]] = y_pred_oof

    y_pred_oof_by_m = dict(y_pred_oof_by_m)
    pool_size = os.cpu_count()
    args = [(n, y_pred_oof_by_m) for n in range(pool_size)]

    with Pool(pool_size) as pool:
        studies = pool.starmap(optimize_weights, args)

    print("\nIndividual model scores")

    for m_name in y_pred_oof_by_m.keys():
        y_pred_oof_by_m[m_name] = rankdata(y_pred_oof_by_m[m_name])
        score = calc_score(y_pred_oof_by_m[m_name])
        print(f"  {score:.4f} {m_name}")

    print("\nEvenly-weighted ensemble score")
    y_pred_oof = sum(y_pred_oof_by_m.values())
    calc_score(y_pred_oof, debug=True, indent=2)
    sorted_studies = []

    for study in studies:
        sorted_studies.append((study.best_value, study))

    sorted_studies.sort()
    study = sorted_studies[-1][1]
    print("\nOptimized weights")

    for m_name, weight in study.best_params.items():
        print(f"  {weight:.4f} {m_name}")

    with open(BEST_WEIGHTS_FILENAME, "w") as f:
        json.dump(study.best_params, f)

    print("\nOptimally-weighted ensemble score")

    for race, score in study.best_trial.user_attrs["by_race"].items():
        print(f"{score:.4f} {race}")

    mean = study.best_trial.user_attrs["mean"]
    stddev = study.best_trial.user_attrs["stddev"]
    print(f"  {study.best_value:.4f}: mean={mean:.4f} stddev={stddev:.4f}")

    # Note: when predicting test use all models of type, one for each fold


if __name__ == "__main__":
    main()
