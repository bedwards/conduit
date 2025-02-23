#!/usr/bin/env python

# Standard library imports and configuration
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

from hct import HCTCompetition

# Unique identifier for this experiment
NAME = "conduit-03"

# Configuration flags
# if True: submission.csv created using all models from PIPE
# if False: uses ensemble combination with best CV score
SUBMIT_FULL_ENSEMBLE = False

# if True: model fit is run on kaggle
# if False: saved models from local run are used from a kaggle dataset
INCLUDE_FIT_ON_KAGGLE = False

# Debug flag to verify scoring implementation matches competition metric
PROVE_MY_SCORE_IS_SAME = False

# Path configuration
C_PATH = "../input/equity-post-HCT-survival-predictions"
LB_PATH = "../input/hct-leaderboard"
CSV_PATH = f"./csv/{NAME}"
os.makedirs(CSV_PATH, exist_ok=True)

MODEL_PATH = "../input/hct-conduit"

# Detect if running in Kaggle environment
try:
    get_ipython
except NameError:
    RUNNING_ON_KAGGLE = False
else:
    RUNNING_ON_KAGGLE = True

# Configure model save paths based on environment
if RUNNING_ON_KAGGLE:
    if INCLUDE_FIT_ON_KAGGLE:
        MODEL_PATH = f"./models/{NAME}"
        os.makedirs(MODEL_PATH, exist_ok=True)
else:
    os.makedirs(MODEL_PATH, exist_ok=True)

# Model hyperparameters from baseline notebook
kwargs_xgb = dict(
    max_depth=3,
    colsample_bytree=0.5,
    subsample=0.8,
    n_estimators=2000,
    learning_rate=0.02,
    min_child_weight=80,
    enable_categorical=True,
    verbosity=0,
)

kwargs_cb = dict(learning_rate=0.1, grow_policy="Lossguide", silent=True)

kwargs_lgb = dict(
    max_depth=3,
    colsample_bytree=0.4,
    n_estimators=2500,
    learning_rate=0.02,
    num_leaves=8,
    verbose=-1,
    verbosity=-1,
)

# 0.6730 0.2806 xgb-lgb-cb-xgb_cox-cb_cox
# 0.6732 0.2806 xgb-cb-xgb_cox-cb_cox

# Pipeline configurations combining different models and target transformations
PIPES = [
    # {  # m must be unique. lgb, xgb, cb, or lgb_extrainfo, etc.
    #     "m": "xgb",  # 0.6674 0.2302 xgb
    #     "X": "label",  # raw, label, onehot
    #     "y": "kmf",  # kmf, naf, cox
    #     "kwargs": kwargs_xgb,
    # },
    # {
    #     "m": "lgb",  # 0.6659 0.2196 lgb
    #     "X": "label",
    #     "y": "kmf",
    #     "kwargs": kwargs_lgb,
    # },
    # {
    #     "m": "cb",  # 0.6655 0.2196 cb
    #     "X": "label",
    #     "y": "kmf",
    #     "kwargs": kwargs_cb,
    # },
    {
        "m": "xgb_cox",  # 0.6669 0.2242 xgb_cox
        "X": "label",
        "y": "cox",
        "kwargs": dict(
            objective="survival:cox", eval_metric="cox-nloglik", **kwargs_xgb
        ),
    },
    {
        "m": "cb_cox",  # 0.6651 0.2196 cb_cox
        "X": "label",
        "y": "cox",
        "kwargs": dict(
            loss_function="Cox", iterations=400, use_best_model=False, **kwargs_cb
        ),
    },
    # KMF by race variations which address race-specific survival patterns
    {
        "m": "xgb_kmfbyrace",  # 0.6716 0.2671 xgb_kmfbyrace
        "X": "label",
        "y": "kmfbyrace",
        "kwargs": kwargs_xgb,
    },
    {
        "m": "lgb_kmfbyrace",  # 0.6702 0.2593 lgb_kmfbyrace
        "X": "label",
        "y": "kmfbyrace",
        "kwargs": kwargs_lgb,
    },
    {
        "m": "cb_kmfbyrace",  # 0.6698 0.2494 cb_kmfbyrace
        "X": "label",
        "y": "kmfbyrace",
        "kwargs": kwargs_cb,
    },
    # LNAF variations as alternative survival analysis
    # {
    #     "m": "xgb_naf",  # 0.6700 0.2494 xgb_naf
    #     "X": "label",
    #     "y": "naf",
    #     "kwargs": kwargs_xgb,
    # },
    # {
    #     "m": "lgb_naf",  # 0.6685 0.2370 lgb_naf
    #     "X": "label",
    #     "y": "naf",
    #     "kwargs": kwargs_lgb,
    # },
    # {
    #     "m": "cb_naf",  # 0.6696 0.2494 cb_naf
    #     "X": "label",
    #     "y": "naf",
    #     "kwargs": kwargs_cb,
    # },
]

# Configure GPU usage when running on Kaggle
if RUNNING_ON_KAGGLE:
    for pipe in PIPES:
        if pipe["m"] in ["xgb", "xgb_cox"]:
            pipe["kwargs"] = dict(device="cuda", **pipe["kwargs"])
            continue

        if pipe["m"] == "cb":
            pipe["kwargs"] = dict(task_type="GPU", **pipe["kwargs"])
            continue

        if pipe["m"] == "lgb":
            pipe["kwargs"] = dict(device="gpu", **pipe["kwargs"])


def read_csv(path):
    """Helper function to read and preprocess CSV files"""
    df = pd.read_csv(path).set_index("ID")
    df.index = df.index.astype("int32")
    fn = path.split("/")[-1].split(".")[0]
    print(f"read {path}")
    return df


competition = HCTCompetition()
train, test = competition.read_data()
# features = competition.identify_feature_types(train)
# FEATURES = features['all']
# CATS = features['categorical']


@cache
def get_X(train_or_test, name):
    """Feature engineering function with caching for efficiency"""
    # Combine train and test for consistent preprocessing
    X = pd.concat([train.drop(columns=["efs", "efs_time"]), test])

    # Split features by type
    Xf = X.select_dtypes("float").astype("float32")
    Xc = X.select_dtypes("object").astype("category")

    # Handle categorical features based on encoding type
    if name == "raw":
        for col in Xc:
            if Xc[col].isna().any():
                Xc[col] = Xc[col].cat.add_categories("Missing").fillna("Missing")

    elif name == "label":
        for col in Xc:
            Xc[col], _ = Xc[col].factorize(use_na_sentinel=False)
            Xc[col] = Xc[col].astype("int32").astype("category")

    elif name == "onehot":
        Xc = pd.get_dummies(Xc, drop_first=True, dtype="int32")
        Xc.columns = Xc.columns.str.replace(r"[\[\]<]", "_", regex=True)

    else:
        raise

    X = pd.concat([Xf, Xc], axis=1)

    # Return appropriate slice based on train_or_test
    if train_or_test == "train":
        return X[: len(train)]

    if train_or_test == "test":
        return X[len(train) :]

    raise


def plot_y_transformation(Y_name, Y):
    """Visualization function for target transformations"""
    if RUNNING_ON_KAGGLE:
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
def get_y(name):
    """Target transformation function with caching"""
    # Different survival analysis transformations
    if name == "naf":
        naf = NelsonAalenFitter(label="y")
        naf.fit(train["efs_time"], event_observed=train["efs"])
        Y = train[["efs", "efs_time"]].join(-naf.cumulative_hazard_, on="efs_time")
        title = "Nelson Aalen cumulative hazard"

    elif name == "kmf":
        kmf = KaplanMeierFitter(label="y")
        kmf.fit(train["efs_time"], event_observed=train["efs"])
        Y = train[["efs", "efs_time"]].join(kmf.survival_function_, on="efs_time")
        title = "Kaplan Meier survival"

    elif name == "cox":
        Y = train[["efs", "efs_time"]]
        Y["y"] = train["efs_time"]
        Y.loc[Y["efs"] == 0, "y"] *= -1
        title = "XGB survival:cox, LGBM Cox"

    elif name == "kmfbyrace":
        Y = train[["efs", "efs_time", "race_group"]].copy()
        Y["y"] = 0
        for race in Y["race_group"].unique():
            mask = Y["race_group"] == race
            kmf = KaplanMeierFitter()
            kmf.fit(Y.loc[mask, "efs_time"], Y.loc[mask, "efs"])
            Y.loc[mask, "y"] = kmf.survival_function_at_times(
                Y.loc[mask, "efs_time"]
            ).values
            # Add gap adjustment for efs=0
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

    elif name == "logitscaled":
        Y = train[["efs", "efs_time"]].copy()
        max_time, min_time = 80, -100
        Y["y"] = Y["efs_time"] / (max_time - min_time)
        Y["y"] = Y["y"].apply(lambda x: np.log(x) - np.log(1 - x))
        Y["y"] += 10
        Y["y"] = -Y["y"]
        title = "logit scaled"

    else:
        raise

    plot_y_transformation(title, Y)
    return Y["y"]


def create_kfold():
    """Create consistent cross-validation splits"""
    return KFold(n_splits=10, shuffle=True, random_state=42)


def fit_fold_model(pipe, fold_n, i_fold, i_oof):
    """Train individual fold model and save to disk"""
    m_name, m_kwargs, X_name, y_name = [pipe[k] for k in ["m", "kwargs", "X", "y"]]
    X = get_X("train", X_name)
    y = get_y(y_name)

    if m_name.startswith("cb") and X_name in ["raw", "label"]:
        m_kwargs["cat_features"] = X.select_dtypes("category").columns.to_list()

        if not m_kwargs["cat_features"]:
            raise

    # Map model names to their constructor classes
    m_constructors = {
        "xgb": xgb.XGBRegressor,
        "lgb": lgb.LGBMRegressor,
        "cb": cb.CatBoostRegressor,
    }

    Model = m_constructors[m_name.split("_")[0]]
    m = Model(**m_kwargs)

    fit_kwargs = dict(
        eval_set=[
            # (X.iloc[i_fold], y.iloc[i_fold]),
            (X.iloc[i_oof], y.iloc[i_oof])
        ]
    )

    if m_name.startswith("xgb"):
        fit_kwargs["verbose"] = False

    if fold_n == 0:
        msg = [f"{m_name}.fit"]

        for k, v in fit_kwargs.items():
            msg.append(f"  {k}={str(v)[:60]}")

        print("\n".join(msg))

    m.fit(X.iloc[i_fold], y.iloc[i_fold], **fit_kwargs)

    filename = f"{MODEL_PATH}/{NAME}-{m_name}-{fold_n}.joblib"
    joblib.dump(m, filename)
    print(f"wrote {filename}")


def fit():
    """Main training function"""
    print("\nfit")
    kfold = create_kfold()

    args = []

    # Create training tasks for each fold and model
    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(train)):
        for pipe in PIPES:
            args.append((pipe, fold_n, i_fold, i_oof))

    # Execute training tasks (sequential on Kaggle, parallel locally)
    if RUNNING_ON_KAGGLE:
        list(starmap(fit_fold_model, args))
    else:
        from multiprocessing import Pool

        with Pool(min(os.cpu_count(), kfold.get_n_splits() * len(args))) as pool:
            pool.starmap(fit_fold_model, args)


def calc_score(y_pred_oof):
    """Calculate competition metric"""
    merged_df = train[["race_group", "efs_time", "efs"]].assign(prediction=y_pred_oof)
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


def cv_score():
    """Calculate cross-validation scores"""
    print("\nCV score")
    kfold = create_kfold()
    y_pred_oofs = defaultdict(lambda: np.zeros(len(train)))

    # Generate OOF predictions for each model
    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(train)):
        for pipe in PIPES:
            m_name = pipe["m"]
            X = get_X("train", pipe["X"])
            for fn in glob(f"{MODEL_PATH}/{NAME}-{m_name}-{fold_n}.joblib"):
                m = joblib.load(fn)
                print(f"read {fn}")
                y_pred_oofs[m_name][i_oof] = m.predict(X.iloc[i_oof])

    # Convert raw predictions to ranks
    for m_name, raw_risk_score in y_pred_oofs.items():
        y_pred_oofs[m_name] = rankdata(raw_risk_score)

    # Load leaderboard for percentile calculation
    fn = sorted(glob(f"{LB_PATH}/*.csv"))[-1]
    lb = pd.read_csv(fn)
    print(f"read {fn}")
    lb = np.sort(lb["Score"])

    def leaderboard_percentile(score):
        return np.searchsorted(lb, score) / len(lb)

    scores = []

    # Try all possible model combinations
    for r in range(1, len(y_pred_oofs) + 1):
        for m_combo in combinations(y_pred_oofs.items(), r):
            m_names, ranked_risk_scores = zip(*m_combo)
            m_names = ", ".join(m_names)
            score = calc_score(sum(ranked_risk_scores))
            lb_pct = leaderboard_percentile(score)
            scores.append((score, lb_pct, m_names))

    scores = sorted(scores)
    fn = f"{CSV_PATH}/scores.csv"

    # Save scores to CSV
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score", "leaderboard_percentile", "models"])
        w.writerows(scores)
        print(f"wrote {fn}")

    # Print scores for each model combination
    for score, lb_pct, m_names in scores:
        print(f"{score:.4f} {lb_pct:.4f} {m_names}")

    # Optional validation against competition metric implementation
    if PROVE_MY_SCORE_IS_SAME:
        print("-" * 79)
        import metric

        solution = train[["race_group", "efs", "efs_time"]].reset_index()
        row_id_column_name = "ID"
        scores = []

        for r in range(1, len(y_pred_oofs) + 1):
            for m_combo in combinations(y_pred_oofs.items(), r):
                m_names, ranked_risk_scores = zip(*m_combo)
                m_names = "-".join(m_names)
                submission = pd.DataFrame(
                    sum(ranked_risk_scores), index=train.index, columns=["prediction"]
                ).reset_index()
                score = metric.score(
                    solution.copy(), submission.copy(), row_id_column_name
                )
                lb_pct = leaderboard_percentile(score)
                scores.append((score, lb_pct, m_names))

        scores = sorted(scores)

        for score, lb_pct, m_names in scores:
            print(f"{score:.4f} {lb_pct:.4f} {m_names}")


def predict():
    """Generate predictions for submission"""
    print("\npredict")
    fn = f"{CSV_PATH}/scores.csv"
    row_models_len_max = 0

    # Load best model combination based on CV scores
    with open(fn, newline="") as f:
        for row in csv.DictReader(f):
            values = lambda: (
                float(row["score"]),
                float(row["leaderboard_percentile"]),
                row["models"].split("-"),
            )

            if not SUBMIT_FULL_ENSEMBLE:
                cv_score, lb_pct, m_names = values()
                continue

            if row_models_len_max < len(row["models"]):
                cv_score, lb_pct, m_names = values()
                m_names = row["models"].split("-")

    msg = "full ensemble" if SUBMIT_FULL_ENSEMBLE else "ensemble with best CV score"
    print(f"using {msg}: {cv_score:.4f} {lb_pct:.4f} {m_names}")
    y_preds = defaultdict(lambda: np.zeros(len(test)))

    # Generate predictions for each selected model
    for pipe in PIPES:
        if pipe["m"] in m_names:
            m_name = pipe["m"]
            X = get_X("test", pipe["X"])
            for fn in glob(f"{MODEL_PATH}/{NAME}-{m_name}-*.joblib"):
                m = joblib.load(fn)
                print(f"read {fn}")
                y_preds[m_name] += m.predict(X)

    kfold = create_kfold()

    # Average predictions across folds and convert to ranks
    for m_name in y_preds.keys():
        y_preds[m_name] /= kfold.get_n_splits()
        y_preds[m_name] = rankdata(y_preds[m_name])

    # Create final submission
    y_pred = sum(y_preds.values())
    s = pd.DataFrame(y_pred, index=test.index, columns=["prediction"])
    s.to_csv("submission.csv")
    print("wrote submission.csv")


if __name__ == "__main__":
    # Local execution path
    if not RUNNING_ON_KAGGLE:
        if sys.argv[1:] and sys.argv[1] == "cv_score":
            cv_score()
            sys.exit()

        if sys.argv[1:] and sys.argv[1] == "predict":
            predict()
            sys.exit()

        pipe_i = None
        if sys.argv[1:]:
            for i, pipe in enumerate(PIPES):
                if pipe["m"] == sys.argv[1]:
                    pipe_i = i
                    break
        if pipe_i is not None:
            PIPES = [PIPES[pipe_i]]

        # Clean previous run artifacts
        for fn in glob(f"{CSV_PATH}/*.csv"):
            os.remove(fn)

        for fn in glob(f"{MODEL_PATH}/*.joblib"):
            os.remove(fn)

        # Run full pipeline
        fit()
        cv_score()
        predict()
        sys.exit()

    # Kaggle execution path
    if INCLUDE_FIT_ON_KAGGLE:
        fit()
    cv_score()
    predict()

    # Cleanup temporary files
    shutil.rmtree(CSV_PATH)

    if INCLUDE_FIT_ON_KAGGLE:
        shutil.rmtree("catboost_info")
        shutil.rmtree(MODEL_PATH)
