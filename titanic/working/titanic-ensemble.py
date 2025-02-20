#!/usr/bin/env python

# Standard library imports and configuration
import warnings
warnings.simplefilter("ignore")
import os
import sys
import csv
import shutil
from glob import glob
from functools import cache
from collections import defaultdict
from itertools import combinations, starmap
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import joblib

# Unique identifier for this experiment
NAME = "titanic-ensemble"

# Configuration flags
SUBMIT_FULL_ENSEMBLE = False  # If True: use all models, if False: use best CV ensemble
INCLUDE_FIT_ON_KAGGLE = False  # If True: fit on Kaggle, if False: use saved models

# Path configuration
C_PATH = "../input/titanic"
CSV_PATH = f"./csv/{NAME}"
os.makedirs(CSV_PATH, exist_ok=True)
MODEL_PATH = "../input/titanic-models"

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

# Model hyperparameters from ensemble.py
kwargs_xgb = dict(
    max_depth=3,
    colsample_bytree=0.5,
    subsample=0.8,
    n_estimators=2000,
    learning_rate=0.02,
    min_child_weight=80,
    enable_categorical=True,
    objective="binary:logistic",
    eval_metric="logloss",
    verbosity=0,
)

kwargs_cb = dict(
    learning_rate=0.1,
    grow_policy="Lossguide",
    objective="Logloss",
    silent=True,
)

kwargs_lgb = dict(
    max_depth=3,
    colsample_bytree=0.4,
    n_estimators=2500,
    learning_rate=0.02,
    num_leaves=8,
    objective="binary",
    verbose=-1,
    verbosity=-1,
)

# Pipeline configurations combining approaches from all scripts
PIPES = [
    {
        "m": "xgb",
        "X": "label",
        "y": "raw",
        "kwargs": kwargs_xgb,
    },
    {
        "m": "lgb",
        "X": "label",
        "y": "raw",
        "kwargs": kwargs_lgb,
    },
    {
        "m": "cb",
        "X": "label",
        "y": "raw",
        "kwargs": kwargs_cb,
    },
]

# Configure GPU usage when running on Kaggle
if RUNNING_ON_KAGGLE:
    for pipe in PIPES:
        if pipe["m"] == "xgb":
            pipe["kwargs"] = dict(device="cuda", **pipe["kwargs"])
        elif pipe["m"] == "cb":
            pipe["kwargs"] = dict(task_type="GPU", **pipe["kwargs"])
        elif pipe["m"] == "lgb":
            pipe["kwargs"] = dict(device="gpu", **pipe["kwargs"])

def read_csv(path):
    df = pd.read_csv(path).set_index("PassengerId")
    df.index = df.index.astype("int32")
    print(f"read {path}")
    return df

# Load train and test data
train = read_csv(f"{C_PATH}/train.csv")
test = read_csv(f"{C_PATH}/test.csv")

def extract_title(X):
    title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Mrs",
        "Dona": "Mrs",
        "the Countess": "Mrs",
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Dr": "Dr",
        "Rev": "Rev",
        "Sir": "Sir",
        "Don": "Sir",
        "Jonkheer": "Sir",
    }
    return title.map(lambda x: title_mapping.get(x, x))

@cache
def get_X(train_or_test, name):
    # Combine train and test for consistent preprocessing
    X = pd.concat([train.drop(columns=["Survived"] if "Survived" in train else []), test])
    
    # Feature engineering from ensemble.py
    X["Title"] = extract_title(X)
    X = X.drop(columns=["Name", "Ticket", "Cabin"])
    
    # Split features by type
    Xi = X.select_dtypes("int").astype("int32")
    Xf = X.select_dtypes("float").astype("float32")
    Xc = X.select_dtypes("object")
    
    # Handle categorical features based on encoding type
    if name == "label":
        for col in Xc:
            Xc[col], _ = Xc[col].factorize(use_na_sentinel=False)
            Xc[col] = Xc[col].astype("int32").astype("category")
    else:
        raise ValueError(f"Unknown encoding type: {name}")
    
    # Combine all feature types
    X = pd.concat([Xi, Xf, Xc], axis=1)
    
    # Fill missing values
    for col in X:
        if X[col].dtype.name in ["int32", "float32"]:
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])
    
    if train_or_test == "train":
        return X[:len(train)]
    elif train_or_test == "test":
        return X[len(train):]
    else:
        raise ValueError(f"Unknown dataset type: {train_or_test}")

@cache
def get_y(name):
    if name == "raw":
        return train["Survived"]
    raise ValueError(f"Unknown target transformation: {name}")

def create_kfold():
    return KFold(n_splits=5, shuffle=True, random_state=42)

def fit_fold_model(X, y, fold_n, i_fold, i_oof, m_name, m):
    fit_kwargs = {}
    
    # Specific eval sets for different models
    if m_name == "lgb":
        fit_kwargs["eval_set"] = [(X.iloc[i_oof], y.iloc[i_oof])]
    elif m_name == "cb":
        fit_kwargs["eval_set"] = [(X.iloc[i_oof], y.iloc[i_oof])]
        fit_kwargs["cat_features"] = X.select_dtypes("category").columns.tolist()
    elif m_name == "xgb":
        fit_kwargs["eval_set"] = [(X.iloc[i_oof], y.iloc[i_oof])]
        fit_kwargs["verbose"] = False

    if fold_n == 0:
        print(f"\n{m_name} fitting with:")
        for k, v in fit_kwargs.items():
            print(f"  {k}={str(v)[:60]}")

    m.fit(X.iloc[i_fold], y.iloc[i_fold], **fit_kwargs)
    
    filename = f"{MODEL_PATH}/{NAME}-{m_name}-{fold_n}.joblib"
    joblib.dump(m, filename)
    print(f"wrote {filename}")

def fit():
    print("\nfit")
    kfold = create_kfold()
    
    m_constructors = {
        "xgb": xgb.XGBClassifier,
        "lgb": lgb.LGBMClassifier,
        "cb": cb.CatBoostClassifier,
    }
    
    models = []
    for pipe in PIPES:
        X = get_X("train", pipe["X"])
        m_name = pipe["m"]
        Model = m_constructors[m_name]
        
        models.append((
            m_name,
            Model(**pipe["kwargs"]),
            X,
            get_y(pipe["y"]),
        ))
        
        print(f"\n{m_name} configuration:")
        for k, v in pipe["kwargs"].items():
            print(f"  {k}={v}")
    
    args = []
    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(train)):
        for m_name, m, X, y in models:
            args.append((X, y, fold_n, i_fold, i_oof, m_name, m))
    
    if RUNNING_ON_KAGGLE:
        list(starmap(fit_fold_model, args))
    else:
        from multiprocessing import Pool
        with Pool(min(os.cpu_count(), kfold.get_n_splits() * len(models))) as pool:
            pool.starmap(fit_fold_model, args)

def cv_score():
    print("\nCV score")
    kfold = create_kfold()
    y_pred_oofs = defaultdict(lambda: np.zeros(len(train)))
    
    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(train)):
        for pipe in PIPES:
            m_name = pipe["m"]
            X = get_X("train", pipe["X"])
            for fn in glob(f"{MODEL_PATH}/{NAME}-{m_name}-{fold_n}.joblib"):
                m = joblib.load(fn)
                print(f"read {fn}")
                y_pred_oofs[m_name][i_oof] = m.predict(X.iloc[i_oof])
    
    scores = []
    for r in range(1, len(y_pred_oofs) + 1):
        for m_combo in combinations(y_pred_oofs.items(), r):
            m_names, predictions = zip(*m_combo)
            m_names = "-".join(m_names)
            
            ensemble_pred = np.round(sum(predictions) / len(predictions))
            score = accuracy_score(train["Survived"], ensemble_pred)
            scores.append((score, m_names))
    
    scores = sorted(scores)
    fn = f"{CSV_PATH}/scores.csv"
    
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score", "models"])
        w.writerows(scores)
        print(f"wrote {fn}")
    
    for score, m_names in scores:
        print(f"{score:.4f} {m_names}")

def predict():
    print("\npredict")
    fn = f"{CSV_PATH}/scores.csv"
    row_models_len_max = 0
    
    with open(fn, newline="") as f:
        for row in csv.DictReader(f):
            values = lambda: (float(row["score"]), row["models"].split("-"))
            
            if not SUBMIT_FULL_ENSEMBLE:
                cv_score, m_names = values()
                continue
            
            if row_models_len_max < len(row["models"]):
                cv_score, m_names = values()
                row_models_len_max = len(row["models"])
    
    msg = "full ensemble" if SUBMIT_FULL_ENSEMBLE else "ensemble with best CV score"
    print(f"using {msg}: {cv_score:.4f} {'-'.join(m_names)}")
    
    y_preds = defaultdict(lambda: np.zeros(len(test)))
    
    for pipe in PIPES:
        if pipe["m"] in m_names:
            m_name = pipe["m"]
            X = get_X("test", pipe["X"])
            for fn in glob(f"{MODEL_PATH}/{NAME}-{m_name}-*.joblib"):
                m = joblib.load(fn)
                print(f"read {fn}")
                y_preds[m_name] += m.predict_proba(X)[:, 1]
    
    kfold = create_kfold()
    for m_name in y_preds:
        y_preds[m_name] /= kfold.get_n_splits()
    
    y_pred = np.round(sum(y_preds.values()) / len(y_preds))
    
    submission = pd.DataFrame(y_pred, index=test.index, columns=["Survived"])
    submission = submission.astype(int)
    submission.to_csv("submission.csv")
    print("wrote submission.csv")

if __name__ == "__main__":
    if not RUNNING_ON_KAGGLE:
        if sys.argv[1:] and sys.argv[1] == "cv_score":
            cv_score()
            sys.exit()
        
        if sys.argv[1:] and sys.argv[1] == "predict":
            predict()
            sys.exit()
        
        for fn in glob(f"{CSV_PATH}/*.csv"):
            os.remove(fn)
        
        for fn in glob(f"{MODEL_PATH}/*.joblib"):
            os.remove(fn)
        
        fit()
        cv_score()
        predict()
        sys.exit()
    
    if INCLUDE_FIT_ON_KAGGLE:
        fit()
    cv_score()
    predict()
    
    shutil.rmtree(CSV_PATH)
    if INCLUDE_FIT_ON_KAGGLE:
        shutil.rmtree("catboost_info")
        shutil.rmtree(MODEL_PATH)