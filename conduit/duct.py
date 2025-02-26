import os
import sys
import csv
from glob import glob
from functools import cache
from collections import defaultdict
from itertools import combinations, starmap
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
import joblib

from conduit.xgb import xgb_factory
from conduit.cb import cb_factory
from conduit.lgb import Lgb


def running_on_kaggle():
    try:
        get_ipython
    except NameError:
        return False
    return True


class Duct:
    def __init__(
        self,
        name,
        pipes,
        id_col,
        target_cols,
        submit_full_ensemble,
        include_fit_on_kaggle,
        data_dir,
        lb_dir,
    ):
        self.name = name
        self.pipes = pipes
        self.id_col = id_col
        self.target_cols = target_cols
        self.submit_full_ensemble = submit_full_ensemble
        self.include_fit_on_kaggle = include_fit_on_kaggle
        self.data_path = f"../input/{data_dir}"
        self.lb_path = f"../input/{lb_dir}"
        self.csv_path = f"./csv/{name}"
        os.makedirs(self.csv_path, exist_ok=True)
        self.running_on_kaggle = running_on_kaggle()

        if self.running_on_kaggle and self.include_fit_on_kaggle:
            self.model_path = f"./models/{self.name}"
        else:
            self.model_path = f"../input/{self.name}"

        if not self.running_on_kaggle or (
            self.running_on_kaggle and self.include_fit_on_kaggle
        ):
            os.makedirs(self.model_path, exist_ok=True)

        self.train = self.read_csv("train")
        self.test = self.read_csv("test")
        self.kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    def read_csv(self, train_or_test):
        path = f"{self.data_path}/{train_or_test}.csv"
        df = pd.read_csv(path).set_index(self.id_col)
        df.index = df.index.astype("int32")
        print(f"read {path} {df.shape}")
        return df

    def get_y(self, encoding_type, X_train, i_fold):
        if len(target_cols) > 1:
            raise

        return self.train[self.target_cols[0]]

    @cache
    def get_X(self, train_or_test, encoding_type):
        X = pd.concat([self.train.drop(columns=["efs", "efs_time"]), self.test])
        Xf = X.select_dtypes(["int", "float"]).astype("float32")
        Xo = X.select_dtypes("object").astype("category")

        if encoding_type == "raw":
            for col in Xo:
                if Xo[col].isna().any():
                    Xo[col] = Xo[col].cat.add_categories("Missing").fillna("Missing")

        elif encoding_type == "label":
            for col in Xo:
                Xo[col], _ = Xo[col].factorize(use_na_sentinel=False)
                Xo[col] = Xo[col].astype("int32").astype("category")

        elif encoding_type == "onehot":
            Xo = pd.get_dummies(Xo, drop_first=True, dtype="int32")
            Xo.columns = Xo.columns.str.replace(r"[\[\]<]", "_", regex=True)

        else:
            raise

        X = pd.concat([Xf, Xo], axis=1)

        if encoding_type != "onehot":
            diff = set(self.test.columns) - set(X.columns)
            if len(diff) != 0:
                print(
                    "\n\n",
                    train_or_test,
                    encoding_type,
                    "\n",
                    set(Xf.columns),
                    "\n\n",
                    set(Xo.columns),
                    "\n\n",
                    diff,
                )
                raise

        if train_or_test == "train":
            return X[: len(self.train)]

        if train_or_test == "test":
            return X[len(self.train) :]

        raise

    def fit_fold_model(self, p_name, pipe, fold_n, i_fold, i_oof):
        m_name, X_name, y_name = [pipe.get(k) for k in ["m", "X", "y"]]
        X = self.get_X("train", X_name)
        y = self.get_y(y_name, X, i_fold)
        cat_features = X.select_dtypes("category").columns.to_list()

        if m_name == "xgb":
            factory = xgb_factory(enable_categorical=False, y_name=y_name)()
        if m_name == "xgbcat":
            factory = xgb_factory(enable_categorical=True, y_name=y_name)()
        elif m_name == "cb":
            factory = cb_factory(y_name)(cat_features=cat_features)
        elif m_name == "lgb":
            factory = Lgb(cat_features=cat_features)

        m = factory.make(self.running_on_kaggle)

        fit_kwargs = dict(
            eval_set=[
                # (X.iloc[i_fold], y.iloc[i_fold]),
                (X.iloc[i_oof], y.iloc[i_oof])
            ],
            **factory.fit_kwargs,
        )

        if fold_n == 0:
            msg = [f"{p_name}.fit"]

            for k, v in fit_kwargs.items():
                msg.append(f"  {k}={str(v)[:60]}")

            print("\n".join(msg))

        m.fit(X.iloc[i_fold], y.iloc[i_fold], **fit_kwargs)

        filename = f"{self.model_path}/{self.name}-{p_name}-{fold_n}.joblib"
        joblib.dump(m, filename)
        print(f"wrote {filename}")

    def fit(self):
        print("\nfit")
        args = []

        for fold_n, (i_fold, i_oof) in enumerate(self.kfold.split(self.train)):
            for p_name, pipe in self.pipes.items():
                if len(self.pipes) == 1 or not glob(
                    f"{self.model_path}/{self.name}-{p_name}-*.joblib"
                ):
                    args.append((p_name, pipe, fold_n, i_fold, i_oof))

        if self.running_on_kaggle:
            list(starmap(self.fit_fold_model, args))
        else:
            from multiprocessing import Pool

            with Pool(
                min(os.cpu_count(), self.kfold.get_n_splits() * len(args))
            ) as pool:
                pool.starmap(self.fit_fold_model, args)

    def calc_score(self, y_pred_oof):
        raise NotImplementedError

    def cv_score(self):
        print("\nCV score")
        y_pred_oofs = defaultdict(lambda: np.zeros(len(self.train)))

        for fold_n, (i_fold, i_oof) in enumerate(self.kfold.split(self.train)):
            for p_name, pipe in self.pipes.items():
                m_name = pipe["m"]
                X = self.get_X("train", pipe["X"])
                for fn in glob(
                    f"{self.model_path}/{self.name}-{p_name}-{fold_n}.joblib"
                ):
                    m = joblib.load(fn)
                    print(f"read {fn}")
                    y_pred_oofs[p_name][i_oof] = m.predict(X.iloc[i_oof])

        for p_name, raw_risk_score in y_pred_oofs.items():
            y_pred_oofs[p_name] = rankdata(raw_risk_score)
            oof_fn = f"{self.csv_path}/{self.name}-{p_name}-oof.csv"
            pd.DataFrame(y_pred_oofs[p_name], index=self.train.index).to_csv(oof_fn)
            print(f"wrote {oof_fn}")

        fn = sorted(glob(f"{self.lb_path}/*.csv"))[-1]
        lb = pd.read_csv(fn)
        print(f"read {fn}")
        lb = np.sort(lb["Score"])

        def leaderboard_percentile(score):
            return np.searchsorted(lb, score) / len(lb)

        scores = []

        for r in range(1, len(y_pred_oofs) + 1):
            for m_combo in combinations(y_pred_oofs.items(), r):
                p_names, ranked_risk_scores = zip(*m_combo)
                p_names = ", ".join(p_names)
                score = self.calc_score(sum(ranked_risk_scores))
                lb_pct = leaderboard_percentile(score)
                scores.append((score, lb_pct, p_names))

        scores = sorted(scores)
        fn = f"{self.csv_path}/scores.csv"

        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["score", "leaderboard_percentile", "models"])
            w.writerows(scores)
            print(f"wrote {fn}")

        for score, lb_pct, p_names in scores:
            print(f"{score:.4f} {lb_pct:.4f} {p_names}")

    def optimize_weights(self):
        """Optimize ensemble weights per race group using Optuna"""
        print("\nOptimizing weights per race group")
        import json
        import optuna

        # Load OOF predictions from CSV files
        oof_preds = {}
        for p_name in self.pipes.keys():
            oof_fn = f"{self.csv_path}/{self.name}-{p_name}-oof.csv"
            if os.path.exists(oof_fn):
                oof_preds[p_name] = pd.read_csv(oof_fn, index_col=0).iloc[:, 0].values
                print(f"read {oof_fn}")

        # Get race groups from train data
        race_groups = self.train["race_group"].values
        unique_races = np.unique(race_groups)

        # Define the objective function for Optuna
        def objective(trial):
            # Initialize weights dictionary for all races
            weights_per_race = {}

            for race in unique_races:
                # Initialize weights for this race
                race_weights = {}
                for p_name in oof_preds.keys():
                    # Use different weight for each model per race
                    race_weights[p_name] = trial.suggest_float(
                        f"{race}_{p_name}", 0.0, 1.0
                    )

                # Normalize weights to sum to 1
                total = sum(race_weights.values())
                if total > 0:
                    race_weights = {k: v / total for k, v in race_weights.items()}

                weights_per_race[race] = race_weights

            # Apply race-specific weights to predictions
            weighted_preds = np.zeros(len(self.train))

            for i, race in enumerate(race_groups):
                race_specific_weights = weights_per_race[race]

                # Apply weights for this specific race
                for p_name, weight in race_specific_weights.items():
                    weighted_preds[i] += weight * oof_preds[p_name][i]

            # Calculate score
            score = self.calc_score(weighted_preds)
            return score

        # Create and run the study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

        # Get best weights
        best_weights = {}
        for race in unique_races:
            race_weights = {}
            for p_name in oof_preds.keys():
                race_weights[p_name] = study.best_params[f"{race}_{p_name}"]

            # Normalize weights
            total = sum(race_weights.values())
            if total > 0:
                race_weights = {k: v / total for k, v in race_weights.items()}

            best_weights[race] = race_weights

        # Print weights
        print(f"\nOptimal weights per race ({study.best_value:.4f} CV score):")
        for race, weights in best_weights.items():
            print(f"\n{race}:")
            for model, weight in sorted(weights.items(), key=lambda x: -x[1]):
                if weight > 0.01:  # Only show significant weights
                    print(f"  {model}: {weight:.4f}")

        # Save weights to JSON
        weights_fn = f"{self.csv_path}/{self.name}-optimal-weights.json"
        with open(weights_fn, "w") as f:
            json.dump(best_weights, f, indent=2)
        print(f"\nwrote {weights_fn}")

        # Return best score and weights
        return study.best_value, best_weights

    def weighted_predict(self):
        """Predict using race-specific weights from optimize_weights"""
        print("\nPredicting with race-specific weights")

        # Load optimal weights
        weights_fn = f"{self.csv_path}/{self.name}-optimal-weights.json"
        if not os.path.exists(weights_fn):
            print(f"Weights file {weights_fn} not found. Run optimize_weights first.")
            return

        import json

        with open(weights_fn, "r") as f:
            weights_per_race = json.load(f)

        # Get test data race groups
        X_test = self.get_X("test", "label")  # Use label encoding for all models
        test_with_pred = self.test.copy()

        # Initialize prediction array
        y_pred = np.zeros(len(test_with_pred))

        # Load and predict with each model
        model_preds = {}
        for p_name, pipe in self.pipes.items():
            model_preds[p_name] = np.zeros(len(test_with_pred))
            for fold_n in range(self.kfold.get_n_splits()):
                fn = f"{self.model_path}/{self.name}-{p_name}-{fold_n}.joblib"
                if os.path.exists(fn):
                    m = joblib.load(fn)
                    print(f"read {fn}")
                    model_preds[p_name] += m.predict(X_test) / self.kfold.get_n_splits()

        # Apply race-specific weights
        for race, weights in weights_per_race.items():
            # Create mask for this race
            race_mask = test_with_pred["race_group"] == race

            # Apply weights for this race
            for p_name, weight in weights.items():
                if p_name in model_preds:
                    y_pred[race_mask] += model_preds[p_name][race_mask] * weight

        # Create submission
        s = pd.DataFrame(y_pred, index=self.test.index, columns=["prediction"])
        s.to_csv("submission.csv")
        print("wrote submission.csv")

    def predict(self):
        print("\npredict")
        fn = f"{self.csv_path}/scores.csv"
        row_models_len_max = 0

        with open(fn, newline="") as f:
            for row in csv.DictReader(f):
                values = lambda: (
                    float(row["score"]),
                    float(row["leaderboard_percentile"]),
                    row["models"].split(", "),
                )

                if not self.submit_full_ensemble:
                    cv_score, lb_pct, p_names = values()
                    continue

                if row_models_len_max < len(row["models"]):
                    cv_score, lb_pct, p_names = values()
                    p_names = row["models"].split("-")

        msg = (
            "full ensemble"
            if self.submit_full_ensemble
            else "ensemble with best CV score"
        )
        print(f"using {msg}: {cv_score:.4f} {lb_pct:.4f} {p_names}")
        y_preds = defaultdict(lambda: np.zeros(len(self.test)))

        for p_name in p_names:
            pipe = self.pipes[p_name]
            X = self.get_X("test", pipe["X"])
            for fn in glob(f"{self.model_path}/{self.name}-{p_name}-*.joblib"):
                m = joblib.load(fn)
                print(f"read {fn}")
                y_preds[p_name] += m.predict(X)

        for p_name in y_preds.keys():
            y_preds[p_name] = rankdata(y_preds[p_name])

        y_pred = sum(y_preds.values())
        s = pd.DataFrame(y_pred, index=self.test.index, columns=["prediction"])
        s.to_csv("submission.csv")
        print("wrote submission.csv")
