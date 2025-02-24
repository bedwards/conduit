from lightgbm import LGBMRegressor


class Lgb:
    def __init__(self, cat_features):
        self.kwargs = dict(
            categorical_feature=cat_features,
            max_depth=3,
            colsample_bytree=0.4,
            n_estimators=2500,
            learning_rate=0.02,
            num_leaves=8,
            verbose=-1,
            verbosity=-1,
        )
        self.fit_kwargs = {}

    def make(self, running_on_kaggle):
        if running_on_kaggle:
            self.kwargs = dict(device="gpu", **self.kwargs)
        return LGBMRegressor(**self.kwargs)


# https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html
# https://lightgbm.readthedocs.io/en/latest/Parameters.html
# objective="regression", metric="l2" (i.e. mean squared error)
# fit(eval_metric="l2")
# https://lightgbm.readthedocs.io/en/latest/Parameters.html#categorical_feature
