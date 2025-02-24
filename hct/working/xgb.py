from xgboost import XGBRegressor


class XgbBase:
    def __init__(self):
        self.kwargs = dict(
            max_depth=3,
            colsample_bytree=0.5,
            subsample=0.8,
            n_estimators=2000,
            learning_rate=0.02,
            min_child_weight=80,
            enable_categorical=True,
            verbosity=0,
        )
        self.fit_kwargs = dict(verbose=False)

    def make(self, running_on_kaggle):
        if running_on_kaggle:
            self.kwargs = dict(device="cuda", **self.kwargs)
        return XGBRegressor(**self.kwargs)


class XgbRegSquaredError(XgbBase):
    # objective="reg:squarederror"
    pass


class XgbSurvivalCox(XgbBase):
    def __init__(self):
        super().__init__()
        self.kwargs = dict(
            objective="survival:cox", eval_metric="cox-nloglik", **self.kwargs
        )

        if pipe["m"] == "lgb":
            pipe["kwargs"] = dict(device="gpu", **pipe["kwargs"])


def xgb_factory(y_name):
    return XgbSurvivalCox if y_name == "cox" else XgbRegSquaredError


# https://xgboost.readthedocs.io/en/stable/parameter.html
# https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html
# TODO: objective="survival:aft"
# TODO: eval_metric="aft-nloglik"
# TODO: eval_metric="interval-regression-accuracy"
