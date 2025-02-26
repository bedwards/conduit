from xgboost import XGBRegressor


class XgbBase:
    def __init__(self):
        self.kwargs = getattr(self, "kwargs", {})
        self.kwargs.update(
            max_depth=3,
            colsample_bytree=0.5,
            subsample=0.8,
            n_estimators=2000,
            learning_rate=0.02,
            min_child_weight=80,
            verbosity=0,
        )
        self.fit_kwargs = dict(verbose=False)

    def make(self, running_on_kaggle):
        if running_on_kaggle:
            self.kwargs = dict(device="cuda", **self.kwargs)
        return XGBRegressor(**self.kwargs)


class XgbSquaredError(XgbBase):
    # objective="reg:squarederror"
    pass


class XgbCoxBase(XgbBase):
    def __init__(self):
        XgbBase.__init__(self)
        self.kwargs = dict(
            objective="survival:cox", eval_metric="cox-nloglik", **self.kwargs
        )


class XgbCox(XgbCoxBase):
    def __init__(self):
        XgbCoxBase.__init__(self)


class XgbCatBase(XgbBase):
    def __init__(self):
        XgbBase.__init__(self)
        self.kwargs = dict(enable_categorical=True, **self.kwargs)


class XgbCatSquaredError(XgbCatBase):
    # objective="reg:squarederror"
    pass


class XgbCatCox(XgbCatBase, XgbCoxBase):
    def __init__(self):
        XgbCatBase.__init__(self)
        XgbCoxBase.__init__(self)


def xgb_factory(enable_categorical, y_name):
    if enable_categorical:
        return XgbCatCox if y_name == "cox" else XgbCatSquaredError
    return XgbCox if y_name == "cox" else XgbSquaredError


# https://xgboost.readthedocs.io/en/stable/parameter.html
# https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html
# TODO: objective="survival:aft"
# TODO: eval_metric="aft-nloglik"
# TODO: eval_metric="interval-regression-accuracy"
