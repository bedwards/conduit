from catboost import CatBoostRegressor


class CbBase:
    def __init__(self, cat_features):
        self.kwargs = dict(
            learning_rate=0.1,
            grow_policy="Lossguide",
            cat_features=cat_features,
            silent=True,
        )
        self.fit_kwargs = {}

    def make(self, running_on_kaggle):
        return CatBoostRegressor(**self.kwargs)


class CbRmse(CbBase):
    def make(self, running_on_kaggle):
        if running_on_kaggle:
            self.kwargs = dict(task_type="GPU", **self.kwargs)
        return super().make(running_on_kaggle)


class CbCox(CbBase):
    def __init__(self, cat_features):
        super().__init__(cat_features)
        # https://catboost.ai/docs/en/concepts/python-reference_catboostregressor
        # https://github.com/catboost/tutorials/blob/master/regression/survival.ipynb
        # loss_function='Cox', eval_metric='Cox'
        # loss_function='SurvivalAft:dist=Normal', eval_metric='SurvivalAft'
        # loss_function='SurvivalAft:dist=Logistic;scale=1.2', eval_metric='SurvivalAft'
        # loss_function='SurvivalAft:dist=Extreme;scale=2', eval_metric='SurvivalAft'
        self.kwargs = dict(
            loss_function="Cox", iterations=400, use_best_model=False, **self.kwargs
        )


def cb_factory(y_name):
    return CbCox if y_name == "cox" else CbRmse
