from lifelines import KaplanMeierFitter, NelsonAalenFitter
import numpy as np
import pandas as pd


class SurvivalTransformer:
    """Base class for survival target transformations"""

    def fit_transform(self, time: np.ndarray, event: np.ndarray) -> np.ndarray:
        pass


class KMFTransformer(SurvivalTransformer):
    """KaplanMeier survival probability transform"""

    pass


class NAFTransformer(SurvivalTransformer):
    """Nelson-Aalen cumulative hazard transform"""

    pass


class StratifiedKMF(SurvivalTransformer):
    """KaplanMeier with group stratification"""

    pass


class LogitTransformer(SurvivalTransformer):
    """Logit transform with scaling"""

    pass
