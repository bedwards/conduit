from lifelines import KaplanMeierFitter, NelsonAalenFitter
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, List


class SurvivalTransformer:
    """Base class for survival target transformations"""

    def fit_transform(
        self, time: np.ndarray, event: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Transform survival data into model target"""
        raise NotImplementedError

    def transform(self, time: np.ndarray) -> np.ndarray:
        """Transform new data using fitted parameters"""
        raise NotImplementedError


class KMFTransformer(SurvivalTransformer):
    """KaplanMeier survival probability transform"""

    def __init__(self):
        self.kmf = KaplanMeierFitter()

    def fit_transform(
        self, time: np.ndarray, event: np.ndarray, **kwargs
    ) -> np.ndarray:
        self.kmf.fit(time, event)
        return self.kmf.survival_function_at_times(time).values

    def transform(self, time: np.ndarray) -> np.ndarray:
        return self.kmf.survival_function_at_times(time).values


class NAFTransformer(SurvivalTransformer):
    """Nelson-Aalen cumulative hazard transform"""

    def __init__(self):
        self.naf = NelsonAalenFitter()

    def fit_transform(
        self, time: np.ndarray, event: np.ndarray, **kwargs
    ) -> np.ndarray:
        self.naf.fit(time, event)
        return -self.naf.cumulative_hazard_at_times(time).values

    def transform(self, time: np.ndarray) -> np.ndarray:
        return -self.naf.cumulative_hazard_at_times(time).values


class StratifiedKMF(SurvivalTransformer):
    """KaplanMeier with group stratification and gap adjustments"""

    def __init__(
        self,
        group_weights: Optional[Dict[str, float]] = None,
        gap_factor: float = 0.7,
        gap_mode: str = "minmax",
    ):
        self.group_weights = group_weights or {}
        self.gap_factor = gap_factor
        self.gap_mode = gap_mode
        self.kmf_by_group = {}

    def _calculate_gap(
        self, event_preds: np.ndarray, censored_preds: np.ndarray
    ) -> float:
        """Calculate gap between event and censored predictions"""
        if self.gap_mode == "median":
            return (np.median(censored_preds) - np.median(event_preds)) / 2
        elif self.gap_mode == "mean":
            return (np.mean(censored_preds) - np.mean(event_preds)) / 2
        elif self.gap_mode == "minmax":
            return (np.max(censored_preds) - np.min(event_preds)) / 2
        else:
            raise ValueError(f"Unknown gap_mode: {self.gap_mode}")

    def fit_transform(
        self, time: np.ndarray, event: np.ndarray, groups: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Transform survival data with group stratification"""
        df = pd.DataFrame(
            {
                "time": time,
                "event": event,
                "group": groups,
                "predictions": np.zeros(len(time)),
            }
        )

        for group in df["group"].unique():
            group_mask = df["group"] == group
            group_df = df[group_mask]
            weight = self.group_weights.get(group, 1.0)

            kmf = KaplanMeierFitter()
            kmf.fit(group_df["time"], group_df["event"])
            self.kmf_by_group[group] = kmf

            group_preds = kmf.survival_function_at_times(group_df["time"]).values
            group_preds = group_preds * weight

            event_mask = group_df["event"] == 1
            censored_mask = group_df["event"] == 0

            gap = self._calculate_gap(
                group_preds[event_mask], group_preds[censored_mask]
            )

            group_preds[censored_mask] -= gap * self.gap_factor
            df.loc[group_mask, "predictions"] = group_preds

        return df["predictions"].values

    def transform(self, time: np.ndarray, groups: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(
            {"time": time, "group": groups, "predictions": np.zeros(len(time))}
        )

        for group in df["group"].unique():
            if group not in self.kmf_by_group:
                continue

            group_mask = df["group"] == group
            weight = self.group_weights.get(group, 1.0)
            kmf = self.kmf_by_group[group]

            group_preds = kmf.survival_function_at_times(
                df.loc[group_mask, "time"]
            ).values
            df.loc[group_mask, "predictions"] = group_preds * weight

        return df["predictions"].values


class LogitScaledTransformer(SurvivalTransformer):
    """Logit transform with scaling"""

    def __init__(
        self, max_time: float = 80, min_time: float = -100, offset: float = 10
    ):
        self.max_time = max_time
        self.min_time = min_time
        self.offset = offset

    def _logit(self, p: np.ndarray) -> np.ndarray:
        return np.log(p) - np.log(1 - p)

    def fit_transform(
        self, time: np.ndarray, event: np.ndarray, **kwargs
    ) -> np.ndarray:
        y = time / (self.max_time - self.min_time)
        y = self._logit(y)
        return -(y + self.offset)

    def transform(self, time: np.ndarray) -> np.ndarray:
        y = time / (self.max_time - self.min_time)
        y = self._logit(y)
        return -(y + self.offset)
