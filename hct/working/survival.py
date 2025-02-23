from typing import Dict, Optional

from lifelines import KaplanMeierFitter, NelsonAalenFitter
import numpy as np
import pandas as pd


class SurvivalTransformer:
    """Base class for survival target transformations"""

    def fit_transform(self, time: np.ndarray, event: np.ndarray) -> np.ndarray:
        """Transform survival data into model target

        Args:
            time: Array of survival times
            event: Binary array indicating events (1) vs censoring (0)

        Returns:
            Array of transformed target values
        """
        raise NotImplementedError


class KMFTransformer(SurvivalTransformer):
    """KaplanMeier survival probability transform"""

    def fit_transform(self, time: np.ndarray, event: np.ndarray) -> np.ndarray:
        kmf = KaplanMeierFitter()
        kmf.fit(time, event)
        return kmf.survival_function_at_times(time).values


class NAFTransformer(SurvivalTransformer):
    """Nelson-Aalen cumulative hazard transform"""

    def fit_transform(self, time: np.ndarray, event: np.ndarray) -> np.ndarray:
        naf = NelsonAalenFitter()
        naf.fit(time, event)
        return -naf.cumulative_hazard_at_times(time).values


class StratifiedKMF(SurvivalTransformer):
    """KaplanMeier with group stratification and gap adjustments

    This transformer:
    1. Fits separate KM curves per group
    2. Applies group-specific weights
    3. Adjusts gaps between event/censored cases
    """

    def __init__(
        self,
        group_weights: Optional[Dict[str, float]] = None,
        gap_factor: float = 0.7,
        gap_mode: str = "median",
    ):
        """Initialize transformer

        Args:
            group_weights: Dictionary mapping group names to weights
            gap_factor: Multiplier for gap between event/censored predictions
            gap_mode: How to calculate gap ('median', 'mean', or 'minmax')
        """
        self.group_weights = group_weights or {}
        self.gap_factor = gap_factor
        self.gap_mode = gap_mode

    def _calculate_gap(
        self, event_preds: np.ndarray, censored_preds: np.ndarray
    ) -> float:
        """Calculate gap between event and censored predictions"""
        if self.gap_mode == "median":
            return (np.median(censored_preds) - np.median(event_preds)) / 2
        elif self.gap_mode == "mean":
            return (np.mean(censored_preds) - np.mean(event_preds)) / 2
        elif self.gap_mode == "minmax":
            return (censored_preds.max() - event_preds.min()) / 2
        else:
            raise ValueError(f"Unknown gap_mode: {self.gap_mode}")

    def fit_transform(
        self, time: np.ndarray, event: np.ndarray, groups: np.ndarray
    ) -> np.ndarray:
        """Transform survival data with group stratification

        Args:
            time: Array of survival times
            event: Binary array indicating events
            groups: Array of group labels

        Returns:
            Array of transformed predictions
        """
        df = pd.DataFrame({"time": time, "event": event, "group": groups})

        predictions = np.zeros(len(df))

        for group in df["group"].unique():
            # Get group mask and weight
            mask = df["group"] == group
            weight = self.group_weights.get(group, 1.0)

            # Fit KM curve for this group
            kmf = KaplanMeierFitter()
            kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"])

            # Get predictions
            group_preds = kmf.survival_function_at_times(df.loc[mask, "time"]).values

            # Calculate gap between event/censored
            event_mask = mask & (df["event"] == 1)
            censored_mask = mask & (df["event"] == 0)
            gap = self._calculate_gap(
                group_preds[event_mask], group_preds[censored_mask]
            )

            # Apply gap adjustment and weight
            predictions[mask] = group_preds * weight
            predictions[censored_mask] -= gap * self.gap_factor

        return predictions


class LogitTransformer(SurvivalTransformer):
    """Logit transform with scaling"""

    def __init__(
        self, max_time: float = 80, min_time: float = -100, offset: float = 10
    ):
        self.max_time = max_time
        self.min_time = min_time
        self.offset = offset

    def fit_transform(self, time: np.ndarray, event: np.ndarray) -> np.ndarray:
        y = time / (self.max_time - self.min_time)
        y = np.log(y) - np.log(1 - y)
        return -(y + self.offset)
