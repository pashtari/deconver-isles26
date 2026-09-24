"""Ignite metric handlers computing per-case Dice and HD95 from full-size predictions."""

from typing import Callable

from monai.handlers.ignite_metric import IgniteMetricHandler
from monai.metrics import DiceMetric, HausdorffDistanceMetric


class MeanDice(IgniteMetricHandler):
    """Dice score averaged over cases; per-case values are kept for ``MetricsSaver``."""

    def __init__(self, output_transform: Callable = lambda x: x, save_details: bool = True, **kwargs):
        super().__init__(
            metric_fn=DiceMetric(**kwargs), output_transform=output_transform, save_details=save_details
        )


class MeanHausdorffDistance(IgniteMetricHandler):
    """Hausdorff distance averaged over cases; per-case values are kept for ``MetricsSaver``."""

    def __init__(self, output_transform: Callable = lambda x: x, save_details: bool = True, **kwargs):
        super().__init__(
            metric_fn=HausdorffDistanceMetric(**kwargs),
            output_transform=output_transform,
            save_details=save_details,
        )
