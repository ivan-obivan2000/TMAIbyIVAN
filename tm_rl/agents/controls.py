"""Action post-processing helpers for Trackmania agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ControlConfig:
    steer_smooth_alpha: float = 0.15
    max_steer: int = 65536
    force_gas_time_ms: int = 1500
    min_launch_speed_kmh: float = 5.0


class SteeringSmoother:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value = 0.0

    def reset(self) -> None:
        self.value = 0.0

    def update(self, steer_raw: float) -> float:
        self.value = (1 - self.alpha) * self.value + self.alpha * steer_raw
        return self.value


def decide_controls(
    steer_raw: float,
    throttle_prob: float,
    brake_prob: float,
    speed_norm: float,
    race_time_ms: int,
    cfg: ControlConfig,
    smoother: SteeringSmoother,
) -> Tuple[bool, bool, int, bool, bool]:
    if race_time_ms < 1500 or speed_norm < 1.0:
        accel_on = True
        brake_on = False
        
    if race_time_ms < cfg.force_gas_time_ms:
        accel_on = True
        brake_on = False
    else:
        accel_on = throttle_prob > 0.5
        brake_on = brake_prob > 0.5
        speed_kmh = speed_norm * 3.6
        if speed_kmh < cfg.min_launch_speed_kmh and not brake_on:
            accel_on = True

    steer_smoothed = smoother.update(steer_raw)
    steer_int = int(np.clip(steer_smoothed, -1.0, 1.0) * cfg.max_steer)
    left = steer_int < -2000
    right = steer_int > 2000

    return accel_on, brake_on, steer_int, left, right
