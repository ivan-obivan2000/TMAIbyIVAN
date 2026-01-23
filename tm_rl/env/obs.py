"""Observation and action vector utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from utils.ghost import GhostData, LookaheadConfig, lookahead_target


def obs_vec(
    rec: Dict,
    ghost: GhostData,
    cfg: LookaheadConfig,
) -> Tuple[np.ndarray, float]:
    position = np.array(rec.get("position", [0, 0, 0]), dtype=np.float32)
    velocity = np.array(rec.get("velocity", [0, 0, 0]), dtype=np.float32)

    target_p, target_v, dist_6d = lookahead_target(position, velocity, ghost, cfg)

    delta_p = (target_p - position) / 50.0
    delta_v = (target_v - velocity) / 50.0

    speed = np.linalg.norm(velocity)
    v_norm = velocity / (speed + 1e-6)
    tv_norm = target_v / (np.linalg.norm(target_v) + 1e-6)
    heading_error = np.dot(v_norm, tv_norm)

    car = rec.get("car") or {}
    eng = rec.get("engine") or {}
    dyna = rec.get("dyna") or {}
    quat = dyna.get("quat") or [0, 0, 0, 1]
    ang_vel = dyna.get("angular_speed") or [0, 0, 0]
    loc_v = car.get("current_local_speed") or [0, 0, 0]

    velocity_scaled = velocity / 50.0

    feat = [
        delta_p[0],
        delta_p[1],
        delta_p[2],
        delta_v[0],
        delta_v[1],
        delta_v[2],
        velocity_scaled[0],
        velocity_scaled[1],
        velocity_scaled[2],
        rec.get("speed_norm", 0) / 100.0,
        heading_error,
        quat[0],
        quat[1],
        quat[2],
        quat[3],
        ang_vel[0],
        ang_vel[1],
        ang_vel[2],
        loc_v[0],
        loc_v[1],
        loc_v[2],
        car.get("turning_rate", 0.0),
        float(car.get("is_sliding", False)),
        eng.get("actual_rpm", 0.0) / 10000.0,
        float(eng.get("gear", 1)) / 5.0,
    ]
    return np.array(feat, dtype=np.float32), dist_6d


def action_vec(rec: Dict) -> np.ndarray:
    inputs = rec.get("inputs", {})
    steer = 0.0
    if inputs.get("left"):
        steer = -1.0
    elif inputs.get("right"):
        steer = 1.0
    throttle = 1.0 if inputs.get("accelerate") else 0.0
    brake = 1.0 if inputs.get("brake") else 0.0
    return np.array([steer, throttle, brake], dtype=np.float32)

def obs_dim(K: int, base_dim: int) -> int:
    return K * base_dim
