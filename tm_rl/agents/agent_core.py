"""Minimal Trackmania RL agent loop helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from agents.controls import ControlConfig, SteeringSmoother, decide_controls
from env import compute_reward, is_done, obs_vec
from utils.ghost import GhostData, LookaheadConfig


@dataclass
class AgentConfig:
    """Configuration for the minimal agent loop."""

    device: str = "cpu"
    control: ControlConfig = field(default_factory=ControlConfig)
    lookahead: LookaheadConfig = field(default_factory=LookaheadConfig)


class TrackmaniaAgent:
    """Minimal executor for driving the car from model outputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        ghost: GhostData,
        cfg: Optional[AgentConfig] = None,
    ) -> None:
        self.model = model
        self.ghost = ghost
        self.cfg = cfg or AgentConfig()
        self.smoother = SteeringSmoother(self.cfg.control.steer_smooth_alpha)
        self.prev_state: Optional[Dict] = None

    def reset(self) -> None:
        self.prev_state = None
        self.smoother.reset()

    def act(
        self,
        state: Dict,
        race_time_ms: int,
    ) -> Tuple[Dict[str, bool], np.ndarray, Optional[Dict[str, float]]]:
        """Return control inputs, raw action, and optional transition info."""
        obs, _ = obs_vec(state, self.ghost, self.cfg.lookahead)

        if not hasattr(self, "hist"):
            self.hist = []

        self.hist.append(obs)
        if len(self.hist) > self.cfg.lookahead.history:
            self.hist.pop(0)

        if len(self.hist) < self.cfg.lookahead.history:
            return {}, np.zeros(3), None  # ждём накопления истории

        obs_stack = np.concatenate(self.hist, axis=0)
        obs_t = torch.tensor(
            obs_stack,
            dtype=torch.float32,
            device=self.cfg.device,
        ).unsqueeze(0)


        with torch.no_grad():
            mu, std, value = self.model(obs_t)
            dist = torch.distributions.Normal(mu, std)
            action_t = dist.sample()
            logprob_t = dist.log_prob(action_t).sum(dim=1)

        action = action_t.squeeze(0).cpu().numpy()

        accel_on, brake_on, steer_int, left, right = decide_controls(
            steer_raw=float(action[0]),
            throttle_prob=float(action[1]),
            brake_prob=float(action[2]),
            speed_norm=float(state.get("speed_norm", 0.0)),
            race_time_ms=race_time_ms,
            cfg=self.cfg.control,
            smoother=self.smoother,
        )

        controls = {
            "accelerate": accel_on,
            "brake": brake_on,
            "left": left,
            "right": right,
            "steer": steer_int,   # <-- было steer_int
        }

        transition: Optional[Dict[str, float]] = None
        if self.prev_state is not None:
            reward = compute_reward(self.prev_state, state, action)
            done = is_done(self.prev_state, state)
            transition = {
                "reward": float(reward),
                "done": float(done),
                "value": float(value.item()),
                "logprob": float(logprob_t.item()),
            }

        self.prev_state = state
        return controls, action, transition
