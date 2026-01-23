import numpy as np
import glob
import json
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from scipy.spatial import KDTree




@dataclass
class LookaheadConfig:
    minimum: int = 4
    maximum: int = 12
    factor: float = 0.1
    history: int = 20 


@dataclass
class GhostData:
    points: np.ndarray
    tree: KDTree


LOG_DIR = "tmi_logs01"
MIN_LEN = 200
N_BEST = 3

def _load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records

def load_best_episodes(
    log_dir: str,
    min_length: int,
    n_best: int,
) -> Tuple[List[List[dict]], np.ndarray]:
    files = sorted(glob.glob(f"{log_dir}/*.jsonl"))
    raw_episodes: List[List[dict]] = []

    for fp in files:
        current_ep: List[dict] = []
        last_ep_id = None
        try:
            for rec in _load_jsonl(fp):
                ep_id = rec.get("episode_id")
                if last_ep_id is None:
                    last_ep_id = ep_id
                if ep_id != last_ep_id:
                    if len(current_ep) > min_length:
                        raw_episodes.append(current_ep)
                    current_ep = []
                    last_ep_id = ep_id
                current_ep.append(rec)
            if len(current_ep) > min_length:
                raw_episodes.append(current_ep)
        except Exception:
            continue

    finished_eps = []
    for ep in raw_episodes:
        if any(r.get("finished", 0) for r in ep[-20:]):
            duration = ep[-1]["time_ms"] - ep[0]["time_ms"]
            finished_eps.append({"data": ep, "time": duration})

    if not finished_eps:
        return [], np.array([])

    finished_eps.sort(key=lambda x: x["time"])
    top_n = min(len(finished_eps), n_best)
    best_runs = finished_eps[:top_n]

    final_episodes = [item["data"] for item in best_runs]
    weights = (
        np.linspace(2.0, 1.0, len(final_episodes))
        if len(final_episodes) > 1
        else np.array([1.0])
    )
    return final_episodes, weights

def build_ghost(episodes: Iterable[Iterable[dict]]) -> GhostData:
    ghost_points = []
    for run in episodes:
        for rec in run:
            p = rec.get("position", [0, 0, 0])
            v = rec.get("velocity", [0, 0, 0])
            ghost_points.append([p[0], p[1], p[2], v[0], v[1], v[2]])

    ghost_array = np.array(ghost_points, dtype=np.float32)
    ghost_tree = KDTree(ghost_array)
    return GhostData(points=ghost_array, tree=ghost_tree)

if __name__ == "__main__":
    episodes, _ = load_best_episodes(
        log_dir=LOG_DIR,
        min_length=MIN_LEN,
        n_best=N_BEST,
    )

    ghost = build_ghost(episodes)

    np.savez(
        "ghost_data01.npz",
        points=ghost.points,
    )

    print(f"[OK] ghost saved: {ghost.points.shape}")
