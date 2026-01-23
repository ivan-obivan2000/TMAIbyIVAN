import numpy as np
from dataclasses import dataclass
from scipy.spatial import KDTree


@dataclass
class GhostData:
    points: np.ndarray
    tree: KDTree


def load_ghost_npz(path: str) -> GhostData:
    data = np.load(path)
    points = data["points"].astype(np.float32)
    tree = KDTree(points)
    print(f"[GHOST] loaded {len(points)} points")
    return GhostData(points=points, tree=tree)
