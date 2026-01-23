import numpy as np

def extract_state(s):
    return {
        "position": list(s.position),
        "velocity": list(s.velocity),
        "speed_norm": float(np.linalg.norm(s.velocity)),
        "race_time_ms": int(getattr(s, "race_time", 0)),
        "race_finished": bool(
            getattr(s, "race_finished", False)
            or getattr(s, "is_finished", False)
            or getattr(s, "finished", False)
        ),
        "car": {},
        "engine": {},
        "dyna": {},
    }
