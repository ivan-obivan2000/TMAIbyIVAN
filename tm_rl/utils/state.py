import numpy as np

def extract_state(s):
    return {
        "position": list(s.position),
        "velocity": list(s.velocity),
        "speed_norm": float(np.linalg.norm(s.velocity)),
        "car": {},
        "engine": {},
        "dyna": {},
    }
