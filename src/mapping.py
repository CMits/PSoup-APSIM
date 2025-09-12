import numpy as np

def map_linear(x: float, L: float = 0.0, U: float = 50.0,
               out_min: float = 0.0, out_max: float = 2.0) -> float:
    if U <= L:
        return float("nan")
    s = (x - L) / (U - L) * (out_max - out_min) + out_min
    return float(np.clip(s, out_min, out_max))
