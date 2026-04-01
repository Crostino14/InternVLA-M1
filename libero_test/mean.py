
import numpy as np

sr = np.array([97.2,93.5,98,85.3,98.8,94.7,99.7,93.4,99.8,99])
mean = np.mean(sr)
std = np.std(sr, ddof=1)
print(f"{mean:.1f}% ± {std:.1f}%".replace(".", ","))
"""

import math

def mean_sr_across_variations(variation_sr: list[float]) -> tuple[float, float]:
    n = len(variation_sr)
    mean = sum(variation_sr) / n
    var  = sum((sr - mean) ** 2 for sr in variation_sr) / max(n - 1, 1)
    std  = math.sqrt(var)
    return mean, std


# Esempi per i tuoi modelli
models = {
    "OpenVLA-OFT":   [0.0,1.3,0.7,0.7,0.0,0.0],
    "TinyVLA":       [0.0,73.3,84.7,18.7,34.0,0.0],
    "InternVLA-M1":  [67.3,12.7,94.7,50.0,64.7,30.7],
}

for name, srs in models.items():
    mean, std = mean_sr_across_variations(srs)
    print(f"{mean:.1f}% ± {std:.1f}%".replace(".", ","))
"""