import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from math import cos, sin, pi

# -------------------------
# Canonical vehicle motifs
# -------------------------
CANONICAL_VEHICLES = {
    "V1":  {"label": "Vehicle 1",  "base_sensors": 1, "base_motors": 1,
            "wiring": "single", "sign": "excitatory", "crossed": False},
    "V2a": {"label": "Vehicle 2a", "base_sensors": 2, "base_motors": 2,
            "wiring": "bilateral", "sign": "inhibitory", "crossed": False},
    "V2b": {"label": "Vehicle 2b", "base_sensors": 2, "base_motors": 2,
            "wiring": "bilateral", "sign": "inhibitory", "crossed": True},
    "V3a": {"label": "Vehicle 3a", "base_sensors": 2, "base_motors": 2,
            "wiring": "bilateral", "sign": "excitatory", "crossed": False},
    "V3b": {"label": "Vehicle 3b", "base_sensors": 2, "base_motors": 2,
            "wiring": "bilateral", "sign": "excitatory", "crossed": True},
    # Higher vehicles: keep motif loose, emphasize “more structure”
    "V4":  {"label": "Vehicle 4",  "base_sensors": 2, "base_motors": 2,
            "wiring": "nonlinear", "sign": "mixed", "crossed": True},
    "V5":  {"label": "Vehicle 5",  "base_sensors": 3, "base_motors": 2,
            "wiring": "logic", "sign": "mixed", "crossed": True},
    "V6":  {"label": "Vehicle 6",  "base_sensors": 3, "base_motors": 2,
            "wiring": "selection", "sign": "mixed", "crossed": True},
    "V7":  {"label": "Vehicle 7",  "base_sensors": 3, "base_motors": 3,
            "wiring": "concept", "sign": "mixed", "crossed": True},
    "V8":  {"label": "Vehicle 8",  "base_sensors": 4, "base_motors": 2,
            "wiring": "spatial", "sign": "mixed", "crossed": True},
    "V9":  {"label": "Vehicle 9",  "base_sensors": 4, "base_motors": 2,
            "wiring": "shape", "sign": "mixed", "crossed": True},
    "V10": {"label": "Vehicle 10", "base_sensors": 4, "base_motors": 3,
            "wiring": "associative", "sign": "mixed", "crossed": True},
    "V11": {"label": "Vehicle 11", "base_sensors": 4, "base_motors": 3,
            "wiring": "rule", "sign": "mixed", "crossed": True},
    "V12": {"label": "Vehicle 12", "base_sensors": 4, "base_motors": 3,
            "wiring": "sequence", "sign": "mixed", "crossed": True},
    "V13": {"label": "Vehicle 13", "base_sensors": 4, "base_motors": 3,
            "wiring": "predictive", "sign": "mixed", "crossed": True},
    "V14": {"label": "Vehicle 14", "base_sensors": 4, "base_motors": 3,
            "wiring": "modulated", "sign": "mixed", "crossed": True},
}

# -------------------------
# Core morphology generator
# -------------------------
def generate_morphology(vehicle_id, diversity=0.3, rng=None):
    """
    Generate a synthetic morphology variant for a canonical Braitenberg vehicle.

    Returns a dict:
      - id, label
      - n_sensors, n_motors
      - sensor_positions, motor_positions
      - wiring_matrix (n_sensors x n_motors)
      - sign_matrix (+1 excitatory, -1 inhibitory, 0 mixed)
      - crossed (bool), motif (string)
    """
    if rng is None:
        rng = np.random.default_rng()

    spec = CANONICAL_VEHICLES[vehicle_id]
    base_s = spec["base_sensors"]
    base_m = spec["base_motors"]

    # jitter counts a bit but keep near canonical
    n_sensors = int(np.clip(base_s + rng.integers(-1, 2), 1, 6))
    n_motors  = int(np.clip(base_m + rng.integers(-1, 2), 1, 6))

    # body radius
    body_radius = 0.08 + 0.12 * rng.random()

    # positions on circle with jitter
    def positions(k, phase=0.0):
        if k == 0:
            return []
        base_angles = np.linspace(0, 2*pi, k, endpoint=False) + phase
        jitter = rng.normal(scale=diversity*0.4, size=k)
        angles = base_angles + jitter
        return [(0.5 + 0.35*cos(a), 0.5 + 0.35*sin(a)) for a in angles]

    sensor_positions = positions(n_sensors, phase=0.0)
    motor_positions  = positions(n_motors,  phase=pi/2)

    # wiring matrix respecting motif
    W = np.zeros((n_sensors, n_motors), dtype=float)
    S = np.zeros_like(W)  # sign matrix

    def base_weight():
        return 0.8 + 0.6 * rng.random()

    # sign pattern
    if spec["sign"] == "excitatory":
        sign_val = +1
    elif spec["sign"] == "inhibitory":
        sign_val = -1
    else:  # mixed
        sign_val = None

    # build canonical pattern for 2x2, then generalize
    for i in range(n_sensors):
        for j in range(n_motors):
            if spec["wiring"] in ("single", "bilateral"):
                # crossed vs uncrossed mapping
                if spec["crossed"]:
                    # sensor i connects strongest to motor (n_motors-1-i) if in range
                    target_j = n_motors - 1 - i
                else:
                    target_j = i
                if 0 <= target_j < n_motors and j == target_j:
                    w = base_weight()
                else:
                    w = diversity * base_weight() * rng.random()
            else:
                # higher vehicles: denser, more varied wiring
                w = base_weight() * (0.5 + diversity * rng.random())

            # sign
            if sign_val is None:
                s = rng.choice([-1, 1])
            else:
                s = sign_val

            # add jitter
            w = w + diversity * rng.normal(scale=0.3)
            W[i, j] = w * s
            S[i, j] = s

    return {
        "id": vehicle_id,
        "label": spec["label"],
        "motif": spec["wiring"],
        "crossed": spec["crossed"],
        "n_sensors": n_sensors,
        "n_motors": n_motors,
        "sensor_positions": sensor_positions,
        "motor_positions": motor_positions,
        "wiring_matrix": W,
        "sign_matrix": S,
        "body_radius": body_radius,
    }

# -------------------------
# Simple visualization
# -------------------------
def draw_vehicle(ax, morph, title=None):
    # Set up the plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title if title else f"{morph['label']} ({morph['id']})")

    # Draw body
    body = Circle((0.5, 0.5), morph['body_radius'], color='lightgray', ec='black', lw=1.5)
    ax.add_patch(body)

    # Draw sensors
    for i, (sx, sy) in enumerate(morph['sensor_positions']):
        ax.plot(sx, sy, 'o', markersize=10, color='blue', markeredgecolor='black', zorder=5)
        ax.text(sx + 0.02, sy + 0.02, f'S{i+1}', fontsize=8, ha='left', va='bottom', zorder=5)

    # Draw motors
    for i, (mx, my) in enumerate(morph['motor_positions']):
        ax.plot(mx, my, 's', markersize=10, color='red', markeredgecolor='black', zorder=5)
        ax.text(mx + 0.02, my + 0.02, f'M{i+1}', fontsize=8, ha='left', va='bottom', zorder=5)

    # Draw wiring
    W = morph['wiring_matrix']
    S = morph['sign_matrix']

    for i in range(morph['n_sensors']):
        for j in range(morph['n_motors']):
            weight = W[i, j]
            sign = S[i, j]

            if abs(weight) > 0.01:  # Only draw significant connections
                sx, sy = morph['sensor_positions'][i]
                mx, my = morph['motor_positions'][j]

                # Determine color based on sign
                color = 'green' if sign > 0 else 'purple'
                # Determine line style based on weight magnitude
                linewidth = 0.5 + 2 * (abs(weight) / np.max(np.abs(W))) # Scale linewidth by weight
                linestyle = '-' if abs(weight) > 0.5 else '--'

                ax.plot([sx, mx], [sy, my], color=color, linewidth=linewidth, linestyle=linestyle, alpha=0.7)
