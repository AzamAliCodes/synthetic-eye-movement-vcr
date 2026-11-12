"""Natural eye-motion engine (pure numpy; no bpy).  Produces per-frame channel traces for the 3 video inputs.

Behaviour model (oculomotor literature, cf. Samsung's reference videos 'Eye Piece' / 'Flexpressions' by Chris Jones):
  saccades: minimum-jerk, duration ~ (21 ms + 2.2 ms/deg) (stretched x1.6 for 24-30 fps readability), both eyes conjugate
  fixation: slow drift + microsaccades
  blinks: ~90 ms close, ~200 ms open, small downward dip, triggered around large saccades
  lid-follow: handled in the .blend by drivers (lid_up / lid_down follow vertical gaze)
  brows: raise with upward gaze / eye roll, slight frown on effortful side glances
  pupil: hippus +-3% plus small constriction when looking to the light (not modelled further)
All angles in degrees; yaw>0 = toward image right (convention 'image'), pitch>0 = up.
"""
import math
import random
import numpy as np

CHANNELS = ("yaw", "pitch", "blink", "brow_raise", "brow_inner", "brow_frown", "squint", "pupil")


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)           # minimum-jerk


class Trace:
    def __init__(self, duration, fps):
        self.fps = fps
        self.n = int(round(duration * fps))
        self.t = np.arange(self.n) / fps
        self.fix = []                                          # (t_start, yaw, pitch) fixation targets
        self.ch = {k: np.zeros(self.n) for k in CHANNELS}
        self.blinks = []                                       # blink start times

    def add_fixation(self, t, yaw, pitch):
        self.fix.append((t, yaw, pitch))

    def add_blink(self, t):
        self.blinks.append(t)


def _saccade_gaze(tr, rng):
    """Piecewise minimum-jerk gaze through the fixation list + drift/microsaccade noise."""
    fix = sorted(tr.fix)
    yaw = np.full(tr.n, fix[0][1], float)
    pit = np.full(tr.n, fix[0][2], float)
    cur_y, cur_p = fix[0][1], fix[0][2]
    for (t0, ty, tp) in fix[1:]:
        amp = math.hypot(ty - cur_y, tp - cur_p)
        dur = max(0.05, (0.021 + 0.0022 * amp) * 1.6)
        s = _smoothstep((tr.t - t0) / dur)
        over = 0.035 * amp * np.exp(-((tr.t - (t0 + dur * 1.15)) / (0.05)) ** 2) * (tr.t > t0 + dur * 0.8)
        dy, dp = ty - cur_y, tp - cur_p
        unit = np.array([dy, dp]) / (amp + 1e-9)
        yaw = yaw + s * dy * (tr.t >= t0) + over * unit[0]
        pit = pit + s * dp * (tr.t >= t0) + over * unit[1]
        cur_y, cur_p = ty, tp
    # drift (slow, 0.12 deg) + microsaccades (every ~0.7 s, 0.2-0.35 deg)
    ph = [rng.uniform(0, 2 * math.pi) for _ in range(4)]
    yaw += 0.12 * (np.sin(2 * math.pi * 0.31 * tr.t + ph[0]) + 0.5 * np.sin(2 * math.pi * 0.83 * tr.t + ph[1]))
    pit += 0.10 * (np.sin(2 * math.pi * 0.27 * tr.t + ph[2]) + 0.5 * np.sin(2 * math.pi * 0.71 * tr.t + ph[3]))
    t = rng.uniform(0.3, 0.9)
    while t < tr.t[-1]:
        a = rng.uniform(0.2, 0.35)
        ang = rng.uniform(0, 2 * math.pi)
        s = _smoothstep((tr.t - t) / 0.04)
        yaw += s * a * math.cos(ang)
        pit += s * a * math.sin(ang)
        # return drift so microsaccades do not accumulate
        yaw -= _smoothstep((tr.t - (t + 0.35)) / 0.30) * a * math.cos(ang)
        pit -= _smoothstep((tr.t - (t + 0.35)) / 0.30) * a * math.sin(ang)
        t += rng.uniform(0.6, 1.2)
    return yaw, pit


def _blink_profile(tr, t0):
    close, hold, opn = 0.09, 0.03, 0.20
    x = tr.t - t0
    down = _smoothstep(x / close)
    up = 1.0 - _smoothstep((x - close - hold) / opn)
    return np.clip(np.where(x < close + hold, down, up), 0, 1) * (x >= 0)


def finalize(tr, rng):
    yaw, pit = _saccade_gaze(tr, rng)
    blink = np.zeros(tr.n)
    dip = np.zeros(tr.n)
    for t0 in tr.blinks:
        p = _blink_profile(tr, t0)
        blink = np.maximum(blink, p)
        dip -= 2.5 * p
    pit = pit + dip                                            # eyes dip slightly during a blink
    ch = tr.ch
    ch["yaw"], ch["pitch"], ch["blink"] = yaw, pit, np.clip(blink, 0, 1)
    up = _smoothstep((pit - 6.0) / 16.0)
    ch["brow_raise"] = np.clip(0.85 * up + ch["brow_raise"], 0, 1)
    ch["brow_frown"] = np.clip(ch["brow_frown"] + 0.30 * _smoothstep((np.abs(yaw) - 12.0) / 10.0), 0, 1)
    ch["squint"] = np.clip(ch["squint"], 0, 1)
    ph = rng.uniform(0, 2 * math.pi)
    ch["pupil"] = 0.5 + 0.5 * 0.06 * np.sin(2 * math.pi * 0.5 * tr.t + ph) + ch["pupil"]   # centred, +-3 %
    return tr


def _ramp(tr, t0, t1, a, b):
    return np.where(tr.t < t0, 0.0, np.where(tr.t > t1, 0.0, 0.0)) + 0.0


def _bump(tr, t0, t1, peak, rise=0.25, fall=0.35):
    return peak * _smoothstep((tr.t - t0) / rise) * (1 - _smoothstep((tr.t - t1) / fall))


def script_left(duration, fps, seed):
    rng = random.Random(seed)
    tr = Trace(duration, fps)
    T = duration
    tr.add_fixation(0.0, 0.0, 0.0)
    tr.add_blink(rng.uniform(0.7, 1.0))
    tr.add_fixation(1.5, -20.0, 2.0)
    tr.add_fixation(1.5 + 2.4, -23.0, 1.0)                    # small secondary saccade further left
    tr.add_blink(1.5 + 3.3)
    tr.add_fixation(T - 4.0, -6.0, 0.0)
    tr.add_fixation(T - 2.2, 0.0, 0.0)
    tr.ch["brow_frown"] += _bump(tr, 1.5, 4.6, 0.25)
    return finalize(tr, rng)


def script_right(duration, fps, seed):
    rng = random.Random(seed + 101)
    tr = Trace(duration, fps)
    T = duration
    tr.add_fixation(0.0, 0.0, 0.0)
    tr.add_blink(rng.uniform(0.8, 1.2))
    tr.add_fixation(1.7, 20.0, 1.5)
    tr.add_fixation(1.7 + 2.2, 23.5, 0.5)
    tr.add_blink(1.7 + 3.4)
    tr.add_fixation(T - 3.8, 6.0, 0.0)
    tr.add_fixation(T - 2.0, 0.0, 0.0)
    tr.ch["brow_frown"] += _bump(tr, 1.7, 4.4, 0.2)
    return finalize(tr, rng)


def script_generate(duration, fps, seed):
    """Centre -> left -> right scan -> blink -> upward eye roll -> down -> back to centre."""
    rng = random.Random(seed)
    tr = Trace(duration, fps)
    T = duration
    u = T / 12.0                                                # time unit so the story scales with duration
    tr.add_fixation(0.0, 0.0, 0.0)
    tr.add_blink(0.6 * u)
    tr.add_fixation(1.1 * u, -18.0, 1.0)
    tr.add_fixation(2.4 * u, 17.0, 0.0)
    tr.add_blink(3.6 * u)
    tr.add_fixation(4.0 * u, 0.0, 0.0)
    # eye roll: smooth pursuit around a circle starting at the left, going up and over to the right
    t0, t1 = 4.9 * u, 6.5 * u
    roll_yaw = np.zeros(tr.n)
    roll_pit = np.zeros(tr.n)
    x = np.clip((tr.t - t0) / (t1 - t0), 0, 1)
    on = (tr.t >= t0) & (tr.t <= t1)
    ang = math.pi - math.pi * _smoothstep(x)                    # pi (left) -> 0 (right) via the top
    rad = 19.0 * np.sin(np.pi * np.clip(x, 0, 1)) ** 0.5
    roll_yaw = np.where(on, rad * np.cos(ang), 0.0)
    roll_pit = np.where(on, 20.0 * np.sin(ang) * np.sin(np.pi * np.clip(x, 0, 1)) ** 0.3, 0.0)
    tr.add_blink(6.9 * u)
    tr.add_fixation(7.3 * u, 0.0, -17.0)                        # look down (lids drop via drivers)
    tr.add_fixation(9.4 * u, 0.0, 0.0)
    tr.add_blink(10.0 * u)
    tr.ch["brow_raise"] += _bump(tr, 4.9 * u, 6.9 * u, 0.55)
    tr.ch["squint"] += _bump(tr, 10.2 * u, 11.2 * u, 0.25, 0.3, 0.5)
    tr.ch["brow_inner"] += _bump(tr, 4.7 * u, 6.8 * u, 0.25)
    out = finalize(tr, rng)
    out.ch["yaw"] = out.ch["yaw"] + roll_yaw
    out.ch["pitch"] = out.ch["pitch"] + roll_pit
    out.ch["brow_raise"] = np.clip(out.ch["brow_raise"] + 0.5 * _smoothstep((roll_pit - 6) / 14.0), 0, 1)
    return out


SCRIPTS = {"left": script_left, "right": script_right, "generate": script_generate}


def generate(kind, duration=10.0, fps=24, seed=7):
    if duration > 15.0:
        raise ValueError("videos must be <= 15 s")
    return SCRIPTS[kind](duration, fps, seed)
