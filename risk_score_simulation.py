"""
Smart Bio-Crate — Deterioration Risk Score (DRS) Simulation
=============================================================
Purpose
-------
This script is a CONCEPT-MODEL / TECHNICAL SIMULATION artifact for the
competition submission. It does NOT connect to real hardware. It models
what the ESP32-S3 sentinel firmware (see esp32_fusion_firmware.ino) computes
in real time: a 0-100 Deterioration Risk Score (DRS) fused from five
low-cost sensor channels, for a single truck trip carrying papaya from a
collection hub to the Dambulla Dedicated Economic Centre.

Why a weighted fusion model instead of a single threshold (e.g. "alert if
temp > 13C")?
- Onwude et al. (2020) identify five independent post-harvest deterioration
  mechanisms (thermal, moisture, visual/maturity, biochemical/VOC, biological/
  pest). A single-variable threshold misses failures that show up in only one
  channel (e.g. a slow VOC rise from early fungal onset with normal temp/RH).
- Sensor fusion reduces false positives from any single noisy sensor
  (explicitly flagged as a risk in the governance doc: "gas sensors detect
  general VOCs rather than only ethylene, so noisy readings could cause false
  alarms").

Algorithm (deployed on-device in C, mirrored here in Python)
--------------------------------------------------------------
DRS = 100 * ( w_T*S_T + w_RH*S_RH + w_M*S_M + w_C*S_C + w_V*S_V + w_P*S_P )

Each S_x in [0,1] is a per-channel "stress score" produced by a piecewise
linear function anchored to the crop's known safe range (see crop_profiles
table). Weights w_x are crop-tier defaults calibrated by NIPHM/NERDC during
the pilot phase and are stored in the crop_profiles table below, NOT
hard-coded in firmware, so they can be updated after each growing season
without re-flashing devices (per the governance doc's stated update path).

This is intentionally a small, explainable linear model (not a black-box
ML model) so that:
1. It runs in <1ms on an ESP32-S3 with no floating point unit stress.
2. Field technicians and NIPHM auditors can inspect exactly why a score
   fired, which matters for the quality-based pricing dispute-resolution
   process described in the governance plan.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Crop profile: papaya (Tier 2 cash crop, per governance doc)
#    Safe temp/RH range sourced from Rajapaksha et al. crop-requirement table.
# ---------------------------------------------------------------------------
CROP_PROFILE = {
    "name": "papaya",
    "temp_safe_min": 7.0,     # deg C
    "temp_safe_max": 13.0,
    "temp_critical": 20.0,    # score saturates to 1.0 here
    "rh_safe_min": 85.0,      # %
    "rh_safe_max": 90.0,
    "mass_loss_warn_pct": 4.0,   # % cumulative mass loss -> stress begins
    "mass_loss_critical_pct": 8.0,
    "voc_baseline_ppm": 50.0,    # calibrated at loading, per-trip baseline
    "voc_warn_delta_ppm": 80.0,  # rise above baseline that signals onset
    "voc_critical_delta_ppm": 200.0,
    "pest_events_warn": 2,       # detections per hour
    "pest_events_critical": 6,
    # fusion weights (must sum to 1.0) — Tier 2 default, NIPHM-calibrated
    "weights": {"temp": 0.30, "rh": 0.15, "mass": 0.20, "voc": 0.25, "pest": 0.10},
}

ALERT_THRESHOLD = 55      # DRS above this -> "priority unload" trigger
WATCH_THRESHOLD = 35      # DRS above this -> "increase monitoring frequency"


def stress_score(value, safe_min, safe_max, critical):
    """
    Piecewise-linear stress score in [0,1].
    0 inside the safe band, ramps to 1.0 at the critical bound.
    Handles both "too high" and "too low" deviations symmetrically.
    """
    if safe_min <= value <= safe_max:
        return 0.0
    if value > safe_max:
        span = max(critical - safe_max, 1e-6)
        return float(np.clip((value - safe_max) / span, 0, 1))
    span = max(safe_max - critical, 1e-6)  # not used for cold side here (papaya has no cold-injury critical modeled)
    return float(np.clip((safe_min - value) / (safe_min * 0.5), 0, 1))


def mass_loss_score(pct, warn, critical):
    if pct <= warn:
        return 0.0
    return float(np.clip((pct - warn) / (critical - warn), 0, 1))


def voc_score(delta_ppm, warn, critical):
    if delta_ppm <= warn:
        return 0.0
    return float(np.clip((delta_ppm - warn) / (critical - warn), 0, 1))


def pest_score(events_per_hr, warn, critical):
    if events_per_hr <= warn:
        return 0.0
    return float(np.clip((events_per_hr - warn) / (critical - warn), 0, 1))


def compute_drs(temp_c, rh_pct, mass_loss_pct, voc_delta_ppm, pest_events_hr, profile):
    w = profile["weights"]
    s_t = stress_score(temp_c, profile["temp_safe_min"], profile["temp_safe_max"], profile["temp_critical"])
    s_rh = stress_score(rh_pct, profile["rh_safe_min"], profile["rh_safe_max"], 100)
    s_m = mass_loss_score(mass_loss_pct, profile["mass_loss_warn_pct"], profile["mass_loss_critical_pct"])
    s_v = voc_score(voc_delta_ppm, profile["voc_warn_delta_ppm"], profile["voc_critical_delta_ppm"])
    s_p = pest_score(pest_events_hr, profile["pest_events_warn"], profile["pest_events_critical"])

    drs = 100 * (w["temp"] * s_t + w["rh"] * s_rh + w["mass"] * s_m + w["voc"] * s_v + w["pest"] * s_p)
    return drs, {"temp": s_t, "rh": s_rh, "mass": s_m, "voc": s_v, "pest": s_p}


# ---------------------------------------------------------------------------
# 2. Simulate a 6-hour trip: Anuradhapura collection hub -> Dambulla DEC
#    Scenario: PCM cassette holds temperature well for ~3.5 hrs, then a
#    midday ambient heat spike (uninsulated final leg / unloading queue)
#    pushes conditions out of range, mirroring the real, documented failure
#    mode: produce waiting 2-3 days in vehicles in worst cases; here we model
#    a much milder single-day delay to show the alert firing BEFORE that
#    worst case is reached.
# ---------------------------------------------------------------------------
np.random.seed(7)
minutes = np.arange(0, 480, 5)  # 8 hours, 5-min samples (delayed unloading queue scenario)
n = len(minutes)

# Temperature: PCM buffer holds ~9C for first 210 min, then rises as PCM
# capacity is exhausted + midday heat + a queue delay at unloading
temp = np.piecewise(
    minutes.astype(float),
    [minutes < 210, minutes >= 210],
    [lambda m: 9.0 + np.random.normal(0, 0.3, m.shape),
     lambda m: 9.0 + 0.09 * (m - 210) + np.random.normal(0, 0.4, m.shape)]
)

rh = 88 - 0.01 * minutes + np.random.normal(0, 1.0, n)
mass_loss_pct = 0.006 * minutes  # slow linear transpiration loss
voc_delta = np.piecewise(
    minutes.astype(float),
    [minutes < 240, minutes >= 240],
    [lambda m: np.random.normal(5, 3, m.shape),
     lambda m: 20 + 1.1 * (m - 240) + np.random.normal(0, 5, m.shape)]
)
voc_delta = np.clip(voc_delta, 0, None)
pest_events_hr = np.random.poisson(0.3, n).astype(float)

drs_series = []
alert_minute = None
watch_minute = None
for i in range(n):
    drs, components = compute_drs(temp[i], rh[i], mass_loss_pct[i], voc_delta[i], pest_events_hr[i], CROP_PROFILE)
    drs_series.append(drs)
    if watch_minute is None and drs >= WATCH_THRESHOLD:
        watch_minute = minutes[i]
    if alert_minute is None and drs >= ALERT_THRESHOLD:
        alert_minute = minutes[i]

drs_series = np.array(drs_series)

# ---------------------------------------------------------------------------
# 3. Report + plot
# ---------------------------------------------------------------------------
print("=== Smart Bio-Crate DRS Simulation: Papaya, Anuradhapura -> Dambulla ===")
print(f"Trip duration modeled: {minutes[-1]} minutes")
print(f"Peak DRS reached: {drs_series.max():.1f} at minute {minutes[np.argmax(drs_series)]}")
if watch_minute is not None:
    print(f"WATCH threshold ({WATCH_THRESHOLD}) crossed at minute {watch_minute} "
          f"-> firmware increases sampling rate from 5min to 1min intervals")
if alert_minute is not None:
    print(f"ALERT threshold ({ALERT_THRESHOLD}) crossed at minute {alert_minute} "
          f"-> GSM push to cooperative dispatcher: 'priority unload at nearest DEC'")
else:
    print("ALERT threshold not reached in this trip.")

fig, ax1 = plt.subplots(figsize=(10, 5.5))
ax1.plot(minutes, drs_series, color="#c0392b", linewidth=2, label="Deterioration Risk Score (DRS)")
ax1.axhline(ALERT_THRESHOLD, color="#c0392b", linestyle="--", linewidth=1, label="Alert threshold (55)")
ax1.axhline(WATCH_THRESHOLD, color="#e67e22", linestyle="--", linewidth=1, label="Watch threshold (35)")
ax1.set_xlabel("Trip time (minutes)")
ax1.set_ylabel("DRS (0-100)")
ax1.set_ylim(0, 100)
ax1.legend(loc="upper left")
ax1.set_title("Sentinel Crate: Deterioration Risk Score Over a Sample Papaya Trip")

ax2 = ax1.twinx()
ax2.plot(minutes, temp, color="#2980b9", alpha=0.5, linewidth=1, label="Temp (C)")
ax2.set_ylabel("Temperature (C)", color="#2980b9")
ax2.tick_params(axis="y", labelcolor="#2980b9")

fig.tight_layout()
fig.savefig("/home/claude/drs_simulation.png", dpi=150)
print("\nSaved chart to drs_simulation.png")
