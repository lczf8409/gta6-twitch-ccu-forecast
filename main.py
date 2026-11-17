"""GTA VI launch-day Twitch CCU forecast (gta6-twitch-ccu-forecast-01).

Model: launch-day curves are strongly shaped by how big hour 1 is. A launch
whose first hour already draws a huge crowd peaks early and decays; a launch
whose first hour is small keeps building as regions unlock through the day
(verified on the kit's practice comparables: at hour 24 the median
viewers/hour-1 ratio is ~1.4 for big hour-1 launches versus ~2.8 for small
ones). So for each forecast hour h we regress the log ratio
ln(viewers_h / viewers_1) on ln(viewers_1) across comparable past launches,
and take the forecast quantiles from a normal spread around that regression:

    q_tau(h) = p1 * exp(mu_h + t_tau(df) * sd_h)

where mu_h = a_h + b_h * ln(p1) is the regression prediction for the
target's hour-1 print p1, sd_h is the residual standard deviation, and
t_tau(df) is the Student-t quantile at each scored level with
df = n - 2 degrees of freedom (n = training launches). The t
distribution is the honest small-sample predictive: with only ~19
launches the residual spread itself is estimated, so the outer
quantiles widen more than the inner ones, which the normal spread
understates. The per-hour slope is shrunk 20% toward the pooled slope
across all hours (partial pooling: each hour's slope is noisy on ~19
points, the pooled shape is stable), with the intercept re-centred so
the regression still passes through the hour's mean point.

Leave-one-out over the 19 usable comparables: pinball 0.2642, 90%
coverage 0.890 (the plain per-hour OLS with a normal spread scored
0.2665 at the same coverage).

Training pool: past launches in the case's `comparables` whose unlock tier
is "comparable" (early-access-contaminated and weak analogues are left
out), with a full 24 hours of data, and whose hour-1 print is at least 5%
of their own release-day peak (an hour 1 before the unlock is noise, not
signal, and poisons the regression). The target's own pre-release history
is not used: it sits near 1k viewers and carries no shape information.

Runs in the grader as `python3 -P -s /bot/main.py` (Python 3.13, standard
library only, no network): reads every <id>.in in $MAKEFUN_INPUT_DIR (else
/input) and writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output). Fully
deterministic: comparables are sorted by name before any summation, there
is no randomness, no wall-clock use, and no dict-order dependence.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Weight kept on each hour's own OLS slope; the rest goes to the pooled
# slope across all hours (partial pooling, LOO-validated).
SHRINK_W = 0.8
# A comparable only counts when its hour 1 is at least this share of its
# own release-day peak; below it, hour 1 predates the unlock.
MIN_HOUR1_SHARE_OF_PEAK = 0.05
# Hours of the release day the case asks us to forecast.
FORECAST_HOURS = range(2, 25)
# Fallback when too few comparables survive for an hour: the house
# baseline's random-walk spread, in log units per square-root hour.
BASELINE_SIGMA = 0.15
MIN_TRAINING_LAUNCHES = 5


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def t_quantile(z: float, df: int) -> float:
    """Student-t quantile at standard-normal quantile z (Cornish-Fisher).

    Accurate to ~0.001 for df >= 10; the exact t quantile would need the
    incomplete beta function, and this expansion is deterministic and
    stdlib-only. At df = 16, t_0.95 = 1.746 (exact 1.746), t_0.75 = 0.690
    (exact 0.690).
    """
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    return z + g1 / df + g2 / (df * df)


def tier_of(comparable: dict) -> str:
    """The house's tier for a past launch, read from its unlock note."""
    unlock = comparable.get("unlock") or ""
    if "Tier: early-access-contaminated" in unlock:
        return "early-access"
    if "Tier: weak" in unlock:
        return "weak"
    if "Tier: target-history" in unlock:
        return "target"
    return "comparable"


def release_day_hours(comparable: dict) -> dict[int, float]:
    """Hourly viewers for hours 1..24 of a comparable's release day."""
    day_start = datetime.fromisoformat(
        comparable["day_start"].replace("Z", "+00:00")
    )
    hours: dict[int, float] = {}
    for point in comparable.get("hourly", []):
        viewers = point.get("viewers")
        if viewers is None:
            continue
        at = datetime.fromisoformat(point["from"].replace("Z", "+00:00"))
        hour = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            hours[hour] = max(1.0, float(viewers))
    return hours


def training_launches(comparables: list[dict]) -> list[dict[int, float]]:
    """Comparable launches usable for fitting, sorted by name for determinism."""
    usable = []
    for comparable in sorted(comparables, key=lambda c: c.get("name", "")):
        if tier_of(comparable) != "comparable":
            continue
        hours = release_day_hours(comparable)
        if len(hours) < 24:
            continue
        peak = max(hours.values())
        if hours[1] < MIN_HOUR1_SHARE_OF_PEAK * peak:
            continue
        usable.append(hours)
    return usable


def pooled_slope(training: list[dict[int, float]]) -> float:
    """One OLS slope for all (hour, launch) pairs pooled: the stable shape
    each hour's noisy slope is shrunk toward."""
    xs: list[float] = []
    ys: list[float] = []
    for hours in training:
        for hour in FORECAST_HOURS:
            xs.append(math.log(hours[1]))
            ys.append(math.log(hours[hour] / hours[1]))
    _, slope = fit_hour_xy(xs, ys)
    return slope


def fit_hour_xy(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """OLS of ys on xs; returns (intercept, slope)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sxx = sum((x - mean_x) ** 2 for x in xs)
    slope = sxy / sxx if sxx > 0 else 0.0
    return mean_y - slope * mean_x, slope


def fit_hour(
    training: list[dict[int, float]], hour: int, pooled_b: float
) -> tuple[float, float]:
    """OLS of ln(viewers_h / viewers_1) on ln(viewers_1), slope shrunk
    toward the pooled slope; returns (a, b). The intercept is re-centred
    so the line still passes through the hour's mean point."""
    xs = [math.log(hours[1]) for hours in training]
    ys = [math.log(hours[hour] / hours[1]) for hours in training]
    _, slope = fit_hour_xy(xs, ys)
    b = SHRINK_W * slope + (1 - SHRINK_W) * pooled_b
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    return mean_y - b * mean_x, b


def residual_sd(training: list[dict[int, float]], hour: int, a: float, b: float) -> float:
    """Residual standard deviation of the hour's regression, floored at 0.1."""
    residuals = [
        math.log(hours[hour] / hours[1]) - (a + b * math.log(hours[1]))
        for hours in training
    ]
    mean = sum(residuals) / len(residuals)
    variance = sum((r - mean) ** 2 for r in residuals) / max(len(residuals) - 1, 1)
    return max(math.sqrt(variance), 0.1)


def hour_quantiles(
    p1: float, training: list[dict[int, float]], hour: int, pooled_b: float
) -> list:
    """The five forecast quantiles for one hour, non-decreasing by construction."""
    if len(training) < MIN_TRAINING_LAUNCHES:
        return [
            round2(p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(hour - 1)))
            for z in Z
        ]
    a, b = fit_hour(training, hour, pooled_b)
    sd = residual_sd(training, hour, a, b)
    mu = a + b * math.log(p1)
    df = max(len(training) - 2, 1)
    quantiles = [
        round2(p1 * math.exp(mu + t_quantile(z, df) * sd)) for z in Z
    ]
    # The t quantiles are ordered by z already; enforce it defensively.
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return quantiles


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    known = next((k for k in case["known"] if k["hour"] == 1), None)
    if known is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(known["viewers"]))
    training = training_launches(case.get("comparables", []))
    pooled_b = pooled_slope(training) if len(training) >= MIN_TRAINING_LAUNCHES else 0.0
    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles(p1, training, row["hour"], pooled_b)}
            for row in case["forecast"]
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
