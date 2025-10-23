import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from scipy.stats import linregress

from .models import HistoricalDataPoint

# --- Template View (for the main interface) ---

def index(request):
    """Renders the main application interface (the map and controls)."""
    return render(request, "weather/index.html")

# -----------------------------------------------------------------------------
# Migrated Statistical & Helper Functions from backend/main.py
# -----------------------------------------------------------------------------

def calculate_heat_index(t, rh):
    if t is None or rh is None or pd.isna(t) or pd.isna(rh):
        return None
    t_f = t * 9 / 5 + 32
    hi_f = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))
    if hi_f < 80:
        return t
    return (hi_f - 32) * 5 / 9

def empirical_probability(values, threshold):
    n = len(values)
    if n == 0:
        return None, 0, 0
    exceed = int(np.sum(values > threshold))
    return float(exceed) / n, exceed, int(n)

def bootstrap_ci(values, threshold, n_boot=1000, ci=95, random_state=0):
    rng = np.random.default_rng(random_state)
    n = len(values)
    if n == 0:
        return None, None, None, None
    probs = [
        np.sum(rng.choice(values, size=n, replace=True) > threshold) / n
        for _ in range(n_boot)
    ]
    lower = float(np.percentile(probs, (100 - ci) / 2))
    upper = float(np.percentile(probs, 100 - (100 - ci) / 2))
    return lower, upper, float(np.mean(probs)), float(np.std(probs))

def compute_linear_trend(years, values):
    if len(years) < 3:
        return None
    res = linregress(years, values)
    return {
        "slope_per_year": res.slope,
        "intercept": res.intercept,
        "r_value": res.rvalue,
        "p_value": res.pvalue,
        "stderr": res.stderr,
    }

def compute_exceedance_trend(years, values, threshold):
    bin_vals = (np.array(values) > threshold).astype(float)
    trend = compute_linear_trend(years, bin_vals)
    return trend, int(bin_vals.sum()), len(bin_vals)

def decadal_summary(years, values, threshold=None, decade_span=10):
    df = pd.DataFrame({"year": list(years), "value": list(values)})
    df["decade_start"] = (df["year"] // decade_span) * decade_span
    summary = []
    for start, g in df.groupby("decade_start"):
        years_used = g["year"].nunique()
        if threshold is None:
            val = g["value"].mean()
            key = "mean_value"
        else:
            val = (g["value"] > threshold).sum() / years_used if years_used > 0 else 0
            key = "prob_exceed"
        summary.append(
            {
                "decade_start": int(start),
                "decade_end": int(start + decade_span - 1),
                "years_used": years_used,
                key: val,
            }
        )
    return summary

def _sanitize_value(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_sanitize_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _sanitize_value(val) for k, val in v.items()}
    return v

def get_yearly_data(lat, lon, var, doy=None, doy_start=None, doy_end=None, agg="mean"):
    """Fetches and processes yearly data from the database."""
    qs = HistoricalDataPoint.objects.filter(latitude=lat, longitude=lon)

    if doy:
        qs = qs.filter(day_of_year=doy)
    elif doy_start and doy_end:
        qs = qs.filter(day_of_year__gte=doy_start, day_of_year__lte=doy_end)

    if var == "heat_index":
        # Heat index must be calculated from T2M and RH2M
        df = pd.DataFrame.from_records(qs.values("year", "t2m", "rh2m"))
        if df.empty:
            return pd.Series([], dtype=float)
        df["heat_index"] = df.apply(
            lambda row: calculate_heat_index(row["t2m"], row["rh2m"]), axis=1
        )
        yearly = df.groupby("year")["heat_index"].mean() # Assuming mean for heat index
    else:
        # For direct variables, use database aggregation
        df = pd.DataFrame.from_records(qs.values("year", var))
        if df.empty:
            return pd.Series([], dtype=float)
        yearly = df.groupby("year")[var].mean() # Default to mean

    return yearly.dropna()

# -----------------------------------------------------------------------------
# API Views
# -----------------------------------------------------------------------------

def percentiles(request):
    try:
        lat = float(request.GET.get("lat"))
        lon = float(request.GET.get("lon"))
        var = request.GET.get("var")
        doy = int(request.GET.get("doy"))
    except (TypeError, ValueError) as e:
        return HttpResponseBadRequest(f"Invalid parameter: {e}")

    yearly_data = get_yearly_data(lat, lon, var, doy=doy)

    if len(yearly_data) < 10:
        return JsonResponse(
            {"error": "Not enough data to calculate percentiles"}, status=400
        )

    percentiles_data = {
        "p10": yearly_data.quantile(0.10),
        "p25": yearly_data.quantile(0.25),
        "p50": yearly_data.quantile(0.50),
        "p75": yearly_data.quantile(0.75),
        "p90": yearly_data.quantile(0.90),
    }
    return JsonResponse(_sanitize_value(percentiles_data))

def probability_view(request):
    try:
        lat = float(request.GET.get("lat"))
        lon = float(request.GET.get("lon"))
        var = request.GET.get("var")
        threshold = float(request.GET.get("threshold"))
        doy = request.GET.get("doy")
        n_boot = int(request.GET.get("n_boot", 1000))
    except (TypeError, ValueError) as e:
        return HttpResponseBadRequest(f"Invalid parameter: {e}")

    yearly = get_yearly_data(lat, lon, var, doy=doy)

    if len(yearly) < 5:
        return JsonResponse({"error": "Insufficient years of data"}, status=400)

    prob, exceed_count, n = empirical_probability(yearly.values, threshold)
    ci_lower, ci_upper, boot_mean, boot_std = bootstrap_ci(
        yearly.values, threshold, n_boot=n_boot
    )

    response = {
        "ok": True,
        "variable": var,
        "doy": int(doy) if doy else None,
        "threshold": threshold,
        "years_used": n,
        "exceed_count": exceed_count,
        "probability": prob,
        "bootstrap": {
            "n_boot": n_boot,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "bootstrap_mean": boot_mean,
            "bootstrap_std": boot_std,
        },
        "computed_on": datetime.now(timezone.utc).isoformat(),
    }
    return JsonResponse(_sanitize_value(response))

def trend_view(request):
    try:
        lat = float(request.GET.get("lat"))
        lon = float(request.GET.get("lon"))
        var = request.GET.get("var")
        doy = request.GET.get("doy")
        threshold = request.GET.get("threshold")
        if threshold:
            threshold = float(threshold)
    except (TypeError, ValueError) as e:
        return HttpResponseBadRequest(f"Invalid parameter: {e}")

    yearly = get_yearly_data(lat, lon, var, doy=doy)
    years = yearly.index.astype(int).tolist()
    values = yearly.values.astype(float).tolist()

    if len(years) < 10:
        return JsonResponse({"error": "Insufficient years for trend analysis"}, status=400)

    value_trend = compute_linear_trend(years, values)
    exceedance_trend, _, _ = (
        compute_exceedance_trend(years, values, threshold)
        if threshold is not None
        else (None, None, None)
    )
    decadal = decadal_summary(years, values, threshold=threshold)

    response = {
        "ok": True,
        "variable": var,
        "doy": int(doy) if doy else None,
        "years_used": len(years),
        "value_trend": value_trend,
        "exceedance_trend": exceedance_trend,
        "decadal_summary": decadal,
        "computed_on": datetime.now(timezone.utc).isoformat(),
    }
    return JsonResponse(_sanitize_value(response))

def history_view(request):
    try:
        lat = float(request.GET.get("lat"))
        lon = float(request.GET.get("lon"))
        var = request.GET.get("var")
        doy = request.GET.get("doy")
    except (TypeError, ValueError) as e:
        return HttpResponseBadRequest(f"Invalid parameter: {e}")

    yearly = get_yearly_data(lat, lon, var, doy=doy)
    years = yearly.index.astype(int).tolist()
    values = yearly.values.astype(float).tolist()

    response = {
        "ok": True,
        "variable": var,
        "doy": int(doy) if doy else None,
        "years": years,
        "values": values,
        "years_used": len(years),
        "generated_on": datetime.now(timezone.utc).isoformat(),
    }
    return JsonResponse(_sanitize_value(response))