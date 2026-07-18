"""
Generates a realistic synthetic dataset simulating daily operations
across multiple e-commerce fulfillment centers (FCs), similar in
structure to real warehouse/logistics operational data.

Why synthetic: real operational data from employers is confidential.
This dataset is built to reflect genuine relationships found in
fulfillment operations (e.g. defect rate rising with backlog and
temp-staff ratio, dispatch performance dropping during peak season)
so the analysis and modeling are meaningful, not just decorative.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ---- Configuration ----
sites = ["LEEDS1", "MANC2", "BIRM1", "DONC3", "PETE1"]
site_baseline_efficiency = {  # baseline productivity multiplier per site
    "LEEDS1": 1.05, "MANC2": 0.95, "BIRM1": 1.00, "DONC3": 0.90, "PETE1": 1.10
}

start_date = pd.Timestamp("2024-01-01")
end_date = pd.Timestamp("2025-12-31")
dates = pd.date_range(start_date, end_date, freq="D")

rows = []

for site in sites:
    base_eff = site_baseline_efficiency[site]
    core_staff = np.random.randint(180, 260)

    for date in dates:
        day_of_week = date.dayofweek  # 0=Mon
        month = date.month
        day_of_year = date.dayofyear

        # --- Seasonality: peak season Nov-Dec (holiday shopping) ---
        is_peak = month in (11, 12)
        peak_multiplier = 1.6 if is_peak else 1.0

        # Prime-day-style spikes (2 random spike days per site per year)
        is_spike_day = np.random.rand() < 0.006

        # --- Weekend effect: lower volume, skeleton staff ---
        is_weekend = day_of_week >= 5
        weekend_multiplier = 0.55 if is_weekend else 1.0

        # --- Demand / order volume ---
        base_orders = 14000 * base_eff
        seasonal_noise = np.random.normal(0, 800)
        order_volume = base_orders * peak_multiplier * weekend_multiplier + seasonal_noise
        if is_spike_day:
            order_volume *= 1.8
        order_volume = max(order_volume, 1000)

        # --- Staffing ---
        # Temp staff ramps up heavily during peak season
        temp_staff = int(np.random.normal(40 if not is_peak else 160, 15))
        temp_staff = max(temp_staff, 0)
        active_core_staff = int(core_staff * np.random.uniform(0.92, 1.0))  # attrition/absence
        total_staff = active_core_staff + temp_staff
        temp_ratio = temp_staff / total_staff if total_staff > 0 else 0

        # --- Absenteeism rate (%) - tends higher in winter (illness) and Mondays ---
        absenteeism = np.random.normal(
            6.5 + (2 if month in (1, 2, 12) else 0) + (1.5 if day_of_week == 0 else 0),
            1.2
        )
        absenteeism = np.clip(absenteeism, 1, 20)

        # --- Throughput (units processed per labor hour) ---
        # Efficiency drops as temp_ratio rises (less trained workforce) and as
        # backlog rises (rushed/error-prone work), improves slightly with tenure/site baseline
        labor_hours = total_staff * 7.5
        throughput_per_hour = (
            18 * base_eff
            - 4.5 * temp_ratio
            - 0.6 * (absenteeism / 10)
            + np.random.normal(0, 0.8)
        )
        throughput_per_hour = max(throughput_per_hour, 5)
        units_processed = throughput_per_hour * labor_hours

        # --- Backlog: unmet orders carried forward pressure ---
        capacity_gap = order_volume - units_processed
        backlog_units = max(capacity_gap, 0) + np.random.normal(0, 300)
        backlog_units = max(backlog_units, 0)

        # --- Defect rate (%) - rises with backlog pressure & temp ratio ---
        defect_rate = (
            1.2
            + 0.0006 * backlog_units
            + 3.5 * temp_ratio
            + (0.8 if is_peak else 0)
            + np.random.normal(0, 0.4)
        )
        defect_rate = np.clip(defect_rate, 0.3, 15)

        # --- On-time dispatch rate (%) - falls with backlog & defect rate ---
        on_time_dispatch = (
            98.5
            - 0.0015 * backlog_units
            - 0.9 * defect_rate
            - (3 if is_spike_day else 0)
            + np.random.normal(0, 1.0)
        )
        on_time_dispatch = np.clip(on_time_dispatch, 60, 100)

        # --- Safety incidents (rare events, more likely when overworked/peak) ---
        safety_incident_prob = 0.01 + (0.02 if is_peak else 0) + (0.015 * temp_ratio)
        safety_incident = 1 if np.random.rand() < safety_incident_prob else 0

        rows.append({
            "date": date,
            "site": site,
            "day_of_week": date.day_name(),
            "is_weekend": is_weekend,
            "is_peak_season": is_peak,
            "order_volume": round(order_volume),
            "core_staff": active_core_staff,
            "temp_staff": temp_staff,
            "total_staff": total_staff,
            "temp_staff_ratio": round(temp_ratio, 3),
            "absenteeism_pct": round(absenteeism, 2),
            "labor_hours": round(labor_hours, 1),
            "units_processed": round(units_processed),
            "throughput_per_labor_hour": round(throughput_per_hour, 2),
            "backlog_units": round(backlog_units),
            "defect_rate_pct": round(defect_rate, 2),
            "on_time_dispatch_pct": round(on_time_dispatch, 2),
            "safety_incident": safety_incident,
        })

df = pd.DataFrame(rows)

# --- Late dispatch risk flag (target variable for modeling) ---
# Defined as the bottom 20% of on-time dispatch performance, calculated
# per site (since sites have different baselines) so the flag represents
# genuine underperformance days rather than an arbitrary global cutoff.
df["site_threshold"] = df.groupby("site")["on_time_dispatch_pct"].transform(lambda x: x.quantile(0.20))
df["late_dispatch_risk"] = (df["on_time_dispatch_pct"] < df["site_threshold"]).astype(int)
df = df.drop(columns=["site_threshold"])

# Inject a small amount of realistic messiness (nulls) to make cleaning meaningful
messy_idx = np.random.choice(df.index, size=int(len(df) * 0.01), replace=False)
df.loc[messy_idx, "absenteeism_pct"] = np.nan

df = df.sort_values(["date", "site"]).reset_index(drop=True)
df.to_csv("data/fulfillment_operations.csv", index=False)

print(f"Generated {len(df):,} rows across {df['site'].nunique()} sites")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(df.head())
