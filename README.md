# Fulfillment Center Operations Analytics
### Investigating and Predicting Late-Dispatch Risk Across a Multi-Site Logistics Network

**Author:** Shreya M. Pillai | [LinkedIn](https://linkedin.com/in/shreyampillai/)

## Business Problem

E-commerce fulfillment operations run on tight dispatch SLAs. When a site starts missing
on-time dispatch targets, it's usually too late to fix by the time leadership sees the daily
report. This project asks two questions a real ops/analytics team would ask:

1. **What actually drives late dispatch and rising defect rates** across sites — staffing mix,
   backlog, seasonality, or something else?
2. **Can we build an early-warning model** that flags high-risk days *before* the shift starts,
   using only information available in real time?

## Dataset

`data/fulfillment_operations.csv` — 3,655 rows: 2 years of daily operational data (2024–2025)
across 5 simulated fulfillment center sites (Leeds, Manchester, Birmingham, Doncaster, Peterborough).

**Note on the data:** This is a synthetic dataset, generated in `generate_data.py` to reflect
real relationships found in fulfillment/warehouse operations (e.g. defect rate rising with
backlog and temp-staff ratio, dispatch performance dropping in peak season). Real operational
data from employers is confidential, so this was built from the ground up to mirror those
genuine patterns rather than pulling a generic public dataset — the structure and metrics
(throughput, backlog, temp-staff ratio, defect rate, on-time dispatch %) reflect metrics I
work with directly in my current role leading fulfillment operations.

**Columns include:** `order_volume`, `core_staff`, `temp_staff`, `temp_staff_ratio`,
`absenteeism_pct`, `backlog_units`, `defect_rate_pct`, `on_time_dispatch_pct`,
`late_dispatch_risk` (target), `safety_incident`, plus site/date/seasonality fields.

## Approach

1. **Data cleaning** — handled missing values (site-level median imputation), validated ranges
2. **Exploratory analysis** — KPI trends over time, site-level benchmarking, peak season impact
3. **Root cause analysis** — correlation analysis to identify controllable levers (backlog,
   temp-staff ratio) vs. uncontrollable ones (demand volume)
4. **Predictive modeling** — Logistic Regression vs. Random Forest classifiers to flag
   high-risk dispatch days, using only real-time-available features (no data leakage from
   outcome metrics)
5. **Business recommendations** — translated findings into concrete, actionable operational
   changes

Full analysis: [`notebooks/01_fulfillment_analysis.ipynb`](notebooks/01_fulfillment_analysis.ipynb)

## Key Findings

- **Backlog pressure is the single strongest predictor** of late dispatch — a leading
  indicator that could be monitored proactively rather than discovered after the fact
- **Temp-staff ratio correlates strongly with defect rate**, especially during peak season
  when temp staffing surges — suggesting a training/mentoring gap rather than a demand problem
- Peak season (Nov–Dec) shows a clear, consistent performance dip across all 5 sites,
  confirming this is a structural capacity issue, not a one-off
- Site-level benchmarking shows meaningful, consistent performance gaps between sites —
  worth investigating as a best-practice-sharing opportunity

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

## Repo Structure

```
fc-analytics-project/
├── data/
│   └── fulfillment_operations.csv
├── notebooks/
│   └── 01_fulfillment_analysis.ipynb
├── charts/
│   └── (exported PNG charts referenced in the notebook)
├── generate_data.py          # synthetic data generation script
├── INTERVIEW_PREP.md         # my own talking-points cheat sheet for this project
└── README.md
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
python generate_data.py                 # regenerates data/fulfillment_operations.csv
jupyter notebook notebooks/01_fulfillment_analysis.ipynb
```
