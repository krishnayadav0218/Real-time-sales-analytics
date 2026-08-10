# PULSE — Python / Streamlit Edition

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A real-data sales analytics dashboard built with **Streamlit, pandas
and Plotly**. This is a Python rewrite of the static HTML PULSE
dashboard, aimed at people who want a proper Python data stack
(pandas for wrangling, Plotly for charts) instead of client-side
JavaScript — and who can run a local Python process (Streamlit needs
a server, unlike the static HTML version).

## Why this version over the HTML one

- **No CORS issues on Live URL** — the fetch happens server-side in
  Python (`requests`), so it isn't restricted by the browser's
  same-origin policy the way a static site's `fetch()` call is.
- **pandas under the hood** — real `resample()`, `groupby()`, and
  vectorized aggregation instead of hand-written JS bucketing.
- **Easy to extend** — add a database connector, a scheduled ETL job,
  a `scikit-learn` forecast, etc. — it's all just Python.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Project structure

```
.
├── app.py                      # the whole app
├── requirements.txt
├── pulse-data-template.csv     # expected column shape for uploads
├── .streamlit/config.toml      # dark theme matching the dashboard
├── .gitignore
├── LICENSE
└── README.md
```

## Getting your data in

Use the sidebar (📊 PULSE — Your Data):

- **Upload** — CSV, Excel (`.xlsx`), or JSON. Paste-from-Excel also
  works via the text box. Columns are auto-detected (date, region,
  category, product, amount, or quantity × unit price) — confirm the
  mapping and click **Load this data**. `pulse-data-template.csv` in
  this folder shows the expected shape.
- **Live URL** — point it at a published Google Sheet CSV link or any
  JSON/CSV API. Polls every 10s–5min using `st_autorefresh`. Because
  the request is made from the Python process, most APIs work without
  any CORS configuration on their end.
- **Manual** — log one transaction at a time from a small form.
- **Demo** — simulated data for testing, same as before.

## What's inside

- Real period-over-period KPI deltas (revenue, orders, AOV, and an
  adaptive 4th metric: conversion rate if you map a visits column,
  unique customers if you map a customer column, else distinct
  products).
- Date range + granularity (hour/day/week/month, `pandas.resample`)
  driving the revenue chart, with an optional previous-period overlay.
- Region/category breakdowns (long tail grouped into "Other"),
  click-free filters via the sidebar-style top bar.
- Revenue goal progress bar.
- Top-products leaderboard.
- Anomaly detection (z-score > 3σ) and big-order alerts, shown as an
  in-page alert log plus `st.toast()` popups.
- CSV export of the filtered table, JSON export of the full dataset.

## Deploying

**Streamlit Community Cloud (free)**
1. Push this folder to a new GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, "New app", pick the repo/branch, set main file to `app.py`.
3. Deploy — it installs `requirements.txt` automatically.

**Anywhere else** — any VM/container that can run
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
and expose that port works (Render, Railway, a Docker container, etc.).

## Notes

- Data lives in `st.session_state` for the life of the running
  process — it isn't written to disk. Use the JSON export button if
  you want a backup, or point Live URL at a spreadsheet you already
  maintain.
- `streamlit-autorefresh` is what drives the demo "live" feel and the
  live-URL polling; if it's not installed, the app still works, just
  without automatic reruns (add a manual refresh by re-running the
  script or clicking any widget).

## License

MIT — see [LICENSE](LICENSE).

