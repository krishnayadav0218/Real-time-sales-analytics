"""
PULSE — Real-Time Sales Intelligence (Python / Streamlit edition)
===================================================================
A real-data sales analytics dashboard built with Streamlit, pandas and
Plotly. Supports three ways to get your own data in:

  1. Upload a CSV / Excel / JSON file
  2. Poll a live URL (CSV or JSON) on an interval — runs server-side,
     so it isn't subject to browser CORS restrictions the way a static
     HTML dashboard would be.
  3. Add transactions manually, one at a time.

...plus a demo mode that simulates a live stream for testing.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import json
import time
import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ============================================================
# PAGE CONFIG + THEME
# ============================================================
st.set_page_config(page_title="PULSE — Sales Intelligence", page_icon="📊", layout="wide")

PALETTE = ["#33E1E0", "#8B7CF6", "#F5A623", "#3DD68C", "#F0556B",
           "#5AC8FA", "#FF9F45", "#B983FF", "#4CD9A8", "#E85D75"]

CUSTOM_CSS = """
<style>
:root{
  --bg-deep:#080B14; --bg-panel:#0F1524; --line:#212B42;
  --text-hi:#E8ECF6; --text-mid:#8B96B3; --cyan:#33E1E0;
}
.stApp{background:var(--bg-deep);}
h1,h2,h3{font-family:'Space Grotesk','Trebuchet MS',sans-serif !important;}
[data-testid="stMetric"]{
  background:linear-gradient(180deg, var(--bg-panel), #131B2E);
  border:1px solid var(--line); border-radius:12px; padding:14px 16px 10px;
}
[data-testid="stMetricLabel"]{color:var(--text-mid) !important; text-transform:uppercase; font-size:11px !important; letter-spacing:1px;}
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace !important;}
div[data-testid="stExpander"]{border:1px solid var(--line); border-radius:10px;}
.pulse-badge{
  display:inline-flex; align-items:center; gap:6px; font-size:12px; color:var(--text-mid);
  border:1px dashed var(--line); border-radius:20px; padding:4px 12px; font-family:monospace;
}
.pulse-dot{width:7px; height:7px; border-radius:50%; background:var(--cyan); display:inline-block;}
.alert-row{
  padding:8px 10px; border-radius:8px; margin-bottom:6px; font-size:13px;
  border:1px solid var(--line); background:var(--bg-panel);
}
.alert-crit{border-left:3px solid #F0556B;}
.alert-warn{border-left:3px solid #F5A623;}
.alert-good{border-left:3px solid #3DD68C;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# COLUMN AUTO-DETECTION
# ============================================================
FIELD_GUESSES = {
    "date": ["date", "timestamp", "time", "order_date", "created_at", "datetime", "order_time"],
    "region": ["region", "country", "location", "market", "geo", "state"],
    "category": ["category", "type", "segment", "product_category", "dept", "department"],
    "product": ["product", "item", "sku", "product_name", "name", "item_name"],
    "amount": ["amount", "revenue", "total", "price", "sales", "value", "order_total", "net_amount", "grand_total"],
    "quantity": ["qty", "quantity", "units", "unit_count"],
    "unit_price": ["unit_price", "price_per_unit", "unitprice", "rate"],
    "customer": ["customer", "customer_id", "client", "user", "email", "buyer"],
    "visits": ["visits", "sessions", "traffic", "impressions"],
}


def guess_field(columns, key):
    lower = [c.lower().strip() for c in columns]
    for pattern in FIELD_GUESSES[key]:
        if pattern in lower:
            return columns[lower.index(pattern)]
    for pattern in FIELD_GUESSES[key]:
        for i, c in enumerate(lower):
            if pattern in c:
                return columns[i]
    return "— none —"


# ============================================================
# PARSING / NORMALIZATION
# ============================================================
def read_any(file_or_text, filename=""):
    """Parse an uploaded file (CSV/Excel/JSON) or raw text into a DataFrame."""
    name = filename.lower()
    if hasattr(file_or_text, "read"):
        raw = file_or_text.read()
        if isinstance(raw, bytes):
            if name.endswith((".xlsx", ".xls")):
                return pd.read_excel(io.BytesIO(raw))
            raw = raw.decode("utf-8", errors="replace")
        text = raw
    else:
        text = file_or_text

    stripped = text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            for key in ("data", "rows", "records"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break
        return pd.DataFrame(parsed)
    return pd.read_csv(io.StringIO(text))


def normalize(df_raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Turn a raw dataframe + column mapping into the internal schema:
    date, region, category, product, amount, customer, visits."""
    out = pd.DataFrame()

    date_col = mapping.get("date")
    out["date"] = pd.to_datetime(df_raw[date_col], errors="coerce", dayfirst=True) if date_col else pd.NaT

    amount_col = mapping.get("amount")
    if amount_col and amount_col != "— none —":
        out["amount"] = pd.to_numeric(
            df_raw[amount_col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce"
        )
    else:
        out["amount"] = np.nan

    qty_col, price_col = mapping.get("quantity"), mapping.get("unit_price")
    if qty_col and qty_col != "— none —" and price_col and price_col != "— none —":
        qty = pd.to_numeric(df_raw[qty_col], errors="coerce")
        price = pd.to_numeric(df_raw[price_col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
        computed = qty * price
        out["amount"] = out["amount"].fillna(computed)

    for field, default in [("region", "Unknown"), ("category", "Uncategorized"), ("product", "Unnamed product")]:
        col = mapping.get(field)
        out[field] = df_raw[col].astype(str).str.strip() if col and col != "— none —" else default
        out[field] = out[field].replace("", default).fillna(default)

    for field in ["customer", "visits"]:
        col = mapping.get(field)
        out[field] = df_raw[col] if col and col != "— none —" else np.nan

    if "visits" in out:
        out["visits"] = pd.to_numeric(out["visits"], errors="coerce")

    before = len(out)
    out = out.dropna(subset=["date", "amount"])
    skipped = before - len(out)
    out = out.reset_index(drop=True)
    return out, skipped


def row_hash(row):
    key = f"{row['date']}|{row['region']}|{row['category']}|{row['product']}|{row['amount']}"
    return hashlib.md5(key.encode()).hexdigest()


# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "mode": "demo",
        "source_label": "Demo (simulated)",
        "data": pd.DataFrame(columns=["date", "region", "category", "product", "amount", "customer", "visits"]),
        "mapping": {},
        "seen_hashes": set(),
        "alerts": [],
        "threshold": 220,
        "goal": 60000,
        "sound": False,
        "live_url": "",
        "live_poll_s": 30,
        "live_active": False,
        "paused": False,
        "recent_amounts": [],
        "goal_hit": False,
        "last_live_fetch": None,
        "demo_seed_done": False,
        "pending_upload_df": None,
        "pending_live_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def push_alert(level, text, toast=True):
    st.session_state.alerts.insert(0, {"level": level, "text": text, "time": datetime.now().strftime("%H:%M:%S")})
    st.session_state.alerts = st.session_state.alerts[:40]
    if toast:
        icon = {"crit": "⛔", "warn": "⚠️", "good": "✅"}.get(level, "ℹ️")
        st.toast(text, icon=icon)


def check_anomaly(amount, region, product, toast=True):
    st.session_state.recent_amounts.append(amount)
    st.session_state.recent_amounts = st.session_state.recent_amounts[-40:]
    if st.session_state.threshold and amount >= st.session_state.threshold:
        push_alert("warn", f"Large order: {product} — ${amount:,.0f} ({region})", toast)
    hist = st.session_state.recent_amounts
    if len(hist) >= 15:
        mean, sd = np.mean(hist), np.std(hist)
        if sd > 0 and amount > mean + 3 * sd and amount > 250:
            push_alert("crit", f"Anomaly: order spike at ${amount:,.0f} (3σ above rolling mean)", toast)


# ============================================================
# DEMO DATA GENERATOR
# ============================================================
DEMO_REGIONS = ["North America", "Europe", "APAC", "Latin America", "MEA"]
DEMO_CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Sporting Goods"]
DEMO_PRODUCTS = {
    "Electronics": ["Wireless Earbuds Pro", "4K Streaming Stick", "Smart Watch S3", "Portable SSD 1TB"],
    "Apparel": ["Merino Wool Sweater", "Running Shorts", "Denim Jacket", "Trail Sneakers"],
    "Home & Kitchen": ["Espresso Maker", "Air Fryer XL", "Ceramic Knife Set", "Aroma Diffuser"],
    "Beauty": ["Vitamin C Serum", "Matte Lip Set", "Hair Repair Mask", "SPF 50 Sunscreen"],
    "Sporting Goods": ["Yoga Mat Pro", "Adjustable Dumbbells", "Cycling Helmet", "Hydration Pack"],
}


def gen_demo_rows(n, spread_minutes=180):
    rows = []
    now = datetime.now()
    for _ in range(n):
        cat = np.random.choice(DEMO_CATEGORIES)
        rows.append({
            "date": now - timedelta(minutes=np.random.uniform(0, spread_minutes)),
            "region": np.random.choice(DEMO_REGIONS),
            "category": cat,
            "product": np.random.choice(DEMO_PRODUCTS[cat]),
            "amount": round(float(np.random.uniform(20, 300)), 2),
            "customer": None,
            "visits": None,
        })
    return pd.DataFrame(rows)


def seed_demo():
    st.session_state.data = gen_demo_rows(220, spread_minutes=180)
    st.session_state.demo_seed_done = True


def tick_demo():
    """Add a few new simulated rows each rerun."""
    new_rows = gen_demo_rows(np.random.randint(1, 4), spread_minutes=1)
    for _, r in new_rows.iterrows():
        check_anomaly(r["amount"], r["region"], r["product"], toast=True)
    st.session_state.data = pd.concat([st.session_state.data, new_rows], ignore_index=True)
    if len(st.session_state.data) > 4000:
        st.session_state.data = st.session_state.data.tail(4000).reset_index(drop=True)


# ============================================================
# INGEST HELPERS
# ============================================================
def ingest(df_norm: pd.DataFrame, append: bool):
    if not append:
        st.session_state.data = df_norm.copy()
        st.session_state.seen_hashes = set(df_norm.apply(row_hash, axis=1))
        return len(df_norm)
    added = 0
    keep_rows = []
    for _, row in df_norm.iterrows():
        h = row_hash(row)
        if h in st.session_state.seen_hashes:
            continue
        st.session_state.seen_hashes.add(h)
        keep_rows.append(row)
        added += 1
    if keep_rows:
        st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame(keep_rows)], ignore_index=True)
    return added


def scan_for_outliers(df: pd.DataFrame):
    if df.empty:
        return
    mean, sd = df["amount"].mean(), df["amount"].std()
    if sd and not np.isnan(sd):
        outliers = df[df["amount"] > mean + 3 * sd].sort_values("amount", ascending=False).head(10)
        for _, r in outliers.iterrows():
            push_alert("crit", f"Outlier in dataset: {r['product']} — ${r['amount']:,.0f} ({r['region']}, {r['date'].strftime('%d/%m/%Y')})", toast=False)
    push_alert("good", f"Loaded {len(df):,} transactions", toast=True)


def fetch_live_df(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    text = resp.text
    if "json" in ctype or url.lower().endswith(".json"):
        return read_any(text)
    return read_any(text)


# ============================================================
# AGGREGATION
# ============================================================
def apply_filters(df, region, category, search, date_from, date_to):
    if df.empty:
        return df
    mask = (df["date"] >= pd.Timestamp(date_from)) & (df["date"] <= pd.Timestamp(date_to) + pd.Timedelta(days=1))
    if region and region != "All":
        mask &= df["region"] == region
    if category and category != "All":
        mask &= df["category"] == category
    if search:
        s = search.lower()
        mask &= (df["product"].str.lower().str.contains(s, na=False) | df["region"].str.lower().str.contains(s, na=False))
    return df[mask]


def previous_period(df, region, category, date_from, date_to):
    if df.empty:
        return df
    span = (date_to - date_from)
    prev_from = date_from - span - timedelta(days=1)
    prev_to = date_from - timedelta(days=1)
    mask = (df["date"] >= pd.Timestamp(prev_from)) & (df["date"] <= pd.Timestamp(prev_to) + pd.Timedelta(days=1))
    if region and region != "All":
        mask &= df["region"] == region
    if category and category != "All":
        mask &= df["category"] == category
    return df[mask]


def pick_fourth_kpi(df, mapping):
    if mapping.get("visits") and mapping.get("visits") != "— none —" and df["visits"].notna().any():
        return "conversion"
    if mapping.get("customer") and mapping.get("customer") != "— none —" and df["customer"].notna().any():
        return "customers"
    return "products"


def compute_kpis(df, prev_df, mapping):
    revenue = df["amount"].sum()
    orders = len(df)
    aov = revenue / orders if orders else 0
    prev_revenue = prev_df["amount"].sum()
    prev_orders = len(prev_df)
    prev_aov = prev_revenue / prev_orders if prev_orders else 0

    kind = pick_fourth_kpi(df, mapping) if st.session_state.mode == "real" else "conversion"
    if kind == "conversion" and st.session_state.mode == "real":
        visits, prev_visits = df["visits"].sum(), prev_df["visits"].sum()
        fourth = (orders / visits * 100) if visits else 0
        prev_fourth = (prev_orders / prev_visits * 100) if prev_visits else 0
        label, fmt = "Conversion Rate", lambda v: f"{v:.1f}%"
    elif kind == "customers":
        fourth = df["customer"].nunique()
        prev_fourth = prev_df["customer"].nunique()
        label, fmt = "Unique Customers", lambda v: f"{v:,.0f}"
    elif kind == "products":
        fourth = df["product"].nunique()
        prev_fourth = prev_df["product"].nunique()
        label, fmt = "Distinct Products", lambda v: f"{v:,.0f}"
    else:
        fourth = np.random.uniform(2.8, 3.6)
        prev_fourth = fourth * np.random.uniform(0.9, 1.05)
        label, fmt = "Conversion Rate", lambda v: f"{v:.1f}%"

    return dict(revenue=revenue, orders=orders, aov=aov, fourth=fourth,
                prev_revenue=prev_revenue, prev_orders=prev_orders, prev_aov=prev_aov, prev_fourth=prev_fourth,
                label=label, fmt=fmt)


def pct_delta(cur, prev):
    if not prev:
        return (100.0, "new") if cur > 0 else (0.0, "flat")
    d = (cur - prev) / prev * 100
    return d, ("up" if d >= 0 else "down")


def auto_granularity(span_days):
    if span_days <= 3:
        return "h", "Hourly"
    if span_days <= 120:
        return "D", "Daily"
    if span_days <= 730:
        return "W", "Weekly"
    return "M", "Monthly"


def build_series(df, freq):
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("date")["amount"].resample(freq).sum()


def top_n_with_other(series: pd.Series, n=8):
    s = series.sort_values(ascending=False)
    if len(s) <= n:
        return s
    top = s.iloc[:n]
    other = pd.Series({"Other": s.iloc[n:].sum()})
    return pd.concat([top, other])


# ============================================================
# INIT
# ============================================================
init_state()
if not st.session_state.demo_seed_done:
    seed_demo()

# ============================================================
# SIDEBAR — DATA SOURCE + SETTINGS
# ============================================================
with st.sidebar:
    st.markdown("### 📊 PULSE — Your Data")
    tab_upload, tab_live, tab_manual, tab_demo = st.tabs(["Upload", "Live URL", "Manual", "Demo"])

    with tab_upload:
        st.caption("CSV, Excel (.xlsx) or JSON. Columns are auto-detected — confirm the mapping below.")
        upfile = st.file_uploader("File", type=["csv", "xlsx", "xls", "json", "txt"], label_visibility="collapsed")
        pasted = st.text_area("...or paste CSV/JSON", height=90, placeholder="date,region,category,product,amount\n2026-01-15,North America,Electronics,Wireless Earbuds Pro,129.99")
        parse_clicked = st.button("Parse pasted data", width='stretch')

        # Parsed data is stashed in session_state so it survives the rerun that
        # happens when the mapping form below is submitted (button/parse_clicked
        # flags only stay True for a single run, but the form needs the data
        # to still be there on the run where "Load this data" is clicked).
        if upfile is not None:
            try:
                st.session_state.pending_upload_df = read_any(upfile, upfile.name)
            except Exception as e:
                st.error(f"Couldn't parse file: {e}")
        elif parse_clicked and pasted.strip():
            try:
                st.session_state.pending_upload_df = read_any(pasted)
            except Exception as e:
                st.error(f"Couldn't parse pasted data: {e}")

        raw_df = st.session_state.get("pending_upload_df")
        if raw_df is not None and not raw_df.empty:
            st.success(f"Parsed {len(raw_df):,} rows, {len(raw_df.columns)} columns.")
            cols = list(raw_df.columns)
            with st.form("upload_mapping_form"):
                st.caption("Map your columns")
                m_date = st.selectbox("Date / time *", cols, index=cols.index(guess_field(cols, "date")) if guess_field(cols, "date") in cols else 0)
                m_amount = st.selectbox("Amount *", ["— none —"] + cols, index=(["— none —"] + cols).index(guess_field(cols, "amount")) if guess_field(cols, "amount") in cols else 0)
                m_region = st.selectbox("Region", ["— none —"] + cols, index=(["— none —"] + cols).index(guess_field(cols, "region")) if guess_field(cols, "region") in cols else 0)
                m_category = st.selectbox("Category", ["— none —"] + cols, index=(["— none —"] + cols).index(guess_field(cols, "category")) if guess_field(cols, "category") in cols else 0)
                m_product = st.selectbox("Product", ["— none —"] + cols, index=(["— none —"] + cols).index(guess_field(cols, "product")) if guess_field(cols, "product") in cols else 0)
                with st.expander("Optional: quantity × unit price, customer, visits"):
                    m_qty = st.selectbox("Quantity", ["— none —"] + cols)
                    m_price = st.selectbox("Unit price", ["— none —"] + cols)
                    m_customer = st.selectbox("Customer", ["— none —"] + cols)
                    m_visits = st.selectbox("Visits / sessions", ["— none —"] + cols)
                append_mode = st.checkbox("Append to existing data instead of replacing")
                submitted = st.form_submit_button("✓ Load this data", width='stretch', type="primary")

            if submitted:
                mapping = {"date": m_date, "amount": m_amount, "region": m_region, "category": m_category,
                           "product": m_product, "quantity": m_qty, "unit_price": m_price,
                           "customer": m_customer, "visits": m_visits}
                norm, skipped = normalize(raw_df, mapping)
                if norm.empty:
                    st.error("No valid rows after mapping — check date/amount formats.")
                else:
                    added = ingest(norm, append_mode)
                    st.session_state.mapping = mapping
                    st.session_state.mode = "real"
                    st.session_state.source_label = f"Uploaded file ({len(st.session_state.data):,} rows)"
                    st.session_state.goal_hit = False
                    st.session_state.pending_upload_df = None
                    scan_for_outliers(st.session_state.data)
                    st.success(f"Loaded {added:,} transactions" + (f", skipped {skipped} invalid row(s)." if skipped else "."))
                    st.rerun()

        st.divider()
        template_csv = "date,region,category,product,amount\n2026-01-15,North America,Electronics,Wireless Earbuds Pro,129.99\n2026-01-15,Europe,Apparel,Denim Jacket,89.50\n2026-01-16,APAC,Beauty,Vitamin C Serum,34.00\n"
        st.download_button("⬇ Download CSV template", template_csv, file_name="pulse-data-template.csv", mime="text/csv", width='stretch')

    with tab_live:
        st.caption("Poll a live CSV/JSON URL. Runs server-side — not affected by browser CORS.")
        url_input = st.text_input("Source URL", value=st.session_state.live_url, placeholder="https://docs.google.com/…/pub?output=csv")
        poll_s = st.select_slider("Poll interval", options=[10, 30, 60, 300], value=st.session_state.live_poll_s,
                                   format_func=lambda v: f"{v}s" if v < 60 else f"{v//60}min")
        c1, c2 = st.columns(2)
        connect_clicked = c1.button("🔌 Connect", width='stretch', type="primary")
        disconnect_clicked = c2.button("Disconnect", width='stretch', disabled=not st.session_state.live_active)

        if connect_clicked and url_input.strip():
            try:
                st.session_state.pending_live_df = fetch_live_df(url_input.strip())
                st.session_state.live_url = url_input.strip()
                st.session_state.live_poll_s = poll_s
            except Exception as e:
                st.session_state.pending_live_df = None
                st.error(f"Fetch failed: {e}")

        live_raw_df = st.session_state.get("pending_live_df")
        if live_raw_df is not None and not live_raw_df.empty:
            cols = list(live_raw_df.columns)
            st.info(f"Fetched {len(live_raw_df):,} rows. Map columns, then confirm.")
            with st.form("live_mapping_form"):
                st.caption("Map your columns")
                m_date = st.selectbox("Date / time *", cols, index=cols.index(guess_field(cols, "date")) if guess_field(cols, "date") in cols else 0)
                m_amount = st.selectbox("Amount *", ["— none —"] + cols, index=(["— none —"] + cols).index(guess_field(cols, "amount")) if guess_field(cols, "amount") in cols else 0)
                m_region = st.selectbox("Region", ["— none —"] + cols, index=(["— none —"] + cols).index(guess_field(cols, "region")) if guess_field(cols, "region") in cols else 0)
                m_category = st.selectbox("Category", ["— none —"] + cols, index=(["— none —"] + cols).index(guess_field(cols, "category")) if guess_field(cols, "category") in cols else 0)
                m_product = st.selectbox("Product", ["— none —"] + cols, index=(["— none —"] + cols).index(guess_field(cols, "product")) if guess_field(cols, "product") in cols else 0)
                m_qty = st.selectbox("Quantity", ["— none —"] + cols)
                m_price = st.selectbox("Unit price", ["— none —"] + cols)
                live_submit = st.form_submit_button("✓ Load & start polling", width='stretch', type="primary")
            if live_submit:
                mapping = {"date": m_date, "amount": m_amount, "region": m_region, "category": m_category,
                           "product": m_product, "quantity": m_qty, "unit_price": m_price,
                           "customer": "— none —", "visits": "— none —"}
                norm, skipped = normalize(live_raw_df, mapping)
                if norm.empty:
                    st.error("No valid rows after mapping — check date/amount formats.")
                else:
                    ingest(norm, append=False)
                    st.session_state.mapping = mapping
                    st.session_state.mode = "real"
                    st.session_state.live_active = True
                    st.session_state.source_label = f"Live: {st.session_state.live_url}"
                    st.session_state.goal_hit = False
                    st.session_state.pending_live_df = None
                    scan_for_outliers(st.session_state.data)
                    st.rerun()

        if disconnect_clicked:
            st.session_state.live_active = False
            st.rerun()

        if st.session_state.live_active:
            st.success(f"🟢 Polling every {st.session_state.live_poll_s}s — last check {st.session_state.last_live_fetch or 'just now'}")

    with tab_manual:
        st.caption("Log one transaction at a time.")
        with st.form("manual_form", clear_on_submit=True):
            m_dt = st.date_input("Date", value=datetime.now().date())
            m_time = st.time_input("Time", value=datetime.now().time())
            m_region = st.text_input("Region", placeholder="e.g. North America")
            m_category = st.text_input("Category", placeholder="e.g. Electronics")
            m_product = st.text_input("Product *")
            m_amount = st.number_input("Amount *", min_value=0.0, step=0.01, format="%.2f")
            add_clicked = st.form_submit_button("+ Add transaction", width='stretch', type="primary")
        if add_clicked:
            if not m_product or m_amount <= 0:
                st.warning("Product and amount are required.")
            else:
                row = pd.DataFrame([{
                    "date": datetime.combine(m_dt, m_time), "region": m_region or "Unknown",
                    "category": m_category or "Uncategorized", "product": m_product,
                    "amount": m_amount, "customer": None, "visits": None,
                }])
                ingest(row, append=True)
                st.session_state.mode = "real"
                if not st.session_state.mapping:
                    st.session_state.mapping = {"date": "date", "amount": "amount", "region": "region", "category": "category", "product": "product"}
                if not st.session_state.source_label.startswith("Live"):
                    st.session_state.source_label = "Manual entries"
                check_anomaly(m_amount, m_region or "Unknown", m_product, toast=True)
                st.success("Added.")
                st.rerun()

    with tab_demo:
        st.caption("Fill the dashboard with a simulated live stream — good for testing before connecting real data.")
        if st.button("▶ Switch to demo data", width='stretch', type="primary"):
            st.session_state.mode = "demo"
            st.session_state.source_label = "Demo (simulated)"
            st.session_state.live_active = False
            seed_demo()
            st.rerun()

    st.divider()
    st.markdown("### ⚙ Settings")
    st.session_state.threshold = st.number_input("Big-order threshold ($)", min_value=10, value=st.session_state.threshold, step=10)
    st.session_state.goal = st.number_input("Revenue goal ($)", min_value=100, value=st.session_state.goal, step=100)
    st.session_state.sound = st.checkbox("Sound cue on critical alerts (browser)", value=st.session_state.sound)
    st.session_state.paused = st.checkbox("⏸ Pause live updates", value=st.session_state.paused)

    st.divider()
    st.caption(f"**Source:** {st.session_state.source_label}")
    st.caption(f"**Rows:** {len(st.session_state.data):,}")
    if not st.session_state.data.empty:
        st.caption(f"**Span:** {st.session_state.data['date'].min():%d/%m/%Y} – {st.session_state.data['date'].max():%d/%m/%Y}")
    if st.session_state.mode == "real" and not st.session_state.data.empty:
        json_bytes = st.session_state.data.to_json(orient="records", date_format="iso").encode()
        st.download_button("⬇ Export dataset (JSON)", json_bytes, file_name="pulse-dataset.json", mime="application/json", width='stretch')
    if st.button("🗑 Clear my data", width='stretch'):
        st.session_state.data = pd.DataFrame(columns=["date", "region", "category", "product", "amount", "customer", "visits"])
        st.session_state.mapping = {}
        st.session_state.seen_hashes = set()
        st.session_state.live_active = False
        st.session_state.mode = "demo"
        st.session_state.source_label = "Demo (simulated)"
        seed_demo()
        st.rerun()

# ============================================================
# AUTO-REFRESH (demo tick / live poll)
# ============================================================
if HAS_AUTOREFRESH and not st.session_state.paused:
    interval_ms = 2500 if st.session_state.mode == "demo" else st.session_state.live_poll_s * 1000
    st_autorefresh(interval=interval_ms, key="pulse_autorefresh")

if st.session_state.mode == "demo" and not st.session_state.paused:
    tick_demo()

if st.session_state.mode == "real" and st.session_state.live_active and not st.session_state.paused:
    try:
        raw_df = fetch_live_df(st.session_state.live_url)
        norm, _ = normalize(raw_df, st.session_state.mapping)
        added = ingest(norm, append=True)
        if added:
            new_rows = st.session_state.data.tail(added)
            for _, r in new_rows.iterrows():
                check_anomaly(r["amount"], r["region"], r["product"], toast=True)
        st.session_state.last_live_fetch = datetime.now().strftime("%H:%M:%S")
    except Exception as e:
        st.session_state.last_live_fetch = f"failed ({e})"

# ============================================================
# TOP BAR — FILTERS
# ============================================================
df_all = st.session_state.data.copy()
if not df_all.empty:
    df_all["date"] = pd.to_datetime(df_all["date"])

st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
  <h1 style="margin:0;">📊 PULSE <span style="font-size:14px; color:#8B96B3; font-weight:400;">Real-Time Sales Intelligence</span></h1>
  <span class="pulse-badge"><span class="pulse-dot"></span>{st.session_state.source_label}</span>
</div>
""", unsafe_allow_html=True)

f1, f2, f3, f4, f5 = st.columns([1, 1, 1, 1, 1.3])
regions = ["All"] + sorted(df_all["region"].dropna().unique().tolist()) if not df_all.empty else ["All"]
categories = ["All"] + sorted(df_all["category"].dropna().unique().tolist()) if not df_all.empty else ["All"]
region_sel = f1.selectbox("Region", regions)
category_sel = f2.selectbox("Category", categories)

if not df_all.empty:
    min_d, max_d = df_all["date"].min().date(), df_all["date"].max().date()
else:
    max_d, min_d = datetime.now().date(), datetime.now().date() - timedelta(days=30)
date_from = f3.date_input("From", value=min_d, min_value=min_d, max_value=max_d)
date_to = f4.date_input("To", value=max_d, min_value=min_d, max_value=max_d)
search = f5.text_input("Search product / region", placeholder="Search…")

gran_options = {"Auto": None, "Hourly": "h", "Daily": "D", "Weekly": "W", "Monthly": "M"}
gran_choice = st.select_slider("Chart granularity", options=list(gran_options.keys()), value="Auto")
compare_prev = st.checkbox("Compare vs previous period", value=False)

filtered = apply_filters(df_all, region_sel, category_sel, search, date_from, date_to)
prev_df = previous_period(df_all, region_sel, category_sel, date_from, date_to)

# ============================================================
# KPI ROW
# ============================================================
k = compute_kpis(filtered, prev_df, st.session_state.mapping)
c1, c2, c3, c4 = st.columns(4)
d_rev, s_rev = pct_delta(k["revenue"], k["prev_revenue"])
d_ord, s_ord = pct_delta(k["orders"], k["prev_orders"])
d_aov, s_aov = pct_delta(k["aov"], k["prev_aov"])
d_4, s_4 = pct_delta(k["fourth"], k["prev_fourth"])

c1.metric("Revenue", f"${k['revenue']:,.0f}", f"{d_rev:+.1f}%")
c2.metric("Orders", f"{k['orders']:,}", f"{d_ord:+.1f}%")
c3.metric("Avg Order Value", f"${k['aov']:,.0f}", f"{d_aov:+.1f}%", delta_color="inverse")
c4.metric(k["label"], k["fmt"](k["fourth"]), f"{d_4:+.1f}%")

# ============================================================
# GOAL BAR
# ============================================================
goal = st.session_state.goal or 1
pct = min(1.0, k["revenue"] / goal)
st.progress(pct, text=f"Revenue goal: ${k['revenue']:,.0f} of ${goal:,.0f} ({pct*100:.1f}%)")
if pct >= 1.0 and not st.session_state.goal_hit:
    st.session_state.goal_hit = True
    push_alert("good", f"Revenue goal of ${goal:,.0f} reached for the selected range")
elif pct < 1.0:
    st.session_state.goal_hit = False

# ============================================================
# CHARTS
# ============================================================
chart_col, side_col = st.columns([2, 1])

with chart_col:
    st.subheader("Revenue over time" + (" (vs previous period)" if compare_prev else ""))
    if filtered.empty:
        st.info("No transactions in this range/filter.")
    else:
        span_days = max((date_to - date_from).days, 1)
        freq = gran_options[gran_choice] or auto_granularity(span_days)[0]
        series = build_series(filtered, freq)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Current period",
                                  line=dict(color="#33E1E0", width=2), fill="tozeroy", fillcolor="rgba(51,225,224,0.15)"))
        if compare_prev and not prev_df.empty:
            pseries = build_series(prev_df, freq)
            pseries.index = pseries.index + (series.index[0] - pseries.index[0]) if len(pseries) and len(series) else pseries.index
            fig.add_trace(go.Scatter(x=series.index[:len(pseries)], y=pseries.values[:len(series)], mode="lines",
                                      name="Previous period", line=dict(color="#5A6584", width=1.5, dash="dash")))
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=10, r=10, t=10, b=10), height=320,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, width='stretch')

with side_col:
    st.subheader("Alert Log")
    if not st.session_state.alerts:
        st.caption("No alerts yet.")
    else:
        for a in st.session_state.alerts[:8]:
            cls = {"crit": "alert-crit", "warn": "alert-warn", "good": "alert-good"}[a["level"]]
            icon = {"crit": "⛔", "warn": "⚠️", "good": "✅"}[a["level"]]
            st.markdown(f'<div class="alert-row {cls}">{icon} {a["text"]}<div style="color:#5A6584; font-size:10px; margin-top:2px;">{a["time"]}</div></div>', unsafe_allow_html=True)

col_region, col_category = st.columns(2)
with col_region:
    st.subheader("Revenue by Region")
    if filtered.empty:
        st.info("No data.")
    else:
        rseries = top_n_with_other(filtered.groupby("region")["amount"].sum(), 10)
        fig_r = px.bar(x=rseries.values, y=rseries.index, orientation="h", color=rseries.index,
                        color_discrete_sequence=PALETTE)
        fig_r.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=280,
                             xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig_r, width='stretch')

with col_category:
    st.subheader("Revenue by Category")
    if filtered.empty:
        st.info("No data.")
    else:
        cseries = top_n_with_other(filtered.groupby("category")["amount"].sum(), 8)
        fig_c = go.Figure(data=[go.Pie(labels=cseries.index, values=cseries.values, hole=0.6,
                                        marker=dict(colors=PALETTE))])
        fig_c.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=10, r=10, t=10, b=10), height=280,
                             legend=dict(orientation="h", yanchor="bottom", y=-0.2))
        st.plotly_chart(fig_c, width='stretch')

# ============================================================
# LEADERBOARD
# ============================================================
st.subheader("Top Products")
if filtered.empty:
    st.info("No data in this range/filter.")
else:
    top_products = filtered.groupby("product")["amount"].sum().sort_values(ascending=False).head(6)
    fig_lb = px.bar(x=top_products.values, y=top_products.index, orientation="h",
                     color=top_products.index, color_discrete_sequence=PALETTE)
    fig_lb.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=260,
                          xaxis_title=None, yaxis_title=None, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_lb, width='stretch')

# ============================================================
# TRANSACTIONS TABLE
# ============================================================
st.subheader(f"Transactions ({len(filtered):,} matching)")
if filtered.empty:
    st.info("No transactions match the current filters.")
else:
    show_df = filtered.sort_values("date", ascending=False).copy()
    show_df["date"] = show_df["date"].dt.strftime("%d/%m/%Y %H:%M")
    show_df["amount"] = show_df["amount"].map(lambda v: f"${v:,.2f}")
    display_cols = ["date", "region", "category", "product", "amount"]
    st.dataframe(show_df[display_cols].head(500), width='stretch', height=320, hide_index=True)
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button("⬇ Export filtered transactions (CSV)", csv_bytes, file_name="pulse-transactions.csv", mime="text/csv")

st.caption("Built with Streamlit · pandas · Plotly. All processing happens in this Python process — no data leaves your machine except the optional Live URL you configure.")
