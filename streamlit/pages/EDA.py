# pages/EDA.py
import os
import pandas as pd
import streamlit as st
import altair as alt
import duckdb

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
GOLD_DB = os.getenv("GOLD_DB_PATH", "/data/gold/gold.duckdb")
GOLD_TBL = os.getenv("GOLD_TABLE", 'gold."gold"."crashes"')

# ---------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------
# --- replace your load_gold() and the preprocessing block with this ---

WANTED_COLS = [
    "crash_record_id",
    "crash_date",
    "crash_type",
    "posted_speed_limit",
    "lighting_condition",
    "roadway_surface_cond",
    "weather_condition",
    "veh_count",
]

@st.cache_data(ttl=60)
def load_gold():
    if not os.path.exists(GOLD_DB):
        return pd.DataFrame()

    con = duckdb.connect(GOLD_DB, read_only=True)
    try:
        # discover available columns
        info = con.execute(f"PRAGMA table_info({GOLD_TBL})").fetchdf()
        have = set(info["name"].str.lower().tolist())

        # keep only columns that exist
        cols = [c for c in WANTED_COLS if c.lower() in have]
        if not cols:
            return pd.DataFrame()

        select_list = ", ".join(cols)
        df = con.execute(f"SELECT {select_list} FROM {GOLD_TBL}").fetchdf()
    finally:
        con.close()

    # Derive time features from crash_date if present
    if "crash_date" in df.columns:
        df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
        df["crash_hour"] = df["crash_date"].dt.hour
        df["crash_day_of_week"] = df["crash_date"].dt.day_name()
        df["crash_month"] = df["crash_date"].dt.month_name()

    # Ensure binary target is numeric {0,1}
    if "crash_type" in df.columns:
        # already 0/1? keep; else try to map strings
        if df["crash_type"].dtype == object:
            m = {
                "injury and / or tow due to crash": 1,
                "no injury / drive away": 0,
                "1": 1, "0": 0, "yes": 1, "no": 0, "y": 1, "n": 0,
            }
            df["crash_type"] = (
                df["crash_type"].astype(str).str.strip().str.lower().map(m)
            )
        df["crash_type"] = pd.to_numeric(df["crash_type"], errors="coerce")

    # Normalize a few categoricals if present
    for c in ["lighting_condition", "roadway_surface_cond", "weather_condition"]:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype("string")
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
            )

    return df



# ---------------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="📊 EDA – Crash Type (Injury/Tow)", layout="wide")
st.title("📊 Exploratory Data Analysis — Crash Type (Injury/Tow)")
st.caption("Analyzing relationships between crash context and injury/tow likelihood.")

df = load_gold()
if df.empty:
    st.warning("No data found in gold.duckdb — make sure the Cleaner stage ran successfully.")
    st.stop()

# Preprocessing
df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
if "crash_hour" not in df:
    df["crash_hour"] = df["crash_date"].dt.hour
if "crash_day_of_week" not in df:
    df["crash_day_of_week"] = df["crash_date"].dt.day_name()
if "crash_month" not in df:
    df["crash_month"] = df["crash_date"].dt.month_name()

# ---------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------
st.subheader("📈 Summary Overview")

total_rows = len(df)
injuries = (df["crash_type"] == 1).sum()
noninj = total_rows - injuries
pos_pct = injuries / total_rows * 100
neg_pct = 100 - pos_pct

st.markdown(
    f"""
**Total Rows:** {total_rows:,}  
**Positives (Injury/Tow):** {injuries:,} ({pos_pct:.1f}%)  
**Negatives (No Injury/Tow):** {noninj:,} ({neg_pct:.1f}%)  
**Ratio:** ~1:{round(noninj/injuries, 1)}
"""
)

st.divider()

# ---------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------

# 1️⃣ Histogram: Speed distribution by crash type
st.subheader("1️⃣ Posted Speed Limit Distribution by Crash Type")
chart1 = (
    alt.Chart(df.dropna(subset=["posted_speed_limit"]))
    .mark_bar(opacity=0.8)
    .encode(
        x=alt.X("posted_speed_limit:Q", bin=alt.Bin(maxbins=20), title="Speed Limit (mph)"),
        y=alt.Y("count()", title="Crashes"),
        color=alt.Color("crash_type:N", legend=alt.Legend(title="Injury/Tow (1=Yes)")),
        tooltip=["posted_speed_limit", "crash_type", "count()"]
    )
    .properties(height=350)
)
st.altair_chart(chart1, use_container_width=True)

# 2️⃣ Bar: Lighting condition vs injury rate
st.subheader("2️⃣ Injury/Tow Rate by Lighting Condition")
light = df.groupby("lighting_condition")["crash_type"].mean().reset_index()
chart2 = (
    alt.Chart(light)
    .mark_bar()
    .encode(
        x=alt.X("lighting_condition:N", sort="-y", title="Lighting Condition"),
        y=alt.Y("crash_type:Q", title="Injury/Tow Rate"),
        color=alt.Color("crash_type:Q", scale=alt.Scale(scheme="reds")),
        tooltip=["lighting_condition", alt.Tooltip("crash_type:Q", format=".1%")]
    )
    .properties(height=350)
)
st.altair_chart(chart2, use_container_width=True)

# 3️⃣ Bar: Roadway surface vs injury rate
st.subheader("3️⃣ Roadway Surface Condition vs Injury/Tow Rate")
surf = df.groupby("roadway_surface_cond")["crash_type"].mean().reset_index()
chart3 = (
    alt.Chart(surf)
    .mark_bar()
    .encode(
        x=alt.X("roadway_surface_cond:N", sort="-y", title="Surface Condition"),
        y=alt.Y("crash_type:Q", title="Injury/Tow Rate"),
        color=alt.Color("crash_type:Q", scale=alt.Scale(scheme="orangered")),
        tooltip=["roadway_surface_cond", alt.Tooltip("crash_type:Q", format=".1%")]
    )
    .properties(height=350)
)
st.altair_chart(chart3, use_container_width=True)

# 4️⃣ Bar: Weather condition vs injury rate
st.subheader("4️⃣ Weather Condition vs Injury/Tow Rate")
weather = df.groupby("weather_condition")["crash_type"].mean().reset_index()
chart4 = (
    alt.Chart(weather)
    .mark_bar()
    .encode(
        x=alt.X("weather_condition:N", sort="-y", title="Weather Condition"),
        y=alt.Y("crash_type:Q", title="Injury/Tow Rate"),
        color=alt.Color("crash_type:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["weather_condition", alt.Tooltip("crash_type:Q", format=".1%")]
    )
    .properties(height=350)
)
st.altair_chart(chart4, use_container_width=True)

# 5️⃣ Line: Monthly injury/tow trend
st.subheader("5️⃣ Injury/Tow Rate Over Time (Monthly)")
trend = (
    df.dropna(subset=["crash_date"])
    .groupby(pd.Grouper(key="crash_date", freq="M"))["crash_type"]
    .mean()
    .reset_index()
)
chart5 = (
    alt.Chart(trend)
    .mark_line(point=True)
    .encode(
        x=alt.X("crash_date:T", title="Month"),
        y=alt.Y("crash_type:Q", title="Injury/Tow Rate"),
        tooltip=["crash_date", alt.Tooltip("crash_type:Q", format=".1%")],
    )
    .properties(height=350)
)
st.altair_chart(chart5, use_container_width=True)

# 6️⃣ Pie chart: Injury share by day of week
st.subheader("6️⃣ Injury/Tow Share by Day of Week")
day_df = df.groupby("crash_day_of_week")["crash_type"].mean().reset_index()
chart6 = (
    alt.Chart(day_df)
    .mark_arc(innerRadius=50)
    .encode(
        theta=alt.Theta("crash_type:Q", stack=True, title="Rate"),
        color=alt.Color("crash_day_of_week:N", legend=alt.Legend(title="Day of Week")),
        tooltip=["crash_day_of_week", alt.Tooltip("crash_type:Q", format=".1%")],
    )
)
st.altair_chart(chart6, use_container_width=True)

# 7️⃣ Bar: Crash count by day of week
st.subheader("7️⃣ Crash Counts by Day of Week (Volume)")
day_ct = df.groupby("crash_day_of_week").size().reset_index(name="count")
chart7 = (
    alt.Chart(day_ct)
    .mark_bar()
    .encode(
        x=alt.X("crash_day_of_week:N", sort=list(day_ct["crash_day_of_week"]), title="Day of Week"),
        y=alt.Y("count:Q", title="Crash Count"),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=["crash_day_of_week", "count"]
    )
)
st.altair_chart(chart7, use_container_width=True)

# 8️⃣ Line: Hourly pattern (by crash_type)
st.subheader("8️⃣ Hourly Pattern of Crashes")
hour = (
    df.dropna(subset=["crash_hour"])
    .groupby("crash_hour")["crash_type"]
    .mean()
    .reset_index()
)
chart8 = (
    alt.Chart(hour)
    .mark_line(point=True)
    .encode(
        x=alt.X("crash_hour:O", title="Hour of Day"),
        y=alt.Y("crash_type:Q", title="Injury/Tow Rate"),
        color=alt.Color("crash_type:Q", scale=alt.Scale(scheme="redyellowblue")),
        tooltip=["crash_hour", alt.Tooltip("crash_type:Q", format=".1%")],
    )
)
st.altair_chart(chart8, use_container_width=True)

# 9️⃣ Heatmap: Day of Week × Hour
st.subheader("9️⃣ Heatmap — Injury/Tow Rate by Hour and Day")
heat = (
    df.dropna(subset=["crash_hour", "crash_day_of_week"])
    .groupby(["crash_day_of_week", "crash_hour"])["crash_type"]
    .mean()
    .reset_index()
)
chart9 = (
    alt.Chart(heat)
    .mark_rect()
    .encode(
        x=alt.X("crash_hour:O", title="Hour of Day"),
        y=alt.Y("crash_day_of_week:N", title="Day of Week"),
        color=alt.Color("crash_type:Q", scale=alt.Scale(scheme="reds")),
        tooltip=["crash_day_of_week", "crash_hour", alt.Tooltip("crash_type:Q", format=".1%")],
    )
    .properties(height=400)
)
st.altair_chart(chart9, use_container_width=True)

# 🔟 Histogram: Crash frequency by month
st.subheader("🔟 Monthly Crash Counts")
month_ct = df.groupby("crash_month").size().reset_index(name="count")
chart10 = (
    alt.Chart(month_ct)
    .mark_bar()
    .encode(
        x=alt.X("crash_month:N", sort=list(month_ct["crash_month"]), title="Month"),
        y=alt.Y("count:Q", title="Number of Crashes"),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="tealblues")),
        tooltip=["crash_month", "count"]
    )
)
st.altair_chart(chart10, use_container_width=True)

st.caption("End of EDA — ten visuals exploring key patterns for injury/tow crashes.")
