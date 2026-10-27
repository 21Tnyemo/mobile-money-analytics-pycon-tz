"""
Reading Mobile Money Data: Beginner Analytics with Pandas and Matplotlib
PyCon Tanzania 2026 - talk prototype

Run:  python mobile_money_demo.py
Output: raw_transactions.csv, clean_transactions.csv, charts/*.png

All data is SYNTHETIC. No real customer or provider data is used.
Sections marked "# %%" can be pasted into Jupyter cells one by one.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent
CHARTS = OUT / "charts"
CHARTS.mkdir(exist_ok=True)

# %% PHASE 0 - Generate a messy synthetic dataset -------------------------
rng = np.random.default_rng(seed=2026)
N = 6000

START, DAYS = pd.Timestamp("2026-01-01"), 120
day_offsets = rng.integers(0, DAYS, N)
# Activity peaks around lunchtime and early evening
weights = np.array([1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 10, 12, 10, 8, 8, 9, 11, 12, 9, 6, 4, 2, 1])
hours = rng.choice(24, size=N, p=weights / weights.sum())
minutes = rng.integers(0, 60, N)
ts = (START + pd.to_timedelta(day_offsets, unit="D")
      + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m"))

types = rng.choice(
    ["P2P Transfer", "Cash Out", "Cash In", "Bill Payment", "Airtime", "Merchant Payment"],
    size=N, p=[0.28, 0.18, 0.14, 0.14, 0.14, 0.12],
)
networks = rng.choice(["M-Pesa", "Tigo Pesa", "Airtel Money"], size=N, p=[0.5, 0.3, 0.2])

base = {"P2P Transfer": 25000, "Cash Out": 60000, "Cash In": 80000,
        "Bill Payment": 45000, "Airtime": 3000, "Merchant Payment": 15000}
amount = np.array([rng.lognormal(np.log(base[t]), 0.6) for t in types]).round(-2)
fee = np.where(types == "Cash Out", amount * 0.02, np.where(types == "P2P Transfer", amount * 0.005, 0)).round(-1)

raw = pd.DataFrame({
    "txn_id": [f"TX{100000 + i}" for i in range(N)],
    "timestamp": ts.strftime("%d/%m/%Y %H:%M"),
    "network": networks,
    "txn_type": types,
    "amount_tzs": amount,
    "fee_tzs": fee,
})

# Make it realistically messy
raw.loc[rng.choice(N, 150, replace=False), "amount_tzs"] = np.nan   # missing amounts
raw.loc[rng.choice(N, 120, replace=False), "network"] = np.nan      # missing network
case_idx = rng.choice(N, 200, replace=False)
raw.loc[case_idx, "txn_type"] = raw.loc[case_idx, "txn_type"].str.upper()  # inconsistent case
raw = pd.concat([raw, raw.sample(80, random_state=1)], ignore_index=True)  # duplicates
raw.to_csv(OUT / "raw_transactions.csv", index=False)

# %% PHASE 1 - Data wrangling with Pandas ---------------------------------
df = pd.read_csv(OUT / "raw_transactions.csv")

print("=== Inspect ===")
print(df.shape)
print(df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%Y %H:%M")
df["txn_type"] = df["txn_type"].str.title().str.replace("P2P", "P2P")  # fix inconsistent case
df["network"] = df["network"].fillna("Unknown")
df = df.dropna(subset=["amount_tzs"])

df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.day_name()
df.to_csv(OUT / "clean_transactions.csv", index=False)
print(f"\nClean rows: {len(df):,}")

# %% PHASE 2 - Aggregation ------------------------------------------------
type_counts = df["txn_type"].value_counts()
avg_cashout_fee = df.loc[df["txn_type"] == "Cash Out", "fee_tzs"].mean()
peak_hour = df.groupby("hour").size().idxmax()
daily = df.groupby("date")["amount_tzs"].sum()
daily.index = pd.to_datetime(daily.index)
by_type = df.groupby("txn_type")["amount_tzs"].sum().sort_values()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heat = (df.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
        .reindex(order).reindex(columns=range(24), fill_value=0))

print("\n=== Key metrics ===")
print(type_counts)
print(f"Average cash-out fee: TZS {avg_cashout_fee:,.0f}")
print(f"Peak transaction hour: {peak_hour}:00")

# %% PHASE 3 - Visualization with Matplotlib ------------------------------
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
GREEN, BLUE = "#1f4d2e", "#2f6fbf"

# Chart 1: trend over time (7-day rolling average)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(daily.index, daily / 1e6, color="#b9c9bd", lw=1, label="Daily total")
ax.plot(daily.index, daily.rolling(7).mean() / 1e6, color=GREEN, lw=2.5, label="7-day average")
ax.set(title="Daily transaction value over time", xlabel="Date", ylabel="Value (million TZS)")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(CHARTS / "1_daily_trend.png", dpi=150); plt.close(fig)

# Chart 2: value by transaction type
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(by_type.index, by_type / 1e6, color=BLUE)
ax.bar_label(bars, fmt="%.0f", padding=3)
ax.set(title="Total value by transaction type", xlabel="Value (million TZS)")
fig.tight_layout(); fig.savefig(CHARTS / "2_value_by_type.png", dpi=150); plt.close(fig)

# Chart 3: peak hours
hourly = df.groupby("hour").size()
fig, ax = plt.subplots(figsize=(9, 4.5))
colors = [GREEN if h == peak_hour else "#b9c9bd" for h in hourly.index]
ax.bar(hourly.index, hourly.values, color=colors)
ax.set(title=f"Transactions by hour of day (peak: {peak_hour}:00)",
       xlabel="Hour of day", ylabel="Number of transactions", xticks=range(0, 24, 2))
fig.tight_layout(); fig.savefig(CHARTS / "3_peak_hours.png", dpi=150); plt.close(fig)

# Chart 4: weekday x hour heatmap
fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(heat.values, aspect="auto", cmap="Greens")
ax.set(title="When do people transact? (weekday by hour)", xlabel="Hour of day",
       xticks=range(0, 24, 2), yticks=range(7))
ax.set_yticklabels(order)
fig.colorbar(im, ax=ax, label="Transactions")
fig.tight_layout(); fig.savefig(CHARTS / "4_weekday_hour_heatmap.png", dpi=150); plt.close(fig)

print("\nDone. Charts saved in:", CHARTS)
