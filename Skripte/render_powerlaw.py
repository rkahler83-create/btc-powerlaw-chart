# Skripte/render_powerlaw.py

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf

OUTFILE = "docs/powerlaw.png"

START_DATE = dt.date(2011, 1, 1)
END_DATE   = dt.date(2040, 1, 1)
GENESIS    = dt.date(2009, 1, 3)

Y_MIN = 1
Y_MAX = 30_000_000

COL_PRICE = "#F7930A"
COL_REG   = "#9EDC7A"
COL_SUP   = "#D65A4A"
COL_RES   = "#8E5BFF"

BG = "#0B0C0F"

def days_since_genesis(d):
    return max(1, (d - GENESIS).days)

def fmt_y(v, _):
    if v < 1:
        return ""
    return f"{int(v):,}"

# ------------------------------
# Daten laden (Yahoo)
# ------------------------------
df = yf.download(
    "BTC-USD",
    start=START_DATE.isoformat(),
    end=(dt.date.today() + dt.timedelta(days=1)).isoformat(),
    interval="1d",
    progress=False
)

df = df[["Close"]].dropna()
df.index = pd.to_datetime(df.index).tz_localize(None)

# tägliche Serie erzwingen
full_idx = pd.date_range(start=START_DATE, end=df.index.max(), freq="D")
series = df["Close"].reindex(full_idx)
series = series.bfill().ffill()
series = series[series > 0]

dates = series.index.date
x_hist = np.array([days_since_genesis(d) for d in dates], dtype=float)
y_hist = series.values.astype(float)

# ------------------------------
# Power Law Regression (log-log)
# ------------------------------
logx = np.log10(x_hist)
logy = np.log10(y_hist)

b, a = np.polyfit(logx, logy, 1)

def model(a, b, x):
    return (10 ** a) * (x ** b)

# Residuen
pred_logy = a + b * logx
resid = logy - pred_logy

# getrennte Fits für Support / Resistance (nicht parallel)
low_mask  = resid <= np.quantile(resid, 0.20)
high_mask = resid >= np.quantile(resid, 0.80)

b_sup, a_sup = np.polyfit(np.log10(x_hist[low_mask]),  np.log10(y_hist[low_mask]),  1)
b_res, a_res = np.polyfit(np.log10(x_hist[high_mask]), np.log10(y_hist[high_mask]), 1)

x_min = days_since_genesis(START_DATE)
x_max = days_since_genesis(END_DATE)

x_line = np.logspace(np.log10(x_min), np.log10(x_max), 1600)

y_reg = model(a,     b,     x_line)
y_sup = model(a_sup, b_sup, x_line)
y_res = model(a_res, b_res, x_line)

# ------------------------------
# Plot
# ------------------------------
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.labelcolor": "#CCCCCC",
    "xtick.color": "#CCCCCC",
    "ytick.color": "#CCCCCC",
    "text.color": "#EAEAEA",
    "font.size": 11
})

fig = plt.figure(figsize=(16, 9), dpi=200)
ax = fig.add_subplot(111)

ax.set_xscale("log")
ax.set_yscale("log")

ax.grid(which="major", alpha=0.18)
ax.grid(which="minor", alpha=0.05)

ax.plot(x_hist, y_hist, color=COL_PRICE, linewidth=2, label="Preis (Tagesschluss)")
ax.plot(x_line, y_reg, color=COL_REG, linewidth=2, label="Regression")
ax.plot(x_line, y_sup, color=COL_SUP, linewidth=2, label="Unterstützung")
ax.plot(x_line, y_res, color=COL_RES, linewidth=2, label="Widerstand")

ax.set_xlim(x_min, x_max)
ax.set_ylim(Y_MIN, Y_MAX)

ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_y))
ax.set_ylabel("USD")

# Alle Jahre 2011–2040, schräg damit nichts überlappt
years = list(range(2011, 2041))
xticks = [days_since_genesis(dt.date(y, 1, 1)) for y in years]
ax.set_xticks(xticks)
ax.set_xticklabels([str(y) for y in years], rotation=60, ha="right", fontsize=8)

plt.subplots_adjust(bottom=0.22)

leg = ax.legend(loc="lower right", frameon=True)
leg.get_frame().set_facecolor(BG)
leg.get_frame().set_edgecolor("white")
leg.get_frame().set_alpha(0.3)

# Quellenbox
ax.text(
    0.08, 0.93,
    "Quelle Kursdaten: Yahoo Finance (Ticker BTC-USD)",
    transform=ax.transAxes,
    ha="left",
    va="top",
    bbox=dict(
        boxstyle="square,pad=0.35",
        facecolor=(0,0,0,0),
        edgecolor="white",
        linewidth=0.8
    )
)

plt.tight_layout()
fig.savefig(OUTFILE, bbox_inches="tight")
plt.close(fig)
