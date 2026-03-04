# Skripte/render_powerlaw.py

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf

OUTFILE = "docs/powerlaw.png"

# ==============================
# EINSTELLUNGEN
# ==============================
START_DATE = dt.date(2011, 1, 1)
END_DATE   = dt.date(2040, 1, 1)
GENESIS    = dt.date(2009, 1, 3)

Y_MIN = 1
Y_MAX = 30_000_000

COL_PRICE = "#F7930A"   # BTC Orange
COL_REG   = "#9EDC7A"
COL_SUP   = "#D65A4A"
COL_RES   = "#8E5BFF"

K_CORRIDOR = 1.25   # kleiner = engerer Widerstand

BG = "#0B0C0F"

# ==============================
# Hilfsfunktionen
# ==============================
def days_since_genesis(d):
    return max(1, (d - GENESIS).days)

def fmt_y(v, _p=None):
    if v < 1:
        return ""
    return f"{int(v):,}"

# ==============================
# Daten laden
# ==============================
df = yf.download(
    "BTC-USD",
    start=START_DATE.isoformat(),
    end=(dt.date.today() + dt.timedelta(days=1)).isoformat(),
    interval="1d",
    progress=False
)

df = df[["Close"]].dropna()
df.index = pd.to_datetime(df.index).tz_localize(None)

# tägliche Serie erzwingen (damit Kurs ganz links startet)
full_idx = pd.date_range(start=START_DATE, end=df.index.max(), freq="D")
series = df["Close"].reindex(full_idx)
series = series.bfill().ffill()
series = series[series > 0]

dates = series.index.date
x_days = np.array([days_since_genesis(d) for d in dates], dtype=float)
y_price = series.values.astype(float)

# ==============================
# Power Law Fit (log-log)
# ==============================
logx = np.log10(x_days)
logy = np.log10(y_price)

b, a = np.polyfit(logx, logy, 1)

def model(days):
    return (10 ** a) * (days ** b)

pred_logy = a + b * logx
resid = logy - pred_logy
sigma = np.std(resid)

support_factor = 10 ** (-K_CORRIDOR * sigma)
resist_factor  = 10 ** ( K_CORRIDOR * sigma)

x_min = days_since_genesis(START_DATE)
x_max = days_since_genesis(END_DATE)

x_line = np.logspace(np.log10(x_min), np.log10(x_max), 1500)
y_reg  = model(x_line)
y_sup  = y_reg * support_factor
y_res  = y_reg * resist_factor

# ==============================
# Plot
# ==============================
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

ax.plot(x_days, y_price, color=COL_PRICE, linewidth=2, label="Preis (Tagesschluss)")
ax.plot(x_line, y_reg,  color=COL_REG, linewidth=2, label="Regression")
ax.plot(x_line, y_sup,  color=COL_SUP, linewidth=2, label="Unterstützung")
ax.plot(x_line, y_res,  color=COL_RES, linewidth=2, label="Widerstand")

ax.set_xlim(x_min, x_max)
ax.set_ylim(Y_MIN, Y_MAX)

ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_y))
ax.set_ylabel("USD")

# Jahres-Ticks (nicht übereinander)
years = list(range(2011, 2027)) + list(range(2028, 2041, 2))
xticks = [days_since_genesis(dt.date(y, 1, 1)) for y in years]
ax.set_xticks(xticks)
ax.set_xticklabels([str(y) for y in years], rotation=30, ha="right")

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
