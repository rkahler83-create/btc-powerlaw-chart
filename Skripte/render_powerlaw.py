import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, LogLocator

OUTFILE = "docs/powerlaw.png"

X_START = dt.date(2011, 1, 1)
X_END   = dt.date(2040, 12, 31)

TICKER = "BTC-USD"

Y_MIN = 1
Y_MAX = 10_000_000

SUPPORT_FACTOR    = 0.20
RESISTANCE_FACTOR = 5.00

COL_PRICE = "#F7931A"
COL_REG   = "#A6CF6C"
COL_SUP   = "#D95C4A"
COL_RES   = "#8E55FF"

BG = "#0B0C0F"

def fmt_usd(x, pos):
    if x <= 0:
        return ""
    return f"{int(x):,}"

def fetch_data():
    df = yf.download(
        TICKER,
        start="2010-01-01",
        progress=False
    )
    s = df["Close"].dropna().astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s[s.index.date >= X_START]
    s = s[s > 0]
    return s

def fit_regression(price_series):
    dates = price_series.index
    t = np.array([(d.date() - X_START).days for d in dates], dtype=float)
    log_y = np.log(price_series.values)
    m, c = np.polyfit(t, log_y, 1)
    return m, c

def main():
    prices = fetch_data()
    m, c = fit_regression(prices)

    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor=BG)
    ax = fig.add_subplot(111, facecolor=BG)

    ax.set_yscale("log")

    full_dates = pd.date_range(start=X_START, end=X_END, freq="D")
    t_full = np.array([(d.date() - X_START).days for d in full_dates], dtype=float)

    reg = np.exp(m * t_full + c)
    sup = reg * SUPPORT_FACTOR
    res = reg * RESISTANCE_FACTOR

    ax.plot(full_dates, res, color=COL_RES, lw=1.6, label="Widerstand")
    ax.plot(full_dates, reg, color=COL_REG, lw=1.6, label="Regression")
    ax.plot(full_dates, sup, color=COL_SUP, lw=1.6, label="Unterstützung")
    ax.plot(prices.index, prices.values, color=COL_PRICE, lw=1.6, label="Preis (Tagesschluss)")

    ax.set_xlim(X_START, X_END)
    ax.set_ylim(Y_MIN, Y_MAX)

    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_usd))

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.tick_params(axis="x", colors="white", labelsize=9, rotation=25)
    ax.tick_params(axis="y", colors="white", labelsize=9)

    ax.grid(True, which="major", alpha=0.15)
    ax.grid(True, which="minor", alpha=0.05)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_ylabel("USD", color="white")

    legend = ax.legend(
        loc="lower right",
        frameon=True,
        facecolor=(0.1, 0.1, 0.1, 0.6),
        edgecolor=(1, 1, 1, 0.2),
        fontsize=9
    )
    for text in legend.get_texts():
        text.set_color("white")

    # 🔥 Quellenhinweis exakt zwischen 1.000.000 und 100.000 platzieren
    # Y = ca. 500.000 (log-Mitte zwischen 1e6 und 1e5)
    ax.text(
        X_START + dt.timedelta(days=120),
        500_000,
        "Quelle Kursdaten: Yahoo Finance (Ticker BTC-USD)",
        color="white",
        fontsize=9,
        ha="left",
        va="center",
        bbox=dict(
            boxstyle="square,pad=0.3",
            facecolor="none",
            edgecolor="white",
            linewidth=1.0   # dünn wie andere Linien
        )
    )

    plt.tight_layout()
    fig.savefig(OUTFILE, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
