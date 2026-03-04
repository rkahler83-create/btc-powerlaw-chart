import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

OUTFILE = "docs/powerlaw.png"

START_DATE = "2011-01-01"
END_YEAR = 2040
GENESIS = dt.date(2009, 1, 3)

Y_MIN = 0.1
Y_MAX = 10_000_000

C_PRICE = "#F7930A"
C_REG   = "#8BC34A"
C_SUP   = "#D04B3F"
C_RES   = "#7E3FBF"

LW_PRICE = 2.0
LW_LINE  = 2.2


def days_since_genesis(d):
    return max(1, (d - GENESIS).days)


def fetch_btc():
    df = yf.download(
        "BTC-USD",
        start=START_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        raise RuntimeError("Keine Daten von Yahoo Finance erhalten.")

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna()
    close = close[close > 0]

    return close


def fit_line(x, y):
    lx = np.log10(x)
    ly = np.log10(y)
    b, a = np.polyfit(lx, ly, 1)
    return a, b


def eval_line(a, b, x):
    return 10 ** (a + b * np.log10(x))


def main():
    close = fetch_btc()

    dates = close.index.date
    x_hist = np.array([days_since_genesis(d) for d in dates], dtype=float)
    y_hist = close.values.astype(float)

    x_start = days_since_genesis(dt.date(2011, 1, 1))
    x_end = days_since_genesis(dt.date(END_YEAR, 12, 31))

    mask = (x_hist >= x_start) & (x_hist <= x_end)
    x_fit = x_hist[mask]
    y_fit = y_hist[mask]

    a_mid, b_mid = fit_line(x_fit, y_fit)
    y_mid = eval_line(a_mid, b_mid, x_fit)

    resid = np.log10(y_fit) - np.log10(y_mid)

    sup_mask = resid <= np.quantile(resid, 0.20)
    res_mask = resid >= np.quantile(resid, 0.80)

    a_sup, b_sup = fit_line(x_fit[sup_mask], y_fit[sup_mask])
    a_res, b_res = fit_line(x_fit[res_mask], y_fit[res_mask])

    x_line = np.geomspace(x_start, x_end, 700)

    y_reg = eval_line(a_mid, b_mid, x_line)
    y_sup = eval_line(a_sup, b_sup, x_line)
    y_res = eval_line(a_res, b_res, x_line)

    plt.close("all")
    fig = plt.figure(figsize=(14, 8), dpi=150)
    ax = fig.add_subplot(111)

    fig.patch.set_facecolor("#0b0c10")
    ax.set_facecolor("#0b0c10")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(x_line, y_res, color=C_RES, linewidth=LW_LINE)
    ax.plot(x_line, y_reg, color=C_REG, linewidth=LW_LINE)
    ax.plot(x_line, y_sup, color=C_SUP, linewidth=LW_LINE)

    ax.plot(x_hist, y_hist, color=C_PRICE, linewidth=LW_PRICE)

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(Y_MIN, Y_MAX)

    years = range(2011, END_YEAR + 1)
    x_ticks = [days_since_genesis(dt.date(y, 1, 1)) for y in years]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(y) for y in years], rotation=30, ha="right")

    y_ticks = [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{int(y):,}" if y >= 1 else "0.1" for y in y_ticks])

    ax.text(
        0.06, 0.92,
        "Quelle Kursdaten: Yahoo Finance (Ticker BTC-USD)",
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        bbox=dict(facecolor="none", edgecolor="white", linewidth=1.0)
    )

    plt.tight_layout()
    fig.savefig(OUTFILE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
