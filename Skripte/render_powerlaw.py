# Skripte/render_powerlaw.py

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import yfinance as yf

OUTFILE = "docs/powerlaw.png"

START_DATE = "2011-01-01"
END_YEAR = 2040

Y_MIN = 0.1
Y_MAX = 10_000_000

COLOR_PRICE = "#F7931A"
COLOR_REG = "#A6CF6C"
COLOR_SUP = "#D95B43"
COLOR_RES = "#8E4EC6"

BG = "#0B0B0B"
LINE_W = 2.2

SOURCE_TEXT = "Quelle Kursdaten: Yahoo Finance (Ticker BTC-USD)"


# =========================
# Format USD Achse
# =========================
def fmt_usd(x, pos=None):
    if x >= 1:
        return f"{int(x):,}"
    return f"{x:.1f}".rstrip("0").rstrip(".")


# =========================
# Powerlaw Fit
# =========================
def fit_powerlaw(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    lx = np.log10(x)
    ly = np.log10(y)

    m, b = np.polyfit(lx, ly, 1)
    return m, b


def eval_powerlaw(x, m, b):
    return 10 ** (m * np.log10(x) + b)


# =========================
# Daten laden
# =========================
def fetch_data():
    df = yf.download(
        "BTC-USD",
        start=START_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise RuntimeError("Keine Kursdaten von Yahoo erhalten.")

    close = df["Close"].dropna()
    close.index = pd.to_datetime(close.index)

    return close.index.to_pydatetime(), close.values.astype(float)


# =========================
# MAIN
# =========================
def main():
    dates_hist, price_hist = fetch_data()

    start_dt = dt.datetime.fromisoformat(START_DATE)
    x_hist = np.array([(d - start_dt).days for d in dates_hist])
    x_hist[x_hist < 1] = 1

    # Regression
    m_reg, b_reg = fit_powerlaw(x_hist, price_hist)
    y_reg_hist = eval_powerlaw(x_hist, m_reg, b_reg)

    resid = np.log10(price_hist) - np.log10(y_reg_hist)

    low_mask = resid <= np.quantile(resid, 0.20)
    high_mask = resid >= np.quantile(resid, 0.80)

    m_sup, b_sup = fit_powerlaw(x_hist[low_mask], price_hist[low_mask])
    m_res, b_res = fit_powerlaw(x_hist[high_mask], price_hist[high_mask])

    end_dt = dt.datetime(END_YEAR, 12, 31)
    all_dates = pd.date_range(start=start_dt, end=end_dt, freq="D").to_pydatetime()
    x_all = np.array([(d - start_dt).days for d in all_dates])
    x_all[x_all < 1] = 1

    y_reg = eval_powerlaw(x_all, m_reg, b_reg)
    y_sup = eval_powerlaw(x_all, m_sup, b_sup)
    y_res = eval_powerlaw(x_all, m_res, b_res)

    # ================= Plot =================
    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)

    ax.plot(all_dates, y_res, color=COLOR_RES, lw=LINE_W, label="Widerstand")
    ax.plot(all_dates, y_reg, color=COLOR_REG, lw=LINE_W, label="Regression")
    ax.plot(all_dates, y_sup, color=COLOR_SUP, lw=LINE_W, label="Unterstützung")
    ax.plot(dates_hist, price_hist, color=COLOR_PRICE, lw=1.8, alpha=0.9, label="Preis (Tagesschluss)")

    ax.set_yscale("log")
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_usd))

    ax.set_xlim(dt.datetime(2011, 1, 1), dt.datetime(END_YEAR, 12, 31))
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for lbl in ax.get_xticklabels():
        lbl.set_rotation(28)
        lbl.set_ha("right")
        lbl.set_fontsize(8)
        lbl.set_color("#D0D0D0")

    ax.grid(True, which="major", color="white", alpha=0.18, lw=0.8)
    ax.grid(True, which="minor", color="white", alpha=0.06, lw=0.5)

    ax.set_ylabel("USD", color="#BEBEBE")
    ax.tick_params(axis="y", colors="#D0D0D0")

    leg = ax.legend(loc="lower right", frameon=True,
                    facecolor="#0F0F0F", edgecolor="#2A2A2A")

    for t in leg.get_texts():
        t.set_color("#EAEAEA")

    ax.text(
        0.045, 0.93,
        SOURCE_TEXT,
        transform=ax.transAxes,
        color="#FFFFFF",
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="square,pad=0.35",
            facecolor="none",
            edgecolor="#FFFFFF",
            linewidth=1.0
        )
    )

    plt.tight_layout(pad=1.2)
    fig.savefig(OUTFILE, bbox_inches="tight", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
