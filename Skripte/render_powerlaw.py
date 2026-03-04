# Skripte/render_powerlaw.py

import datetime as dt
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTFILE = "docs/powerlaw.png"

START_DATE = dt.date(2011, 1, 1)
END_DATE = dt.date(2040, 12, 31)
GENESIS = dt.date(2009, 1, 3)

Y_MIN_FLOOR = 0.1
Y_MAX = 10_000_000

COL_PRICE = "#F7930A"
COL_REG   = "#7FC64B"
COL_SUP   = "#D04B3F"
COL_RES   = "#7E3FBF"

BG = "#0B0C10"

GRID_MAJOR_A = 0.20
GRID_MINOR_A = 0.07

LW_PRICE = 2.2
LW_LINE  = 2.2

Q_SUP = 0.20
Q_RES = 0.80

SOURCE_TEXT = "Quelldaten: Yahoo Finance (Ticker BTC-USD)"


def days_since_genesis(d: dt.date) -> int:
    return max(1, (d - GENESIS).days)


def fetch_yahoo(start: dt.date, end: dt.date) -> pd.Series:

    df = yf.download(
        "BTC-USD",
        start=start.isoformat(),
        end=(end + dt.timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        return pd.Series(dtype=float)

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna().astype(float)
    close = close[close > 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)

    return close


def fetch_cryptocompare(start: dt.date, end: dt.date) -> pd.Series:

    all_rows = []

    to_ts = int(dt.datetime(end.year, end.month, end.day).timestamp())

    while True:

        url = (
            "https://min-api.cryptocompare.com/data/v2/histoday"
            f"?fsym=BTC&tsym=USD&limit=2000&toTs={to_ts}"
        )

        r = requests.get(url, timeout=45)

        if r.status_code == 429:
            time.sleep(2)
            continue

        r.raise_for_status()

        j = r.json()

        if j.get("Response") != "Success":
            break

        data = j["Data"]["Data"]

        if not data:
            break

        for row in data:

            d = dt.datetime.utcfromtimestamp(row["time"]).date()

            if d < start or d > end:
                continue

            c = row.get("close")

            if c is None or c <= 0:
                continue

            all_rows.append((pd.Timestamp(d), float(c)))

        earliest = dt.datetime.utcfromtimestamp(data[0]["time"]).date()

        if earliest <= start:
            break

        to_ts = int(dt.datetime(earliest.year, earliest.month, earliest.day).timestamp()) - 1

    if not all_rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(all_rows, columns=["Date", "Close"]).drop_duplicates("Date")

    df = df.set_index("Date").sort_index()

    s = df["Close"].astype(float)

    s.index = pd.to_datetime(s.index).tz_localize(None)

    return s[s > 0]


def build_close_series(start: dt.date, end: dt.date) -> pd.Series:

    cc = fetch_cryptocompare(start, end)
    yh = fetch_yahoo(start, end)

    if cc.empty and yh.empty:
        raise RuntimeError("Keine Kursdaten verfügbar")

    if cc.empty:
        s = yh.copy()
    else:
        s = cc.copy()

        if not yh.empty:
            s = s.combine_first(yh)
            s.update(yh)

    full_idx = pd.date_range(start=start, end=end, freq="D")

    s = s.reindex(full_idx).bfill().ffill()

    return s[s > 0]


def fit_powerlaw(x: np.ndarray, y: np.ndarray):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lx = np.log10(x)
    ly = np.log10(y)

    b, a = np.polyfit(lx, ly, 1)

    return a, b


def eval_powerlaw(a, b, x):

    x = np.asarray(x, dtype=float)

    return 10 ** (a + b * np.log10(x))


def fmt_y(v, _):

    if v < 1:
        return f"{v:g}"

    return f"{int(v):,}"


def main():

    today = dt.date.today()

    close = build_close_series(START_DATE, today)

    dates = close.index.date

    x_hist = np.array([days_since_genesis(d) for d in dates], dtype=float)
    y_hist = close.to_numpy(dtype=float)

    x_min = float(days_since_genesis(START_DATE))
    x_max = float(days_since_genesis(END_DATE))

    early = close.iloc[:365]

    y_min = max(Y_MIN_FLOOR, float(early.min()) * 0.6)

    a_mid, b_mid = fit_powerlaw(x_hist, y_hist)

    y_mid_hist = eval_powerlaw(a_mid, b_mid, x_hist)

    resid = np.log10(y_hist) - np.log10(y_mid_hist)

    q_sup = np.quantile(resid, Q_SUP)
    q_res = np.quantile(resid, Q_RES)

    sup_mask = resid <= q_sup
    res_mask = resid >= q_res

    a_sup, b_sup = fit_powerlaw(x_hist[sup_mask], y_hist[sup_mask])
    a_res, b_res = fit_powerlaw(x_hist[res_mask], y_hist[res_mask])

    x_line = np.geomspace(x_min, x_max, 1100)

    y_reg = eval_powerlaw(a_mid, b_mid, x_line)
    y_sup = eval_powerlaw(a_sup, b_sup, x_line)
    y_res = eval_powerlaw(a_res, b_res, x_line)

    plt.close("all")

    plt.rcParams.update({

        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.labelcolor": "#CFCFCF",
        "xtick.color": "#CFCFCF",
        "ytick.color": "#CFCFCF",
        "text.color": "#EAEAEA",
        "font.size": 11,
    })

    fig = plt.figure(figsize=(16, 9), dpi=180)

    ax = fig.add_subplot(111)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(which="major", color="white", alpha=GRID_MAJOR_A, linewidth=0.95)
    ax.grid(which="minor", color="white", alpha=GRID_MINOR_A, linewidth=0.55)

    ax.plot(x_hist, y_hist, color=COL_PRICE, linewidth=LW_PRICE, alpha=0.95, label="Preis (Tagesschluss)")
    ax.plot(x_line, y_reg, color=COL_REG, linewidth=LW_LINE, label="Lineare Regression")
    ax.plot(x_line, y_sup, color=COL_SUP, linewidth=LW_LINE, label="Unterstützung")
    ax.plot(x_line, y_res, color=COL_RES, linewidth=LW_LINE, label="Widerstand")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, Y_MAX)

    # ⭐ Änderung: komplette Log-Skala mit 1-9 Beschriftung

    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1,2,3,4,5,6,7,8,9)))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_y))

    ax.set_ylabel("USD")

    years = list(range(2011, 2041))

    xticks = [days_since_genesis(dt.date(y,1,1)) for y in years]

    ax.set_xticks(xticks)

    ax.set_xticklabels([str(y) for y in years], rotation=63, ha="right")

    for t in ax.get_xticklabels():
        t.set_fontsize(8)

    plt.subplots_adjust(bottom=0.23)

    leg = ax.legend(loc="lower right", frameon=True, fontsize=10, borderpad=1)

    leg.get_frame().set_facecolor(BG)
    leg.get_frame().set_edgecolor("white")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.35)

    ax.text(
        0.10,
        0.93,
        SOURCE_TEXT,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="white",
        bbox=dict(
            boxstyle="square,pad=0.35",
            facecolor=(0,0,0,0),
            edgecolor="white",
            linewidth=0.9,
        ),
    )

    for spine in ax.spines.values():
        spine.set_color("#22252A")

    fig.savefig(
        OUTFILE,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        pad_inches=0.15,
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
