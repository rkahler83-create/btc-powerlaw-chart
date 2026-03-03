import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yfinance as yf

# ==============================
# OUTPUT
# ==============================
OUTFILE = "docs/powerlaw.png"

# ==============================
# DESIGN
# ==============================
BG = "#0b0b0b"
GRID_MAJOR = "#3a3a3a"
GRID_MINOR = "#2a2a2a"
TEXT = "#eaeaea"

BTC_ORANGE = "#F7931A"
REG_COLOR = "#6fdc3f"
SUPPORT_COLOR = "#ff2d2d"
RES_COLOR = "#a64dff"

# ==============================
# POWER LAW SETTINGS
# ==============================
GENESIS = dt.date(2009, 1, 3)

SUPPORT_MULT = 0.25
RESIST_MULT = 4.0

END_DATE = dt.date(2040, 12, 31)
Y_MAX = 10_000_000

LEFT_PAD_FACTOR = 0.55


def days_since_genesis(d):
    return max(1, (d - GENESIS).days)


def fetch_btc_daily_close(start="2010-07-17"):
    df = yf.download("BTC-USD", start=start, interval="1d", progress=False)
    df = df.dropna()

    close = df["Close"]

    try:
        prices = close.values.astype(float).flatten()
    except Exception:
        prices = np.array(list(close), dtype=float)

    dates = [d.date() for d in df.index]
    return dates, prices


def fit_powerlaw(dates, prices):
    x = np.array([days_since_genesis(d) for d in dates], dtype=float)
    y = np.array(prices, dtype=float)

    lx = np.log10(x)
    ly = np.log10(y)

    b, a = np.polyfit(lx, ly, 1)
    k = 10 ** a
    return k, b


def generate_daily_dates(start_date, end_date):
    n = (end_date - start_date).days
    return [start_date + dt.timedelta(days=i) for i in range(n + 1)]


def main():

    hist_dates, hist_prices = fetch_btc_daily_close()
    first_date = hist_dates[0]

    k, b = fit_powerlaw(hist_dates, hist_prices)

    model_dates = generate_daily_dates(first_date, END_DATE)

    x_hist = np.array([days_since_genesis(d) for d in hist_dates], dtype=float)
    x_model = np.array([days_since_genesis(d) for d in model_dates], dtype=float)

    reg = k * (x_model ** b)
    support = reg * SUPPORT_MULT
    resistance = reg * RESIST_MULT

    plt.figure(figsize=(15.5, 7.8), dpi=220)
    ax = plt.gca()

    ax.set_facecolor(BG)
    plt.gcf().patch.set_facecolor(BG)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f"{y:,.0f}")
    )

    ymin_auto = max(0.1, float(np.nanmin(hist_prices)) * 0.6)
    ax.set_ylim(ymin_auto, Y_MAX)

    x_min = max(1, int(days_since_genesis(first_date) * LEFT_PAD_FACTOR))
    x_max = days_since_genesis(END_DATE)
    ax.set_xlim(x_min, x_max)

    ax.grid(True, which="major", color=GRID_MAJOR, linewidth=0.9, alpha=0.85)
    ax.grid(True, which="minor", color=GRID_MINOR, linewidth=0.45, alpha=0.35)

    start_year = first_date.year
    years = list(range(start_year, END_DATE.year + 1))
    year_tick_vals = [
        days_since_genesis(dt.date(y, 1, 1))
        for y in years
        if days_since_genesis(dt.date(y, 1, 1)) >= x_min
    ]

    ax.set_xticks(year_tick_vals)
    ax.set_xticklabels(
        [str(y) for y in years if days_since_genesis(dt.date(y, 1, 1)) >= x_min],
        rotation=35,
        ha="right",
        color=TEXT,
        fontsize=9
    )

    ax.plot(x_model, resistance, color=RES_COLOR, linewidth=2.2, label="Widerstand")
    ax.plot(x_model, reg, color=REG_COLOR, linewidth=2.2, label="Regression")
    ax.plot(x_model, support, color=SUPPORT_COLOR, linewidth=2.2, label="Unterstützung")
    ax.plot(x_hist, hist_prices, color=BTC_ORANGE, linewidth=2.0, label="Preis (Tagesschluss)")

    ax.tick_params(colors=TEXT)
    ax.set_ylabel("USD", color=TEXT)

    leg = ax.legend(
        loc="lower right",
        frameon=True,
        facecolor=BG,
        edgecolor=GRID_MAJOR,
        framealpha=1.0,
        fontsize=10,
        borderpad=0.8
    )
    for t in leg.get_texts():
        t.set_color(TEXT)

    # Quelle links unten
    plt.figtext(
        0.015,
        0.015,
        "Quelle Kursdaten: Yahoo Finance (Ticker BTC-USD)",
        ha="left",
        fontsize=8,
        color="#aaaaaa"
    )

    plt.tight_layout()
    plt.savefig(OUTFILE)
    plt.close()


if __name__ == "__main__":
    main()
