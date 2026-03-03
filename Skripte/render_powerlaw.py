import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yfinance as yf

# ==============================
# OUTPUT DATEI
# ==============================
OUTFILE = "docs/powerlaw.png"

# ==============================
# DESIGN / FARBEN
# ==============================
BG = "#0b0b0b"
GRID_MAJOR = "#3a3a3a"
GRID_MINOR = "#2a2a2a"
TEXT = "#eaeaea"

BTC_ORANGE = "#F7931A"      # Preislinie
REG_COLOR = "#6fdc3f"       # Regression (grün)
SUPPORT_COLOR = "#ff6a00"   # Support (orange)
RES_COLOR = "#a64dff"       # Resistance (lila)

# ==============================
# MODELL EINSTELLUNGEN
# ==============================
GENESIS = dt.date(2009, 1, 3)
CHANNEL_SIGMA = 2.0
EXTEND_TO_YEAR = 2040
Y_MAX = 10000000


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


def fit_powerlaw_channel(dates, prices):
    x = np.array([days_since_genesis(d) for d in dates], dtype=float)
    y = np.array(prices, dtype=float)

    lx = np.log10(x)
    ly = np.log10(y)

    b, a = np.polyfit(lx, ly, 1)
    residuals = ly - (a + b * lx)
    sigma = float(np.std(residuals))

    return a, b, sigma


def generate_dates(start_date, end_year):
    end_date = dt.date(end_year, 12, 31)
    n = (end_date - start_date).days
    return [start_date + dt.timedelta(days=i) for i in range(n + 1)]


def main():

    # ==============================
    # HISTORISCHE DATEN
    # ==============================
    hist_dates, hist_prices = fetch_btc_daily_close()

    # ==============================
    # POWER LAW FIT
    # ==============================
    a, b, sigma = fit_powerlaw_channel(hist_dates, hist_prices)

    model_dates = generate_dates(hist_dates[0], EXTEND_TO_YEAR)
    x_model = np.array([days_since_genesis(d) for d in model_dates], dtype=float)
    lx_model = np.log10(x_model)

    reg_log = a + b * lx_model
    reg = 10 ** reg_log

    support = 10 ** (reg_log - CHANNEL_SIGMA * sigma)
    resistance = 10 ** (reg_log + CHANNEL_SIGMA * sigma)

    # ==============================
    # PLOT
    # ==============================
    plt.figure(figsize=(15.5, 7.8), dpi=220)
    ax = plt.gca()

    ax.set_facecolor(BG)
    plt.gcf().patch.set_facecolor(BG)

    ax.set_yscale("log")

    # Dollar Labels statt 10^x
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x:,.0f}")
    )

    ymin_auto = max(0.1, float(np.nanmin(hist_prices)) * 0.6)

    ax.set_ylim(ymin_auto, Y_MAX)
    ax.set_xlim(hist_dates[0], dt.date(EXTEND_TO_YEAR, 12, 31))

    ax.grid(True, which="major", color=GRID_MAJOR, linewidth=0.9, alpha=0.85)
    ax.grid(True, which="minor", color=GRID_MINOR, linewidth=0.4, alpha=0.35)

    years = list(range(hist_dates[0].year, EXTEND_TO_YEAR + 1))
    year_ticks = [dt.date(y, 1, 1) for y in years]

    ax.set_xticks(year_ticks)
    ax.set_xticklabels([str(y) for y in years], rotation=35, ha="right", color=TEXT)

    # Linien
    ax.plot(model_dates, resistance, color=RES_COLOR, linewidth=2.2, label="Resistance")
    ax.plot(model_dates, reg, color=REG_COLOR, linewidth=2.2, label="Linear regression fit")
    ax.plot(model_dates, support, color=SUPPORT_COLOR, linewidth=2.2, label="Support")

    ax.plot(hist_dates, hist_prices, color=BTC_ORANGE, linewidth=2.0, label="Price end of day")

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

    updated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    plt.figtext(
        0.99,
        0.015,
        f"Updated daily • {updated}",
        ha="right",
        fontsize=9,
        color="#bdbdbd"
    )

    plt.tight_layout()
    plt.savefig(OUTFILE)
    plt.close()


if __name__ == "__main__":
    main()
