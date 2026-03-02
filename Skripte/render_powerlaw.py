import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yfinance as yf

# ===== Einstellungen =====
OUTFILE = "docs/powerlaw.png"
GENESIS = dt.date(2009, 1, 3)

# Santostasi-style Power Law: price ≈ A * (days_since_genesis)^N
A = 10 ** (-16.493)
N = 5.68

def days_since_genesis(d: dt.date) -> int:
    return max(1, (d - GENESIS).days)

def powerlaw_price(d: dt.date) -> float:
    x = days_since_genesis(d)
    return A * (x ** N)

def main():
    # BTC-USD von Yahoo Finance (kein API-Key)
    data = yf.download("BTC-USD", start="2013-01-01", interval="1d", progress=False)
    data = data.dropna()

    # X (Dates)
    xs = [d.date() for d in data.index]

    # Y (Close) - robust, egal ob Series oder DataFrame
    close = data["Close"]
    try:
        price = list(close.values.flatten())
    except Exception:
        price = list(close)

    # Modell + Skalierung auf letzten echten Preis
    model_raw = [powerlaw_price(d) for d in xs]
    scale = price[-1] / model_raw[-1] if model_raw[-1] > 0 else 1.0
    model = [m * scale for m in model_raw]

    # Korridor (einfacher Faktorbereich)
    band_low = [m * 0.5 for m in model]
    band_high = [m * 2.0 for m in model]

    # ===== Dark Chart =====
    plt.figure(figsize=(14, 7), dpi=220)
    ax = plt.gca()

    bg = "#0b0b0b"
    ax.set_facecolor(bg)
    plt.gcf().patch.set_facecolor(bg)

    # Log Scale (aber mit normalen $ Labels)
    ax.set_yscale("log")

    # >>> DAS IST DIE WICHTIGE ÄNDERUNG: $ statt 10^x <<<
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"${x:,.0f}")
    )

    ax.fill_between(xs, band_low, band_high, alpha=0.15)

    ax.plot(xs, price, linewidth=2.2, label="BTC Price (USD)")
    ax.plot(xs, model, linewidth=2.8, label="Power Law")

    ax.set_title("Bitcoin Power Law (Daily • Log Scale)", pad=12, color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price (USD)", color="white")

    ax.grid(True, which="both", linewidth=0.6, alpha=0.18)
    ax.tick_params(colors="#eaeaea")

    leg = ax.legend(loc="upper left")
    for t in leg.get_texts():
        t.set_color("white")

    updated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    plt.figtext(
        0.99, 0.015,
        f"Updated daily • {updated}",
        ha="right",
        fontsize=9,
        color="#cfcfcf"
    )

    plt.tight_layout()
    plt.savefig(OUTFILE)
    plt.close()

if __name__ == "__main__":
    main()
