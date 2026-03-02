import datetime as dt
import requests
import matplotlib.pyplot as plt

# ===== Einstellungen =====
DAYS = 365 * 6                 # 6 Jahre Historie
OUTFILE = "docs/powerlaw.png"  # Output im Repo

# Santostasi-style Power Law: price ≈ A * (days_since_genesis)^N
A = 10 ** (-16.493)
N = 5.68
GENESIS = dt.date(2009, 1, 3)

def days_since_genesis(d: dt.date) -> int:
    return max(1, (d - GENESIS).days)

def fetch_btc_usd_history(days: int):
    url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        f"?vs_currency=usd&days={days}&interval=daily"
    )
    r = requests.get(url, timeout=45, headers={"User-Agent": "powerlaw-chart/1.0"})
    r.raise_for_status()

    data = r.json()["prices"]  # [[ms, price], ...]
    xs, ys = [], []
    for ms, price in data:
        d = dt.datetime.utcfromtimestamp(ms / 1000).date()
        xs.append(d)
        ys.append(float(price))
    return xs, ys

def powerlaw_price(d: dt.date) -> float:
    x = days_since_genesis(d)
    return A * (x ** N)

def main():
    xs, price = fetch_btc_usd_history(DAYS)

    # Rohmodell + Skalierung auf letzten echten Preis (damit die Kurve sichtbar passt)
    model_raw = [powerlaw_price(d) for d in xs]
    scale = price[-1] / model_raw[-1] if model_raw[-1] > 0 else 1.0
    model = [m * scale for m in model_raw]

    # Korridor (einfacher Faktorbereich)
    band_low = [m * 0.5 for m in model]
    band_high = [m * 2.0 for m in model]

    # ===== Dark Plot =====
    plt.figure(figsize=(14, 7), dpi=220)
    ax = plt.gca()

    bg = "#0b0b0b"
    ax.set_facecolor(bg)
    plt.gcf().patch.set_facecolor(bg)

    ax.set_yscale("log")

    # Corridor
    ax.fill_between(xs, band_low, band_high, alpha=0.18)

    # Lines
    ax.plot(xs, price, linewidth=2.2, label="BTC Price (USD)")
    ax.plot(xs, model, linewidth=2.8, label="Power Law")

    ax.set_title("Bitcoin Power Law (Daily • Log Scale)", pad=12, color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price (USD, log)", color="white")

    ax.grid(True, which="both", linewidth=0.6, alpha=0.18)
    ax.tick_params(colors="#eaeaea")

    leg = ax.legend(loc="upper left")
    for t in leg.get_texts():
        t.set_color("white")

    updated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    plt.figtext(0.99, 0.015, f"Updated daily • {updated}", ha="right", fontsize=9, color="#cfcfcf")

    plt.tight_layout()
    plt.savefig(OUTFILE)
    plt.close()

if __name__ == "__main__":
    main()
