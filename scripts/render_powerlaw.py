4import math
import datetime as dt
import requests
import matplotlib.pyplot as plt

DAYS = 365 * 6
OUTFILE = "docs/powerlaw.png"

A = 10 ** (-16.493)
N = 5.68
GENESIS = dt.date(2009, 1, 3)

def days_since_genesis(d):
    return max(1, (d - GENESIS).days)

def fetch_btc_usd_history(days):
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}&interval=daily"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()["prices"]
    xs, ys = [], []
    for ms, price in data:
        d = dt.datetime.utcfromtimestamp(ms / 1000).date()
        xs.append(d)
        ys.append(float(price))
    return xs, ys

def powerlaw_price(d):
    x = days_since_genesis(d)
    return A * (x ** N)

def main():
    xs, price = fetch_btc_usd_history(DAYS)

    model_raw = [powerlaw_price(d) for d in xs]
    scale = price[-1] / model_raw[-1]
    model = [m * scale for m in model_raw]
    band_low = [m * 0.5 for m in model]
    band_high = [m * 2.0 for m in model]

    plt.figure(figsize=(14, 7), dpi=200)
    ax = plt.gca()
    ax.set_facecolor("#0b0b0b")
    plt.gcf().patch.set_facecolor("#0b0b0b")

    ax.set_yscale("log")
    ax.fill_between(xs, band_low, band_high, alpha=0.18)

    ax.plot(xs, price, linewidth=2.2, label="BTC Price (USD)")
    ax.plot(xs, model, linewidth=2.8, label="Power Law")

    ax.set_title("Bitcoin Power Law (Daily • Log Scale)", color="white")
    ax.grid(True, which="both", alpha=0.18)

    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    leg = ax.legend()
    for text in leg.get_texts():
        text.set_color("white")

    updated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    plt.figtext(0.99, 0.015, f"Updated daily • {updated}", ha="right", fontsize=9, color="white")

    plt.tight_layout()
    plt.savefig(OUTFILE)
    plt.close()

if __name__ == "__main__":
    main()
