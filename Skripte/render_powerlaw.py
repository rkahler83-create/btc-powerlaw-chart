import datetime as dt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt

OUTFILE = "docs/powerlaw.png"

START_DATE = dt.date(2011, 1, 1)
END_DATE = dt.date(2040, 12, 31)
GENESIS = dt.date(2009, 1, 3)

Y_MIN = 0.1
Y_MAX = 10_000_000

COL_PRICE = "#F7930A"
COL_REG   = "#7FC64B"
COL_SUP   = "#D04B3F"
COL_RES   = "#7E3FBF"

BG = "#0B0C10"

SOURCE_TEXT = "Quelldaten: Yahoo Finance (Ticker BTC-USD)"


def days_since_genesis(d):
    return max(1, (d - GENESIS).days)


def fetch_data():
    df = yf.download(
        "BTC-USD",
        start=START_DATE.isoformat(),
        interval="1d",
        progress=False
    )

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:,0]

    close = close.dropna()
    close = close[close > 0]

    return close


def fit_powerlaw(x,y):

    lx = np.log10(x)
    ly = np.log10(y)

    b,a = np.polyfit(lx,ly,1)

    return a,b


def eval_powerlaw(a,b,x):

    return 10**(a + b*np.log10(x))


def main():

    close = fetch_data()

    dates = close.index.date

    x_hist = np.array([days_since_genesis(d) for d in dates])
    y_hist = close.values

    a_mid,b_mid = fit_powerlaw(x_hist,y_hist)

    y_mid_hist = eval_powerlaw(a_mid,b_mid,x_hist)

    resid = np.log10(y_hist) - np.log10(y_mid_hist)

    q_sup = np.quantile(resid,0.2)
    q_res = np.quantile(resid,0.8)

    sup_mask = resid <= q_sup
    res_mask = resid >= q_res

    a_sup,b_sup = fit_powerlaw(x_hist[sup_mask],y_hist[sup_mask])
    a_res,b_res = fit_powerlaw(x_hist[res_mask],y_hist[res_mask])

    x_line = np.geomspace(
        days_since_genesis(START_DATE),
        days_since_genesis(END_DATE),
        1000
    )

    y_reg = eval_powerlaw(a_mid,b_mid,x_line)
    y_sup = eval_powerlaw(a_sup,b_sup,x_line)
    y_res = eval_powerlaw(a_res,b_res,x_line)

    plt.close("all")

    plt.rcParams.update({
        "figure.facecolor":BG,
        "axes.facecolor":BG,
        "text.color":"white",
        "axes.labelcolor":"white",
        "xtick.color":"white",
        "ytick.color":"white"
    })

    fig = plt.figure(figsize=(16,9),dpi=180)

    ax = fig.add_subplot(111)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(True,which="major",color="white",alpha=0.2)
    ax.grid(True,which="minor",color="white",alpha=0.06)

    ax.plot(x_hist,y_hist,color=COL_PRICE,linewidth=2.2,label="Preis (Tagesschluss)")
    ax.plot(x_line,y_reg,color=COL_REG,linewidth=2.2,label="Lineare Regression")
    ax.plot(x_line,y_sup,color=COL_SUP,linewidth=2.2,label="Unterstützung")
    ax.plot(x_line,y_res,color=COL_RES,linewidth=2.2,label="Widerstand")

    ax.set_xlim(days_since_genesis(START_DATE),days_since_genesis(END_DATE))
    ax.set_ylim(Y_MIN,Y_MAX)

    # Y Achse exakt wie gewünscht
    y_ticks = [
        0.1,
        1,
        5,
        10,
        50,
        100,
        500,
        1000,
        5000,
        10000,
        50000,
        100000,
        500000,
        1000000,
        5000000,
        10000000
    ]

    ax.set_yticks(y_ticks)

    ax.set_yticklabels([
        "0",
        "1",
        "5",
        "10",
        "50",
        "100",
        "500",
        "1000",
        "5000",
        "10000",
        "50000",
        "100000",
        "500000",
        "1000000",
        "5000000",
        "10000000"
    ])

    ax.set_ylabel("USD")

    years = list(range(2011,2041))

    xticks = [days_since_genesis(dt.date(y,1,1)) for y in years]

    ax.set_xticks(xticks)
    ax.set_xticklabels([str(y) for y in years],rotation=60,ha="right")

    for t in ax.get_xticklabels():
        t.set_fontsize(8)

    plt.subplots_adjust(bottom=0.23)

    leg = ax.legend(loc="lower right")

    leg.get_frame().set_facecolor(BG)
    leg.get_frame().set_edgecolor("white")

    ax.text(
        0.10,
        0.93,
        SOURCE_TEXT,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(
            boxstyle="square,pad=0.35",
            facecolor=(0,0,0,0),
            edgecolor="white",
            linewidth=0.9
        )
    )

    fig.savefig(
        OUTFILE,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        pad_inches=0.15
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
