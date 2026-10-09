"""
Auto Save: watchlist.txt ke sab stocks ki prediction predictions_log.csv mein save karta hai.
Streamlit app kholne ki zarurat nahi. Windows Task Scheduler se roz subah chalao (steps README mein).

Rakho ek hi folder mein:  trading_app.py, auto_save.py, watchlist.txt, auto_save.bat
Python packages: pip install yfinance pandas numpy

Settings neeche trading_app.py ki default settings jaisi hain (EMA 20/50, RSI 14).
Agar app mein settings badalte ho to yahan bhi badlo, warna dono ki predictions alag logic se banengi.
"""
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FAST, SLOW, RSI_N = 20, 50, 14
PERIOD = "2y"

IST = timezone(timedelta(hours=5, minutes=30))
HERE = Path(__file__).resolve().parent
LOG = HERE / "predictions_log.csv"
WATCH = HERE / "watchlist.txt"
RUNLOG = HERE / "auto_save_log.txt"
COLS = ["saved_at", "symbol", "base_date", "base_close", "signal", "rsi", "prob_high", "prob_low",
        "avg_high", "avg_low", "pivot", "r1", "s1", "r2", "s2", "settings"]


def now_ist():
    return datetime.now(IST)


def add_indicators(df):
    c = df["Close"]
    df["EMA_F"] = c.ewm(span=FAST, adjust=False).mean()
    df["EMA_S"] = c.ewm(span=SLOW, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / RSI_N, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_N, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD_H"] = macd - macd.ewm(span=9, adjust=False).mean()
    df["BUY"] = (df.EMA_F > df.EMA_S) & (df.MACD_H > 0) & df.RSI.between(50, 70)
    df["SELL"] = (df.EMA_F < df.EMA_S) | (df.MACD_H < 0) | (df.RSI > 75)
    df["UP"] = df.High / df.Close.shift(1) - 1
    df["DN"] = 1 - df.Low / df.Close.shift(1)
    return df


def predict(sym):
    df = yf.download(sym, period=PERIOD, interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if len(df) < 3:
        raise ValueError("data nahi mila (symbol check karo)")
    df = add_indicators(df.copy())
    n = now_ist()
    live = df.index[-1].date() == n.date() and n.time() < time(15, 45)
    done = df.iloc[:-1] if live else df          # chal rahi candle bahar
    new = len(done) < SLOW + 30
    last = done.iloc[-1]
    base = float(last.Close)
    ups, dns = done.UP.tail(20), done.DN.tail(20)
    piv = (last.High + last.Low + last.Close) / 3
    sig = "NEW" if new else ("BUY" if last.BUY else ("SELL" if last.SELL else "HOLD"))
    return {
        "saved_at": n.strftime("%Y-%m-%d %H:%M"), "symbol": sym,
        "base_date": str(done.index[-1].date()), "base_close": round(base, 2), "signal": sig,
        "rsi": None if new else round(float(last.RSI), 1),
        "prob_high": round(base * (1 + ups.quantile(0.8)), 2),
        "prob_low": round(base * (1 - dns.quantile(0.8)), 2),
        "avg_high": round(base * (1 + ups.mean()), 2), "avg_low": round(base * (1 - dns.mean()), 2),
        "pivot": round(float(piv), 2), "r1": round(float(2 * piv - last.Low), 2),
        "s1": round(float(2 * piv - last.High), 2),
        "r2": round(float(piv + (last.High - last.Low)), 2),
        "s2": round(float(piv - (last.High - last.Low)), 2),
        "settings": f"NEW-STOCK {len(done)} candles" if new else f"EMA{FAST}/{SLOW} RSI{RSI_N}",
    }


def read_watchlist():
    if not WATCH.exists():
        raise SystemExit(f"watchlist.txt nahi mili: {WATCH}")
    out = []
    for line in WATCH.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip().upper()
        if line and line not in out:
            out.append(line)
    return out


def main():
    syms = read_watchlist()
    preds, failed = [], []
    for s in syms:
        try:
            preds.append(predict(s))
        except Exception as e:
            failed.append(f"{s} ({e})")

    if preds:
        new = pd.DataFrame(preds)
        old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=COLS)
        if len(old):   # same stock + same base_date ho to purani hata ke nayi rakho (duplicate nahi)
            key_new = set(zip(new.symbol, new.base_date.astype(str)))
            old = old[[(a, str(b)) not in key_new for a, b in zip(old.symbol, old.base_date)]]
        pd.concat([old, new], ignore_index=True)[COLS].to_csv(LOG, index=False)

    msg = f"{now_ist():%Y-%m-%d %H:%M} | saved {len(preds)}/{len(syms)}"
    if failed:
        msg += " | FAILED: " + ", ".join(failed)
    print(msg)
    with open(RUNLOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


if __name__ == "__main__":
    main()
