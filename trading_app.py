"""
Personal Trading Signal App + Prediction Journal (Streamlit)
Run:  pip install streamlit yfinance pandas numpy
      streamlit run trading_app.py

Tab 1: Signal + expected High/Low  -> "Save prediction" button
Tab 2: Journal -> saved predictions ko agle din ke asli data se compare karke report banata hai
Tab 3: Backtest

Predictions 'predictions_log.csv' mein is file ke saath hi save hoti hain (Excel mein bhi khul jati hai).
Educational use only - not financial advice.
"""
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="My Trading Signals", layout="wide")
st.title("📈 My Trading Signal App")

IST = timezone(timedelta(hours=5, minutes=30))
LOG = Path(__file__).with_name("predictions_log.csv")
WATCH = Path(__file__).with_name("watchlist.txt")
# Streamlit Community Cloud par app /mount/src/... se chalta hai; wahan disk restart par reset hoti hai,
# isliye wahan save-buttons band rakhte hain (save GitHub Actions karta hai).
CLOUD = Path(__file__).resolve().as_posix().startswith("/mount/src")
COLS = ["saved_at", "symbol", "base_date", "base_close", "signal", "rsi", "prob_high", "prob_low",
        "avg_high", "avg_low", "pivot", "r1", "s1", "r2", "s2", "settings"]

# ---------- Sidebar settings ----------
FALLBACK_STOCKS = [  # chhoti backup list; poori list ke liye update_stock_list.py chalao
    ("Reliance Industries Limited", "RELIANCE"), ("Tata Consultancy Services Limited", "TCS"),
    ("HDFC Bank Limited", "HDFCBANK"), ("Infosys Limited", "INFY"), ("ICICI Bank Limited", "ICICIBANK"),
    ("ITC Limited", "ITC"), ("State Bank of India", "SBIN"), ("Bharti Airtel Limited", "BHARTIARTL"),
    ("Larsen & Toubro Limited", "LT"), ("Tata Steel Limited", "TATASTEEL"),
    ("Jio Financial Services Limited", "JIOFIN"), ("NBCC (India) Limited", "NBCC"),
    ("HFCL Limited", "HFCL"), ("Jamna Auto Industries Limited", "JAMNAAUTO"),
    ("Samvardhana Motherson International Limited", "MOTHERSON"),
]


@st.cache_data
def load_stocks():
    f = Path(__file__).with_name("stocks.csv")
    if f.exists():
        d = pd.read_csv(f)
        d.columns = [c.strip().lower() for c in d.columns]
        if {"name", "symbol"} <= set(d.columns):
            return d[["name", "symbol"]].dropna().astype(str).sort_values("name").reset_index(drop=True)
    return pd.DataFrame(FALLBACK_STOCKS, columns=["name", "symbol"])


STOCKS = load_stocks()
LABEL2SYM = {f"{n} ({s})": s for n, s in zip(STOCKS.name, STOCKS.symbol)}
SYM2LABEL = {v: k for k, v in LABEL2SYM.items()}
LABELS = list(LABEL2SYM)
picked = st.sidebar.selectbox("Stock chuno (naam type karke search karo)", LABELS,
                              index=LABELS.index(SYM2LABEL["JAMNAAUTO"]) if "JAMNAAUTO" in SYM2LABEL else 0)
manual = st.sidebar.text_input("Ya symbol khud likho (optional)", "").strip().upper()
symbol = (manual if "." in manual else manual + ".NS") if manual else LABEL2SYM[picked] + ".NS"
if len(STOCKS) < 100:
    st.sidebar.caption("Chhoti list chal rahi hai. Poori NSE list ke liye update_stock_list.py chalao.")
period = st.sidebar.selectbox("History", ["2y", "5y", "10y"], index=1)
fast = st.sidebar.number_input("EMA fast", 5, 50, 20)
slow = st.sidebar.number_input("EMA slow", 20, 200, 50)
rsi_n = st.sidebar.number_input("RSI period", 5, 30, 14)
sl_pct = st.sidebar.slider("Stop-loss %", 1.0, 15.0, 5.0, 0.5)
capital = st.sidebar.number_input("Capital (₹)", 10000, 10000000, 100000, 10000)


# ---------- Data + indicators ----------
@st.cache_data(ttl=900)
def load(sym, per):
    try:
        df = yf.download(sym, period=per, interval="1d", auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def now_ist():
    return datetime.now(IST)


def add_indicators(df):
    c = df["Close"]
    df["EMA_F"] = c.ewm(span=fast, adjust=False).mean()
    df["EMA_S"] = c.ewm(span=slow, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_n, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD_H"] = macd - macd.ewm(span=9, adjust=False).mean()
    df["BUY"] = (df.EMA_F > df.EMA_S) & (df.MACD_H > 0) & df.RSI.between(50, 70)
    df["SELL"] = (df.EMA_F < df.EMA_S) | (df.MACD_H < 0) | (df.RSI > 75)
    df["UP"] = df.High / df.Close.shift(1) - 1   # prev close se kitna upar gaya
    df["DN"] = 1 - df.Low / df.Close.shift(1)    # prev close se kitna neeche gaya
    return df


def prepare(sym, per):
    """df = poora data, done = sirf poori hui candles, live = aaj ki candle chal rahi hai ya nahi"""
    df = load(sym, per)
    if df.empty or len(df) < 3:   # naye IPO ke liye bhi kam se kam 3 candles chahiye
        return None, None, False
    df = add_indicators(df.copy())
    n = now_ist()
    live = df.index[-1].date() == n.date() and n.time() < time(15, 45)
    done = df.iloc[:-1] if live else df
    return df, done, live


def is_new(done):
    """Naya IPO / kam data: EMA-RSI-MACD signal ke liye kam se kam slow+30 candles chahiye."""
    return len(done) < int(slow) + 30


def make_prediction(sym, done):
    """Last poori hui candle ke basis par agle trading din ki prediction."""
    last = done.iloc[-1]
    base = float(last.Close)
    ups, dns = done.UP.tail(20), done.DN.tail(20)
    piv = (last.High + last.Low + last.Close) / 3
    sig = "NEW" if is_new(done) else ("BUY" if last.BUY else ("SELL" if last.SELL else "HOLD"))
    return {
        "saved_at": now_ist().strftime("%Y-%m-%d %H:%M"), "symbol": sym,
        "base_date": str(done.index[-1].date()), "base_close": round(base, 2), "signal": sig,
        "rsi": None if is_new(done) else round(float(last.RSI), 1),
        "prob_high": round(base * (1 + ups.quantile(0.8)), 2),
        "prob_low": round(base * (1 - dns.quantile(0.8)), 2),
        "avg_high": round(base * (1 + ups.mean()), 2), "avg_low": round(base * (1 - dns.mean()), 2),
        "pivot": round(float(piv), 2), "r1": round(float(2 * piv - last.Low), 2),
        "s1": round(float(2 * piv - last.High), 2),
        "r2": round(float(piv + (last.High - last.Low)), 2),
        "s2": round(float(piv - (last.High - last.Low)), 2),
        "settings": f"NEW-STOCK {len(done)} candles" if is_new(done) else f"EMA{fast}/{slow} RSI{rsi_n}",
    }


# ---------- Journal storage ----------
def read_watch():
    if not WATCH.exists():
        return []
    out = []
    for line in WATCH.read_text(encoding="utf-8").splitlines():
        x = line.split("#")[0].strip().upper()
        if x and x not in out:
            out.append(x)
    return out


def read_log():
    if LOG.exists():
        return pd.read_csv(LOG)
    return pd.DataFrame(columns=COLS)


def save_prediction(p):
    log = read_log()
    same = (log.symbol == p["symbol"]) & (log.base_date.astype(str) == p["base_date"])
    updated = bool(same.any())
    log = log[~same]
    log = pd.concat([log, pd.DataFrame([p])], ignore_index=True) if len(log) else pd.DataFrame([p])
    log[COLS].to_csv(LOG, index=False)
    return updated


def evaluate(log):
    """Har saved prediction ko agle trading din ki asli candle se compare karo."""
    rows, cache, n = [], {}, now_ist()
    for _, r in log.iterrows():
        if r.symbol not in cache:
            cache[r.symbol] = load(r.symbol, "2y")
        d = cache[r.symbol]
        base_date = pd.to_datetime(r.base_date).date()
        out = {"Stock": r.symbol, "Base date": r.base_date, "Signal": r.signal,
               "Base close": r.base_close, "Pred High": r.prob_high, "Pred Low": r.prob_low}
        nxt = d[d.index.date > base_date] if not d.empty else d
        if nxt.empty:
            out.update({"Actual date": "-", "Status": "⏳ Pending"})
            rows.append(out)
            continue
        a, adate = nxt.iloc[0], nxt.index[0].date()
        live = adate == n.date() and n.time() < time(15, 45)
        base = float(r.base_close)
        high_ok, low_ok = a.High <= r.prob_high, a.Low >= r.prob_low
        if r.signal == "BUY":
            direction = "✅ Sahi" if a.Close > base else "❌ Galat"
        elif r.signal == "SELL":
            direction = "✅ Sahi" if a.Close < base else "❌ Galat"
        else:
            direction = "— (HOLD)" if r.signal == "HOLD" else "— (no signal)"
        notes = []
        if high_ok and low_ok:
            notes.append("Range ke andar raha")
        if not high_ok:
            notes.append(f"High range se {(a.High / r.prob_high - 1) * 100:.1f}% upar gaya")
        if not low_ok:
            notes.append(f"Low range se {(1 - a.Low / r.prob_low) * 100:.1f}% neeche gaya")
        out.update({
            "Actual date": str(adate), "Status": "🔴 Live (din chal raha)" if live else "✔ Final",
            "Open": round(float(a.Open), 2), "High": round(float(a.High), 2),
            "Low": round(float(a.Low), 2), "Close": round(float(a.Close), 2),
            "High vs base %": round((a.High / base - 1) * 100, 2),
            "Low vs base %": round((a.Low / base - 1) * 100, 2),
            "Close vs base %": round((a.Close / base - 1) * 100, 2),
            "High range": "✅" if high_ok else "❌", "Low range": "✅" if low_ok else "❌",
            "R1 touch": "✅" if a.High >= r.r1 else "—", "S1 touch": "✅" if a.Low <= r.s1 else "—",
            "Direction": direction, "Note": "; ".join(notes),
        })
        rows.append(out)
    return pd.DataFrame(rows)


def backtest(df, sl):
    pos, entry, entry_date = 0, 0.0, None
    trades, equity = [], [1.0]
    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        eq = equity[-1]
        if pos == 0 and prev.BUY:
            pos, entry, entry_date = 1, row.Open, df.index[i]
            eq *= row.Close / entry
        elif pos == 1:
            stop = entry * (1 - sl / 100)
            exit_price = None
            if row.Low <= stop:
                exit_price, why = min(stop, row.Open), "Stop-loss"
            elif prev.SELL:
                exit_price, why = row.Open, "Sell signal"
            if exit_price is not None:
                eq *= exit_price / prev.Close
                trades.append({"Entry": entry_date.date(), "Exit": df.index[i].date(),
                               "Buy": round(entry, 2), "Sell": round(exit_price, 2),
                               "Return %": round((exit_price / entry - 1) * 100, 2), "Why": why})
                pos = 0
            else:
                eq *= row.Close / prev.Close
        equity.append(eq)
    return pd.DataFrame(trades), pd.Series(equity, index=df.index)


df, done, live = prepare(symbol, period)
if df is None:
    st.error("Data nahi mila (ya 3 candles se kam hai). Symbol check karo: company ka naam nahi, ticker likho, "
             "jaise TCS.NS. Bahut naya listing ho to Yahoo par abhi na ho, tab NSE se symbol dekho ya .BO try karo.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Aaj ka Signal", "📝 Prediction Journal", "🧪 Backtest"])

# =====================  TAB 1: SIGNAL  =====================
with tab1:
    pred = make_prediction(symbol, done)
    st.subheader(f"{symbol}  |  base candle: {pred['base_date']}")
    if live:
        st.info("Market abhi chal raha hai. Signal aur range pichhli poori hui candle (base) se bane hain. "
                f"Live price: ₹{df.Close.iloc[-1]:,.2f}")
    new_stock = is_new(done)
    if new_stock:
        st.warning(f"Naya stock: sirf {len(done)} candles ka data hai. EMA/RSI/MACD signal nahi bante, isliye "
                   "sirf probable High/Low range aur pivot levels dikha raha hoon. Confidence kam hai, "
                   "ise sirf seekhne ke liye use karo.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Base close", f"₹{pred['base_close']:,.2f}")
    c2.metric("RSI", "—" if new_stock else f"{pred['rsi']:.1f}")
    c3.metric("MACD hist", "—" if new_stock else f"{done.MACD_H.iloc[-1]:.2f}")
    st.markdown({"BUY": "## 🟢 BUY", "SELL": "## 🔴 SELL / AVOID", "HOLD": "## 🟡 HOLD / WAIT",
                 "NEW": "## ⚪ Signal nahi (data kam)"}[pred["signal"]])
    if pred["signal"] == "BUY":
        st.caption(f"Stop-loss level: ₹{pred['base_close'] * (1 - sl_pct / 100):,.2f}")

    st.line_chart(done[["Close"] if new_stock else ["Close", "EMA_F", "EMA_S"]].tail(250))

    st.subheader("Agle trading din ka expected High / Low (probable range)")
    h1, h2, h3 = st.columns(3)
    h1.metric("Probable High (80%)", f"₹{pred['prob_high']:,.2f}")
    h2.metric("Probable Low (80%)", f"₹{pred['prob_low']:,.2f}")
    h3.metric("Base (last close)", f"₹{pred['base_close']:,.2f}")
    st.dataframe(pd.DataFrame({
        "Level": ["Resistance 2", "Resistance 1 (target)", "Pivot", "Support 1", "Support 2",
                  "Avg High (20d)", "Avg Low (20d)"],
        "Price ₹": [pred["r2"], pred["r1"], pred["pivot"], pred["s1"], pred["s2"],
                    pred["avg_high"], pred["avg_low"]]}), use_container_width=True)

    qu = done.UP.rolling(20).quantile(0.8).shift(1)
    qd = done.DN.rolling(20).quantile(0.8).shift(1)
    chk = pd.DataFrame({"hi": done.UP <= qu, "lo": done.DN <= qd}).where(qu.notna()).dropna()
    if len(chk) >= 10:
        k1, k2, k3 = st.columns(3)
        k1.metric("High range sahi (history)", f"{chk.hi.mean() * 100:.0f}%")
        k2.metric("Low range sahi (history)", f"{chk.lo.mean() * 100:.0f}%")
        k3.metric("Dono sahi", f"{(chk.hi & chk.lo).mean() * 100:.0f}%")
    else:
        st.info("Range ki history accuracy ke liye kam se kam ~30 candles chahiye. Abhi data kam hai.")

    st.divider()
    if CLOUD:
        st.info("Cloud par predictions roz subah automatic save hoti hain. Report ke liye Journal tab dekho.")
    elif st.button("💾 Is prediction ko Journal mein save karo", type="primary"):
        was_update = save_prediction(pred)
        st.success("Prediction update ho gayi." if was_update else
                   "Prediction save ho gayi. Agle trading din ke baad 'Prediction Journal' tab mein report dekho.")
    st.caption("Ye exact prediction nahi, ek probable range hai. Bade news ya gap-up/gap-down par range tut sakti hai.")

# =====================  TAB 2: JOURNAL  =====================
with tab2:
    st.subheader("Prediction vs Actual report")
    st.caption(f"Saved file: {LOG}")

    with st.expander("⭐ Watchlist (auto_save.py bhi isi list ko use karta hai)"):
        cur = read_watch()

        if CLOUD:
            st.write(", ".join(cur) if cur else "(watchlist khali hai)")
            st.info("Watchlist badalne ke liye GitHub par watchlist.txt edit karo (ek line mein ek stock, NSE ke liye .NS). "
                    "Agli subah se auto-save naye list ko use karega.")
        else:
            def lab(x):
                return SYM2LABEL.get(x.replace(".NS", ""), x)

            options = LABELS + [lab(x) for x in cur if lab(x) not in LABEL2SYM]
            chosen = st.multiselect("Stocks chuno (naam se search karo)", options, default=[lab(x) for x in cur])
            extra = st.text_input("Ya extra symbols (comma se alag, optional)", "")
            wl_syms = [LABEL2SYM[c] + ".NS" if c in LABEL2SYM else c for c in chosen]
            for x in extra.split(","):
                x = x.strip().upper()
                if x:
                    wl_syms.append(x if "." in x else x + ".NS")
            wl_syms = list(dict.fromkeys(wl_syms))
            b1, b2 = st.columns(2)
            if b1.button("💾 Watchlist file update karo"):
                WATCH.write_text("# Ek line mein ek stock (NSE ke liye .NS)\n" + "\n".join(wl_syms) + "\n", encoding="utf-8")
                st.success(f"Watchlist mein {len(wl_syms)} stocks save huye. Auto-save kal subah se inhe use karega.")
            if b2.button("▶ Abhi sab ki prediction save karo"):
                saved, failed = 0, []
                for x in wl_syms:
                    _, d2, _ = prepare(x, "2y")
                    if d2 is None:
                        failed.append(x)
                        continue
                    save_prediction(make_prediction(x, d2))
                    saved += 1
                st.success(f"{saved} predictions save huin." + (f" Data nahi mila: {', '.join(failed)}" if failed else ""))

    log = read_log()
    if log.empty:
        st.info("Abhi koi prediction save nahi hai. Pehle tab mein 'Save' dabao.")
    else:
        rep = evaluate(log.sort_values("saved_at", ascending=False))
        fin = rep[rep.Status == "✔ Final"]
        pend = (rep.Status == "⏳ Pending").sum()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total saved", len(rep))
        m2.metric("Check ho chuki", len(fin))
        m3.metric("Pending", int(pend))
        if len(fin):
            m4.metric("High range sahi", f"{(fin['High range'] == '✅').mean() * 100:.0f}%")
            m5.metric("Low range sahi", f"{(fin['Low range'] == '✅').mean() * 100:.0f}%")
            dirs = fin[fin.Direction.isin(["✅ Sahi", "❌ Galat"])]
            if len(dirs):
                st.metric("BUY/SELL direction sahi", f"{(dirs.Direction == '✅ Sahi').mean() * 100:.0f}%  ({len(dirs)} calls)")
        st.dataframe(rep, use_container_width=True)
        st.download_button("⬇ Report CSV download", rep.to_csv(index=False).encode("utf-8"),
                           file_name="prediction_report.csv", mime="text/csv")
        st.caption("Direction: BUY tab sahi jab agle din ka close base close se upar ho; SELL tab sahi jab neeche ho. "
                   "10-20 se kam samples par koi nateeja mat nikalo.")
        if not CLOUD and st.button("🗑 Poora journal clear karo"):
            LOG.unlink(missing_ok=True)
            st.rerun()

# =====================  TAB 3: BACKTEST  =====================
with tab3:
    st.subheader("Backtest (purane data par strategy ka result)")
    trades, eq = backtest(done, sl_pct) if not is_new(done) else (pd.DataFrame(), None)
    bh = done.Close / done.Close.iloc[0]
    if trades.empty:
        st.info("Naya stock: backtest ke liye data kam hai." if is_new(done) else "Is period mein koi trade nahi bana.")
    else:
        ret, bh_ret = (eq.iloc[-1] - 1) * 100, (bh.iloc[-1] - 1) * 100
        dd = ((eq / eq.cummax()) - 1).min() * 100
        win = (trades["Return %"] > 0).mean() * 100
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Strategy return", f"{ret:.1f}%")
        b2.metric("Buy & hold", f"{bh_ret:.1f}%")
        b3.metric("Win rate", f"{win:.0f}%")
        b4.metric("Max drawdown", f"{dd:.1f}%")
        b5.metric("Trades", len(trades))
        st.line_chart(pd.DataFrame({"Strategy": eq * capital, "Buy & hold": bh * capital}))
        st.dataframe(trades.iloc[::-1], use_container_width=True)
        st.caption("Brokerage, tax aur slippage shamil nahi hain. Past result future ki guarantee nahi hai.")

st.warning("Educational tool hai, financial advice nahi. Pehle paper trading karo, phir chhote amount se shuru karo.")
