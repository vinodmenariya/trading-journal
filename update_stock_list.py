"""
NSE ki official list se stocks.csv banata hai (company name + symbol), taaki app mein naam se dropdown mil sake.
Chalao:  python update_stock_list.py     (mahine mein ek baar, ya naya IPO list hone par)

Agar NSE se download na ho (kabhi-kabhi block karta hai):
  1. NSE site -> Market Data -> 'Securities available for trading' se EQUITY_L.csv download karo
  2. Use isi folder mein rakho aur script dobara chalao
"""
import csv
import io
import urllib.request
from pathlib import Path

URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
HERE = Path(__file__).resolve().parent
LOCAL = HERE / "EQUITY_L.csv"
OUT = HERE / "stocks.csv"


def fetch():
    try:
        req = urllib.request.Request(URL, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
            "Accept": "text/csv,*/*", "Referer": "https://www.nseindia.com/"})
        with urllib.request.urlopen(req, timeout=30) as r:
            print("NSE se download hua.")
            return r.read().decode("utf-8-sig")
    except Exception as e:
        print(f"Download nahi hua ({e}).")
        if LOCAL.exists():
            print(f"Local file use kar raha hoon: {LOCAL.name}")
            return LOCAL.read_text(encoding="utf-8-sig")
        raise SystemExit("EQUITY_L.csv is folder mein rakho (upar instructions dekho) aur dobara chalao.")


def main():
    text = fetch()
    stocks = {}
    for row in csv.DictReader(io.StringIO(text)):
        row = {(k or "").strip().upper(): (v or "").strip() for k, v in row.items()}
        sym, name = row.get("SYMBOL"), row.get("NAME OF COMPANY")
        if sym and name:
            stocks[sym] = name
    if not stocks:
        raise SystemExit("File padh nahi payi, format check karo.")
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "symbol"])
        for sym, name in sorted(stocks.items(), key=lambda x: x[1].lower()):
            w.writerow([name, sym])
    print(f"{len(stocks)} stocks -> {OUT.name}")


if __name__ == "__main__":
    main()
