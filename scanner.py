import ccxt
import pandas as pd
import ta
import requests
import schedule
import time
import json
import csv
import os
import threading
import re
from groq import Groq
import alpaca_trade_api as tradeapi
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIG — loaded from environment variables
# ============================================================
BOT_TOKEN    = os.environ.get("BOT_TOKEN")
CHAT_ID      = os.environ.get("CHAT_ID")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET  = os.environ.get("ALPACA_SECRET")
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Startup validation — crash early with a clear message
_required = {
    "BOT_TOKEN": BOT_TOKEN,
    "CHAT_ID": CHAT_ID,
    "GROQ_API_KEY": GROQ_API_KEY,
    "ALPACA_API_KEY": ALPACA_API_KEY,
    "ALPACA_SECRET": ALPACA_SECRET,
}
for _name, _val in _required.items():
    if not _val:
        raise ValueError(f"❌ Missing environment variable: {_name}")

# ============================================================
# MINIMUM QUALITY THRESHOLDS PER ASSET TYPE
# ============================================================
HIGH_QUALITY_SCORES = {
    "crypto": {"confidence": 7, "profitability": 7},
    "stock":  {"confidence": 6, "profitability": 6},
    "forex":  {"confidence": 6, "profitability": 6},
}

# ============================================================
# CSV HISTORY
# ============================================================
HISTORY_FILE = "signal_history.csv"
HISTORY_FIELDS = [
    "date", "time", "symbol", "asset_type", "signal",
    "price", "rsi", "macd", "confidence", "profitability",
    "safety", "risk", "entry", "stop_loss", "take_profit", "reason",
]

def init_csv():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writeheader()

def save_signal_to_csv(signal_data: dict):
    with csv_lock:
        with open(HISTORY_FILE, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writerow(
                {k: signal_data.get(k, "") for k in HISTORY_FIELDS}
            )

def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return list(csv.DictReader(f))

# ============================================================
# THREAD LOCKS
# ============================================================
signal_lock = threading.Lock()
csv_lock    = threading.Lock()

# ============================================================
# GLOBAL STATE
# ============================================================
all_signals: list        = []
total_signals_found: int = 0
last_scan_time: str      = "Never"

# ============================================================
# API CLIENTS
# ============================================================
exchange = ccxt.kraken()
groq_client = Groq(api_key=GROQ_API_KEY)
alpaca = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET, ALPACA_BASE_URL)

# ============================================================
# ASSET LISTS  (duplicates removed)
# ============================================================
crypto_pairs = [
    "BTC/USD",  "ETH/USD",  "SOL/USD",  "XRP/USD",  "DOGE/USD",
    "ADA/USD",  "AVAX/USD", "LINK/USD", "DOT/USD",  "ATOM/USD",
    "LTC/USD",  "UNI/USD",  "NEAR/USD", "AAVE/USD", "FIL/USD",
    "XLM/USD",  "ETC/USD",  "BCH/USD",  "ZEC/USD",  "APT/USD",
    "SUI/USD",  "OP/USD",   "ARB/USD",  "RENDER/USD","WLD/USD",
    "LDO/USD",  "GMX/USD",  "CRV/USD",  "HBAR/USD", "SEI/USD",
]

stock_pairs = [
    "AAPL",  "TSLA",  "NVDA",  "AMZN",  "META",
    "GOOGL", "MSFT",  "AMD",   "NFLX",  "COIN",
    "SPY",   "QQQ",   "DIA",   "GLD",   "SLV",
    "USO",   "BNO",   "TLT",   "ARKK",  "ARKW",
    "SOFI",  "RIOT",  "MARA",  "MSTR",  "CLSK",
    "XLK",   "XLF",   "XLE",   "XLV",   "XLI",
    "UPRO",  "TQQQ",  "SSO",   "EEM",   "VTI",
    "VOO",   "VIX",   "SCHX",  "BRK.B", "SCHB",
]

forex_pairs = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "EURCHF",
    "GBPJPY", "GBPCHF", "AUDJPY", "CADJPY", "CHFJPY",
    "EURAUD", "EURNZD", "GBPAUD", "AUDNZD", "USDSEK",
    "USDNOK", "EURSEK", "EURNOK", "USDSGD", "USDHKD",
    "AUDSGD", "EURSGD", "NZDCAD", "NZDCHF", "NZDJPY",
]

# ============================================================
# TELEGRAM
# ============================================================
def sanitize(text: str) -> str:
    """Remove Markdown special chars that break Telegram."""
    return re.sub(r"[_*\[\]()~`>#+\-=|{}.!]", "", str(text))

def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram send error: {e}")

def handle_telegram_commands():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    last_update_id = None
    while True:
        try:
            params = {"timeout": 30, "offset": last_update_id}
            resp = requests.get(url, params=params, timeout=35)
            data = resp.json()
            for update in data.get("result", []):
                last_update_id = update["update_id"] + 1
                msg  = update.get("message", {})
                text = msg.get("text", "").strip().lower()
                cid  = msg.get("chat", {}).get("id")
                if not cid:
                    continue

                if text == "/status":
                    with signal_lock:
                        n = len(all_signals)
                    reply = (
                        f"*📊 Bot Status*\n"
                        f"Last scan: `{last_scan_time}`\n"
                        f"Assets monitored: `{len(crypto_pairs)+len(stock_pairs)+len(forex_pairs)}`\n"
                        f"Signals this scan: `{n}`\n"
                        f"Total signals found: `{total_signals_found}`\n"
                        f"Scanning every: `5 minutes`\n"
                        f"Status: `✅ Running`"
                    )
                    send_telegram(reply)

                elif text == "/topsignals":
                    with signal_lock:
                        sigs = list(all_signals)
                    if not sigs:
                        send_telegram("No signals in current scan yet. Try again soon.")
                    else:
                        top = sorted(sigs, key=lambda x: x.get("profitability", 0), reverse=True)[:5]
                        reply = "*🏆 Top Signals Right Now*\n\n"
                        for i, s in enumerate(top, 1):
                            reply += (
                                f"*#{i} {s['signal']} {s['symbol']}*\n"
                                f"💰 Profit: `{s['profitability']}/10` | 🛡 Safety: `{s['safety']}/10`\n"
                                f"📍 Entry: `{s['entry']}` | SL: `{s['stop_loss']}` | TP: `{s['take_profit']}`\n"
                                f"💡 _{sanitize(s['reason'])}_\n\n"
                            )
                        send_telegram(reply)

                elif text == "/help":
                    send_telegram(
                        "*🤖 Trade Grid Analysis Bot*\n\n"
                        "Commands:\n"
                        "`/status` — Bot status\n"
                        "`/topsignals` — Top signals now\n"
                        "`/help` — This message"
                    )
        except Exception as e:
            print(f"Telegram command error: {e}")
            time.sleep(5)

# ============================================================
# DATA FETCHING
# ============================================================
def get_crypto_ohlcv(symbol: str) -> pd.DataFrame | None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=500)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"Crypto OHLCV error {symbol}: {e}")
        return None

def get_alpaca_ohlcv(symbol: str) -> pd.DataFrame | None:
    try:
        bars = alpaca.get_bars(symbol, tradeapi.rest.TimeFrame.Hour, limit=500).df
        if bars.empty:
            return None
        rename = {}
        for col in bars.columns:
            cl = col.lower()
            if cl in ("o","open"):   rename[col] = "open"
            elif cl in ("h","high"): rename[col] = "high"
            elif cl in ("l","low"):  rename[col] = "low"
            elif cl in ("c","close"):rename[col] = "close"
            elif cl in ("v","volume"):rename[col] = "volume"
        bars = bars.rename(columns=rename)
        for c in ["open","high","low","close","volume"]:
            if c not in bars.columns:
                return None
        return bars
    except Exception as e:
        print(f"Alpaca OHLCV error {symbol}: {e}")
        return None

# ============================================================
# INDICATORS
# ============================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"]  = ta.trend.ema_indicator(df["close"], window=20)
    df["ema50"]  = ta.trend.ema_indicator(df["close"], window=50)
    df["ema200"] = ta.trend.ema_indicator(df["close"], window=200)
    df["rsi"]    = ta.momentum.rsi(df["close"], window=14)
    df["vol_avg"]= df["volume"].rolling(window=20).mean()
    macd_obj     = ta.trend.MACD(df["close"])
    df["macd"]       = macd_obj.macd()
    df["macd_signal"]= macd_obj.macd_signal()
    df["macd_diff"]  = macd_obj.macd_diff()
    df["atr"]    = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    return df

# ============================================================
# AI ANALYSIS
# ============================================================
def get_ai_analysis(symbol, signal, price, ema20, ema50, ema200,
                    rsi, macd, macd_sig, volume, vol_avg, atr) -> str | None:
    prompt = (
        f"You are a professional trading analyst.\n\n"
        f"Pair: {symbol}\nSignal: {signal}\nPrice: {price}\n"
        f"EMA20: {ema20:.4f}\nEMA50: {ema50:.4f}\nEMA200: {ema200:.4f}\n"
        f"RSI: {rsi:.2f}\nMACD: {macd:.6f}\nMACD Signal: {macd_sig:.6f}\n"
        f"ATR: {atr:.4f}\nVolume: {volume:.2f} (Avg: {vol_avg:.2f})\n\n"
        f"Respond in EXACTLY this format (numbers only for scores):\n"
        f"Confidence: X/10\nProfitability: X/10\nSafety: X/10\nRisk: X/10\n"
        f"Entry: X\nStop Loss: X\nTake Profit: X\nReason: one sentence"
    )
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            timeout=15,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"AI error for {symbol}: {e}")
        return None

def parse_ai_scores(text: str) -> dict:
    scores = {
        "confidence": 5, "profitability": 5,
        "safety": 5,     "risk": 5,
        "entry": "N/A",  "stop_loss": "N/A",
        "take_profit": "N/A", "reason": "",
    }
    if not text:
        return scores
    for line in text.splitlines():
        line = line.strip()
        try:
            if line.startswith("Confidence:"):
                scores["confidence"] = int(line.split(":")[1].strip().split("/")[0])
            elif line.startswith("Profitability:"):
                scores["profitability"] = int(line.split(":")[1].strip().split("/")[0])
            elif line.startswith("Safety:"):
                scores["safety"] = int(line.split(":")[1].strip().split("/")[0])
            elif line.startswith("Risk:"):
                scores["risk"] = int(line.split(":")[1].strip().split("/")[0])
            elif line.startswith("Entry:"):
                scores["entry"] = line.split(":", 1)[1].strip()
            elif line.startswith("Stop Loss:"):
                scores["stop_loss"] = line.split(":", 1)[1].strip()
            elif line.startswith("Take Profit:"):
                scores["take_profit"] = line.split(":", 1)[1].strip()
            elif line.startswith("Reason:"):
                scores["reason"] = line.split(":", 1)[1].strip()
        except Exception:
            pass
    return scores

# ============================================================
# SIGNAL CHECK  (the core function — fully implemented)
# ============================================================
def check_signal(symbol: str, df, asset_type: str):
    global total_signals_found, all_signals

    if df is None or df.empty or len(df) < 210:
        return

    df = add_indicators(df)
    latest = df.iloc[-1]

    # Skip if any key indicator is NaN
    for col in ["ema20","ema50","ema200","rsi","macd","macd_signal","macd_diff","atr","vol_avg"]:
        if pd.isna(latest[col]):
            return

    price      = float(latest["close"])
    ema20      = float(latest["ema20"])
    ema50      = float(latest["ema50"])
    ema200     = float(latest["ema200"])
    rsi        = float(latest["rsi"])
    volume     = float(latest["volume"])
    vol_avg    = float(latest["vol_avg"])
    macd       = float(latest["macd"])
    macd_sig   = float(latest["macd_signal"])
    macd_diff  = float(latest["macd_diff"])
    atr        = float(latest["atr"])

    # ---- TECHNICAL SIGNAL LOGIC ----
    signal = None

    bullish = (
        ema20 > ema50
        and ema50 > ema200
        and rsi > 52
        and rsi < 75
        and macd_diff > 0
        and volume > vol_avg * 0.8
    )
    bearish = (
        ema20 < ema50
        and ema50 < ema200
        and rsi < 48
        and rsi > 25
        and macd_diff < 0
        and volume > vol_avg * 0.8
    )

    if bullish:
        signal = "BUY"
    elif bearish:
        signal = "SELL"
    else:
        return   # No technical setup — skip AI entirely (saves API cost)

    # ---- AI ANALYSIS (only called after tech filter passes) ----
    ai_text = get_ai_analysis(
        symbol, signal, price,
        ema20, ema50, ema200,
        rsi, macd, macd_sig,
        volume, vol_avg, atr
    )
    if ai_text is None:
        return

    scores = parse_ai_scores(ai_text)
    threshold = HIGH_QUALITY_SCORES.get(asset_type, {"confidence": 6, "profitability": 6})

    if (scores["confidence"] < threshold["confidence"]
            or scores["profitability"] < threshold["profitability"]):
        print(f"  {symbol} — below threshold "
              f"(conf {scores['confidence']}, prof {scores['profitability']})")
        return

    now = datetime.now()
    signal_label = "BUY 🟢" if signal == "BUY" else "SELL 🔴"

    signal_data = {
        # display fields
        "symbol":       symbol,
        "asset_type":   asset_type,
        "signal":       signal_label,
        "price":        round(price, 4),
        "rsi":          round(rsi, 2),
        "macd":         round(macd, 6),
        "timestamp":    now.strftime("%H:%M:%S"),
        # scores
        "confidence":   scores["confidence"],
        "profitability":scores["profitability"],
        "safety":       scores["safety"],
        "risk":         scores["risk"],
        # trade levels
        "entry":        scores["entry"],
        "stop_loss":    scores["stop_loss"],
        "take_profit":  scores["take_profit"],
        "reason":       scores["reason"],
        # csv extras
        "date":         now.strftime("%Y-%m-%d"),
        "time":         now.strftime("%H:%M:%S"),
    }

    with signal_lock:
        all_signals.append(signal_data)
        total_signals_found_ref = total_signals_found  # read before increment

    # increment outside lock (atomic in CPython, avoids holding lock)
    global total_signals_found
    total_signals_found += 1

    save_signal_to_csv(signal_data)
    print(f"  ✅ SIGNAL: {signal_label} {symbol} "
          f"| Profit {scores['profitability']}/10 "
          f"| Conf {scores['confidence']}/10")

# ============================================================
# TELEGRAM SUMMARY
# ============================================================
def send_telegram_summary(signals: list):
    if not signals:
        return
    top = sorted(signals, key=lambda x: x.get("profitability", 0), reverse=True)[:10]
    msg = (
        f"*📊 TRADE GRID ANALYSIS*\n"
        f"_{len(signals)} signal(s) — {datetime.now().strftime('%d %b %Y %H:%M')}_\n\n"
    )
    for i, s in enumerate(top, 1):
        msg += (
            f"*#{i} {s['signal']} {s['symbol']}*\n"
            f"💰 Profit: `{s['profitability']}/10` | "
            f"🛡 Safety: `{s['safety']}/10` | "
            f"⚠️ Risk: `{s['risk']}/10`\n"
            f"📍 Entry: `{s['entry']}` | SL: `{s['stop_loss']}` | TP: `{s['take_profit']}`\n"
            f"💡 _{sanitize(s['reason'])}_\n\n"
        )
    msg += "🌐 _Open Trade Grid dashboard for full rankings_"
    send_telegram(msg)

# ============================================================
# DASHBOARD HTML
# ============================================================
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade Grid Analysis</title>
<style>
:root{--bg:#0a0a0f;--surface:#111118;--surface2:#16161f;--border:#1e1e2e;--accent:#00d4ff;--green:#00c48c;--red:#ff4d6d;--yellow:#f59e0b;--text:#e2e8f0;--muted:#4a5568;--muted2:#718096}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
.header{border-bottom:1px solid var(--border);padding:0 24px;display:flex;align-items:center;justify-content:space-between;height:58px;background:var(--surface)}
.logo{display:flex;align-items:center;gap:10px;font-size:15px;font-weight:700;letter-spacing:.4px}
.logo-dot{width:8px;height:8px;background:var(--accent);border-radius:50%;box-shadow:0 0 8px var(--accent)}
.hdr-r{display:flex;align-items:center;gap:16px}
.live{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--muted2)}
.live-dot{width:6px;height:6px;background:var(--green);border-radius:50%;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.upd{font-size:11px;color:var(--muted)}
.main{padding:20px 24px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px}
.sc{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px}
.sl{font-size:10px;color:var(--muted2);text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px}
.sv{font-size:22px;font-weight:700}
.ss{font-size:10px;color:var(--muted2);margin-top:2px}
.tabs{display:flex;border-bottom:1px solid var(--border);margin-bottom:16px;overflow-x:auto}
.tab{padding:10px 16px;font-size:12px;color:var(--muted2);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;background:none;border-top:none;border-left:none;border-right:none;white-space:nowrap;transition:color .15s}
.tab:hover{color:var(--text)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-content{display:none}.tab-content.active{display:block}
.ctrl{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap;align-items:center}
.fl{font-size:11px;color:var(--muted2);margin-right:2px}
.sb{padding:5px 11px;background:var(--surface);border:1px solid var(--border);color:var(--muted2);border-radius:6px;cursor:pointer;font-size:11px;transition:all .15s;white-space:nowrap}
.sb:hover{border-color:var(--accent);color:var(--text)}
.sb.active{border-color:var(--accent);color:var(--accent);background:rgba(0,212,255,.05)}
.rb{padding:5px 11px;background:rgba(0,212,255,.1);border:1px solid var(--accent);color:var(--accent);border-radius:6px;cursor:pointer;font-size:11px;margin-left:auto}
.rb:hover{background:rgba(0,212,255,.2)}
.tw{background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12px}
thead{background:var(--surface2)}
th{padding:8px 10px;text-align:left;font-size:10px;font-weight:600;color:var(--muted2);text-transform:uppercase;letter-spacing:.6px;cursor:pointer;white-space:nowrap;border-bottom:1px solid var(--border)}
th:hover{color:var(--text)}
td{padding:8px 10px;border-bottom:1px solid var(--border);color:var(--text)}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--surface2)}
.sym{font-weight:600;font-size:12px}
.at{font-size:9px;color:var(--muted2);text-transform:uppercase}
.buy{color:var(--green);font-weight:700;font-size:11px}
.sell{color:var(--red);font-weight:700;font-size:11px}
.sp{display:inline-flex;align-items:center;justify-content:center;width:38px;height:20px;border-radius:4px;font-size:10px;font-weight:700}
.sh{background:rgba(0,196,140,.15);color:var(--green)}
.sm{background:rgba(245,158,11,.15);color:var(--yellow)}
.sl2{background:rgba(255,77,109,.15);color:var(--red)}
.rt{font-size:10px;color:var(--muted2);max-width:140px;line-height:1.3}
.ns{text-align:center;padding:50px 20px;color:var(--muted);font-size:12px}
.nsi{font-size:28px;margin-bottom:8px}
.hctrl{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap;align-items:center}
.si{padding:6px 10px;background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:6px;font-size:11px;flex:1;min-width:140px}
.si:focus{outline:none;border-color:var(--accent)}
.si::placeholder{color:var(--muted)}
.gf{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:14px}
.gf h2{font-size:13px;font-weight:600;color:var(--accent);margin-bottom:10px}
.gf p{font-size:12px;color:var(--muted2);line-height:1.6;margin-bottom:6px}
.pr{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:14px}
.pc{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px}
.pc h4{font-size:13px;font-weight:600;margin-bottom:5px}
.pc p{font-size:11px;color:var(--muted2);line-height:1.5}
.badge{display:inline-block;padding:2px 6px;border-radius:4px;font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.bg{background:rgba(0,196,140,.15);color:var(--green)}
.by{background:rgba(245,158,11,.15);color:var(--yellow)}
.steplist{display:flex;flex-direction:column;gap:8px}
.step{display:flex;gap:12px;align-items:flex-start}
.sn{min-width:22px;height:22px;background:rgba(0,212,255,.1);border:1px solid var(--accent);color:var(--accent);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;margin-top:2px}
.st{font-size:12px;color:var(--muted2);line-height:1.5}
.st strong{color:var(--text)}
.rg{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px}
.ru{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:10px 12px;font-size:11px;color:var(--muted2);line-height:1.4}
.ru strong{color:var(--text);display:block;margin-bottom:2px}
.gg{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:640px){.gg{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
  <div class="logo"><div class="logo-dot"></div>Trade Grid Analysis</div>
  <div class="hdr-r">
    <div class="live"><div class="live-dot"></div>Live</div>
    <div class="upd" id="upd">Loading...</div>
  </div>
</div>
<div class="main">
  <div class="stats">
    <div class="sc"><div class="sl">Current Signals</div><div class="sv" id="stC">0</div><div class="ss">This scan</div></div>
    <div class="sc"><div class="sl">Total History</div><div class="sv" id="stT">0</div><div class="ss">All time</div></div>
    <div class="sc"><div class="sl">Assets Monitored</div><div class="sv">90+</div><div class="ss">Crypto, Stocks, Forex</div></div>
    <div class="sc"><div class="sl">Scan Interval</div><div class="sv">5m</div><div class="ss">Auto-refresh 30s</div></div>
  </div>
  <div class="tabs">
    <button class="tab active" onclick="sw('signals',event)">📊 Live Signals</button>
    <button class="tab" onclick="sw('history',event)">📁 History</button>
    <button class="tab" onclick="sw('guide',event)">📖 How To Trade</button>
  </div>

  <!-- LIVE SIGNALS -->
  <div id="tab-signals" class="tab-content active">
    <div class="ctrl">
      <span class="fl">Sort:</span>
      <button class="sb active" onclick="st('profitability',event)">💰 Profitability</button>
      <button class="sb" onclick="st('safety',event)">🛡 Safety</button>
      <button class="sb" onclick="st('risk',event)">⚠️ Risk</button>
      <button class="sb" onclick="st('confidence',event)">🎯 Confidence</button>
      <button class="sb" onclick="st('symbol',event)">A–Z</button>
      <button class="rb" onclick="ls()">↻ Refresh</button>
    </div>
    <div class="tw"><table>
      <thead><tr>
        <th onclick="st('symbol',event)">Symbol ↕</th>
        <th onclick="st('signal',event)">Signal ↕</th>
        <th onclick="st('price',event)">Price ↕</th>
        <th onclick="st('profitability',event)">Profit ↕</th>
        <th onclick="st('safety',event)">Safety ↕</th>
        <th onclick="st('risk',event)">Risk ↕</th>
        <th onclick="st('confidence',event)">Conf ↕</th>
        <th>Entry</th><th>Stop Loss</th><th>Take Profit</th>
        <th>MACD</th><th>RSI</th><th>Reason</th>
        <th onclick="st('timestamp',event)">Time ↕</th>
      </tr></thead>
      <tbody id="tb"><tr><td colspan="14" class="ns"><div class="nsi">📡</div>Scanning markets...</td></tr></tbody>
    </table></div>
  </div>

  <!-- HISTORY -->
  <div id="tab-history" class="tab-content">
    <div class="hctrl">
      <input class="si" type="text" id="hs" placeholder="Search symbol..." oninput="fh()">
      <button class="sb" onclick="sh2('profitability',event)">💰 Profit</button>
      <button class="sb" onclick="sh2('confidence',event)">🎯 Conf</button>
      <button class="sb" onclick="sh2('date',event)">📅 Date</button>
      <button class="rb" onclick="lh()">↻ Refresh</button>
    </div>
    <div class="tw"><table>
      <thead><tr>
        <th>Date</th><th>Time</th><th>Symbol</th><th>Type</th><th>Signal</th>
        <th>Price</th><th>Profit</th><th>Safety</th><th>Risk</th><th>Conf</th>
        <th>Entry</th><th>Stop Loss</th><th>Take Profit</th><th>Reason</th>
      </tr></thead>
      <tbody id="hb"><tr><td colspan="14" class="ns"><div class="nsi">📂</div>No history yet</td></tr></tbody>
    </table></div>
  </div>

  <!-- HOW TO TRADE -->
  <div id="tab-guide" class="tab-content">
    <div class="gf">
      <h2>What This Bot Does</h2>
      <p>Trade Grid Analysis scans 90+ assets every 5 minutes across crypto, stocks and forex. Each potential trade is first filtered by technical indicators (EMA, RSI, MACD, ATR), then scored by AI. Only high quality signals reach your Telegram and dashboard.</p>
      <p>The bot tells you when to trade and gives you entry, stop loss and take profit levels. You place the trade manually on your platform.</p>
    </div>
    <div class="pr">
      <div class="pc"><div class="badge bg">Recommended</div><h4>Plus500</h4><p>Best for beginners. Supports crypto, stocks, forex, gold, oil and indices. Simple interface, CFDs, no commissions. Great for small accounts.</p></div>
      <div class="pc"><div class="badge by">Crypto Only</div><h4>Binance</h4><p>Best for crypto trading only. Very low fees, huge selection.</p></div>
      <div class="pc"><div class="badge by">Practice First</div><h4>Alpaca Paper</h4><p>Free paper trading with real market data. Practice before risking real money.</p></div>
    </div>
    <div class="gf">
      <h2>How To Read A Signal</h2>
      <div class="gg" style="margin-top:10px">
        <ul style="list-style:none;padding:0">
          <li style="padding:5px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--green)">BUY 🟢</strong> — Price trending up</li>
          <li style="padding:5px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--red)">SELL 🔴</strong> — Price trending down</li>
          <li style="padding:5px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">Entry</strong> — Open your trade here</li>
          <li style="padding:5px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">Stop Loss</strong> — Close here to limit loss</li>
          <li style="padding:5px 0;font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">Take Profit</strong> — Close here to lock profit</li>
        </ul>
        <ul style="list-style:none;padding:0">
          <li style="padding:5px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">Profitability</strong> — Profit potential (7+ is good)</li>
          <li style="padding:5px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">Safety</strong> — Trade safety (7+ is good)</li>
          <li style="padding:5px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">Risk</strong> — How risky (lower is better)</li>
          <li style="padding:5px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">Confidence</strong> — AI confidence score</li>
          <li style="padding:5px 0;font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">MACD / ATR</strong> — Trend and volatility</li>
        </ul>
      </div>
    </div>
    <div class="gf">
      <h2>Step By Step — How To Place A Trade</h2>
      <div class="steplist" style="margin-top:10px">
        <div class="step"><div class="sn">1</div><div class="st"><strong>Wait for a high quality signal</strong> — 7+ on Confidence and Profitability.</div></div>
        <div class="step"><div class="sn">2</div><div class="st"><strong>Open Plus500 and search the asset</strong> — e.g. BTC/USD, AAPL, Gold.</div></div>
        <div class="step"><div class="sn">3</div><div class="st"><strong>Set your trade size</strong> — Never risk more than R25–R50 per trade (5–10% of R500).</div></div>
        <div class="step"><div class="sn">4</div><div class="st"><strong>Set your Stop Loss</strong> — Always. No exceptions. This limits your loss automatically.</div></div>
        <div class="step"><div class="sn">5</div><div class="st"><strong>Set your Take Profit</strong> — Locks in gains automatically.</div></div>
        <div class="step"><div class="sn">6</div><div class="st"><strong>Open and wait</strong> — Trust your stop loss. Don't panic on small moves.</div></div>
      </div>
    </div>
    <div class="gf">
      <h2>Golden Rules</h2>
      <div class="rg">
        <div class="ru"><strong>Max 10% per trade</strong>Never risk more than 10% of your account on one trade.</div>
        <div class="ru"><strong>Always set a stop loss</strong>No exceptions. Ever.</div>
        <div class="ru"><strong>7/10 minimum</strong>Only trade signals scoring 7+ on confidence and profitability.</div>
        <div class="ru"><strong>Never chase losses</strong>If a trade goes wrong, step away.</div>
        <div class="ru"><strong>Keep a trading journal</strong>Note every trade and outcome.</div>
        <div class="ru"><strong>Quality over quantity</strong>2 great trades a week beats 20 bad ones.</div>
      </div>
    </div>
  </div>
</div>

<script>
let signals=[],historyData=[],sortKey='profitability',sortAsc=false,hsk='date',hsa=false;
function sw(t,e){
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(x=>x.classList.remove('active'));
  document.getElementById('tab-'+t).classList.add('active');
  if(e)e.target.classList.add('active');
  if(t==='history')lh();
}
function sc(v){v=parseInt(v);return v>=7?'sh':v>=4?'sm':'sl2'}
function sp(v){return`<span class="sp ${sc(v)}">${v}/10</span>`}
function st(k,e){
  if(sortKey===k)sortAsc=!sortAsc;else{sortKey=k;sortAsc=false;}
  document.querySelectorAll('#tab-signals .sb').forEach(b=>b.classList.remove('active'));
  if(e&&e.target.classList.contains('sb'))e.target.classList.add('active');
  rs();
}
function sh2(k){if(hsk===k)hsa=!hsa;else{hsk=k;hsa=false;}rh();}
function fh(){rh();}
function rs(){
  const s=[...signals].sort((a,b)=>{
    let av=a[sortKey],bv=b[sortKey];
    if(typeof av==='string')av=av.toLowerCase();
    if(typeof bv==='string')bv=bv.toLowerCase();
    return sortAsc?(av>bv?1:-1):(av<bv?1:-1);
  });
  const tb=document.getElementById('tb');
  if(!s.length){tb.innerHTML='<tr><td colspan="14" class="ns"><div class="nsi">📡</div>No high quality signals yet</td></tr>';return;}
  tb.innerHTML=s.map(x=>`<tr>
    <td><div class="sym">${x.symbol}</div><div class="at">${x.asset_type}</div></td>
    <td class="${x.signal.includes('BUY')?'buy':'sell'}">${x.signal}</td>
    <td>${x.price}</td>
    <td>${sp(x.profitability)}</td><td>${sp(x.safety)}</td><td>${sp(x.risk)}</td><td>${sp(x.confidence)}</td>
    <td>${x.entry}</td><td>${x.stop_loss}</td><td>${x.take_profit}</td>
    <td style="font-size:10px;color:var(--muted2)">${x.macd}</td>
    <td style="font-size:11px">${x.rsi}</td>
    <td><div class="rt">${x.reason}</div></td>
    <td style="color:var(--muted2);font-size:11px">${x.timestamp}</td>
  </tr>`).join('');
}
function rh(){
  const q=document.getElementById('hs').value.toLowerCase();
  let d=historyData.filter(x=>x.symbol&&x.symbol.toLowerCase().includes(q));
  d.sort((a,b)=>{let av=a[hsk],bv=b[hsk];if(typeof av==='string')av=av.toLowerCase();if(typeof bv==='string')bv=bv.toLowerCase();return hsa?(av>bv?1:-1):(av<bv?1:-1);});
  const tb=document.getElementById('hb');
  if(!d.length){tb.innerHTML='<tr><td colspan="14" class="ns"><div class="nsi">📂</div>No history yet</td></tr>';return;}
  tb.innerHTML=d.map(x=>`<tr>
    <td style="font-size:11px;color:var(--muted2)">${x.date}</td>
    <td style="font-size:11px;color:var(--muted2)">${x.time}</td>
    <td><div class="sym">${x.symbol}</div></td>
    <td><div class="at">${x.asset_type}</div></td>
    <td class="${x.signal&&x.signal.includes('BUY')?'buy':'sell'}">${x.signal}</td>
    <td>${x.price}</td>
    <td>${sp(x.profitability)}</td><td>${sp(x.safety)}</td><td>${sp(x.risk)}</td><td>${sp(x.confidence)}</td>
    <td>${x.entry}</td><td>${x.stop_loss}</td><td>${x.take_profit}</td>
    <td><div class="rt">${x.reason}</div></td>
  </tr>`).join('');
}
function ls(){
  fetch('/signals').then(r=>r.json()).then(d=>{
    signals=d.signals||[];
    document.getElementById('stC').textContent=signals.length;
    document.getElementById('stT').textContent=d.total||0;
    document.getElementById('upd').textContent='Updated '+new Date().toLocaleTimeString();
    rs();
  }).catch(()=>{document.getElementById('upd').textContent='Connection error';});
}
function lh(){
  fetch('/history').then(r=>r.json()).then(d=>{historyData=d;rh();});
}
ls();setInterval(ls,30000);
</script>
</body>
</html>"""

# ============================================================
# WEB SERVER
# ============================================================
class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._respond(200, "text/html", DASHBOARD_HTML.encode())
        elif self.path == "/signals":
            with signal_lock:
                payload = json.dumps({"signals": list(all_signals), "total": total_signals_found})
            self._respond(200, "application/json", payload.encode())
        elif self.path == "/history":
            data = json.dumps(load_history())
            self._respond(200, "application/json", data.encode())
        elif self.path == "/health":
            self._respond(200, "text/plain", b"OK")
        else:
            self._respond(404, "text/plain", b"Not found")

    def _respond(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

def start_dashboard():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"🌐 Dashboard → http://0.0.0.0:{port}")
    server.serve_forever()

# ============================================================
# MAIN SCANNER
# ============================================================
def scan_asset(args):
    symbol, fetch_fn, asset_type = args
    try:
        df = fetch_fn(symbol)
        if df is not None and not df.empty:
            check_signal(symbol, df, asset_type)
    except Exception as e:
        print(f"  ✗ {symbol}: {e}")

def run_scanner():
    global all_signals, last_scan_time
    last_scan_time = datetime.now().strftime("%d %b %Y %H:%M")
    print(f"\n🔍 Full scan — {last_scan_time}")

    new_signals: list = []

    # Build task list
    tasks = []
    for p in crypto_pairs:
        tasks.append((p, get_crypto_ohlcv, "crypto"))
    for p in stock_pairs:
        tasks.append((p, get_alpaca_ohlcv, "stock"))
    for p in forex_pairs:
        tasks.append((p, get_alpaca_ohlcv, "forex"))

    # Clear signals atomically before scan
    with signal_lock:
        all_signals = new_signals

    # Run with thread pool (max 10 workers to avoid rate limits)
    with ThreadPoolExecutor(max_workers=10) as ex:
        ex.map(scan_asset, tasks)

    with signal_lock:
        found = len(all_signals)

    if found:
        with signal_lock:
            sigs = list(all_signals)
        send_telegram_summary(sigs)
        print(f"✅ Scan complete — {found} signals")
    else:
        print("✅ Scan complete — no signals this round")

# ============================================================
# STARTUP
# ============================================================
if __name__ == "__main__":
    init_csv()

    threading.Thread(target=start_dashboard, daemon=True).start()
    threading.Thread(target=handle_telegram_commands, daemon=True).start()
    print("📱 Telegram commands ready (/status /topsignals /help)")

    run_scanner()

    schedule.every(5).minutes.do(run_scanner)
    print("⏰ Scanning every 5 mins — Ctrl+C to stop\n")

    while True:
        schedule.run_pending()
        time.sleep(20)