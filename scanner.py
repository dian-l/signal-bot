import ccxt
import pandas as pd
import ta
import requests
import schedule
import time
import json
import csv
import os
import logging
import hashlib
import threading
import traceback
from groq import Groq
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ── Optional Alpaca import (graceful fallback) ──────────────────────────────
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("[WARN] alpaca-trade-api not installed - stock/forex scanning disabled")

# ════════════════════════════════════════════════════════════════════════════
# LOGGING  — plain ASCII only so Railway never miscolours lines
# ════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("tradegrid.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("tradegrid")

# ════════════════════════════════════════════════════════════════════════════
# SYSTEM HEALTH TRACKER
# Every major function reports its status here so the dashboard can display it
# ════════════════════════════════════════════════════════════════════════════
health_lock = threading.Lock()
system_health = {
    # key -> { status: 'ok'|'warn'|'error'|'idle', last_run: str, message: str, count: int }
    "scanner":           {"status": "idle", "last_run": "Never",   "message": "Not yet run",          "count": 0},
    "crypto_fetch":      {"status": "idle", "last_run": "Never",   "message": "Not yet run",          "count": 0},
    "stock_fetch":       {"status": "idle", "last_run": "Never",   "message": "Not yet run",          "count": 0},
    "forex_fetch":       {"status": "idle", "last_run": "Never",   "message": "Not yet run",          "count": 0},
    "indicators":        {"status": "idle", "last_run": "Never",   "message": "Not yet run",          "count": 0},
    "ai_analysis":       {"status": "idle", "last_run": "Never",   "message": "Not yet run",          "count": 0},
    "ai_cache":          {"status": "idle", "last_run": "Never",   "message": "Cache empty",          "count": 0},
    "mtf_confirm":       {"status": "idle", "last_run": "Never",   "message": "Not yet run",          "count": 0},
    "signal_check":      {"status": "idle", "last_run": "Never",   "message": "Not yet run",          "count": 0},
    "telegram_send":     {"status": "idle", "last_run": "Never",   "message": "Not yet sent",         "count": 0},
    "telegram_commands": {"status": "idle", "last_run": "Never",   "message": "Listener not started", "count": 0},
    "paper_trade":       {"status": "idle", "last_run": "Never",   "message": "No trades yet",        "count": 0},
    "csv_write":         {"status": "idle", "last_run": "Never",   "message": "Not yet written",      "count": 0},
    "csv_read":          {"status": "idle", "last_run": "Never",   "message": "Not yet read",         "count": 0},
    "unit_tests":        {"status": "idle", "last_run": "Never",   "message": "Not yet run",          "count": 0},
    "web_server":        {"status": "idle", "last_run": "Never",   "message": "Not yet started",      "count": 0},
    "alpaca_connection": {"status": "idle", "last_run": "Never",   "message": "Not attempted",        "count": 0},
    "exchange_connection":{"status": "idle","last_run": "Never",   "message": "Not attempted",        "count": 0},
}

def update_health(key, status, message, increment=True):
    """Update a health entry. status: 'ok' | 'warn' | 'error' | 'idle'"""
    ts = datetime.now().strftime("%H:%M:%S")
    with health_lock:
        if key in system_health:
            system_health[key]["status"]   = status
            system_health[key]["last_run"] = ts
            system_health[key]["message"]  = message
            if increment:
                system_health[key]["count"] += 1
    log.info("[HEALTH] %-22s -> %-5s | %s", key, status.upper(), message)

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — all from environment variables
# ════════════════════════════════════════════════════════════════════════════
BOT_TOKEN        = os.environ.get("BOT_TOKEN")
CHAT_ID          = os.environ.get("CHAT_ID")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY")
ALPACA_API_KEY   = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET    = os.environ.get("ALPACA_SECRET")
ALPACA_BASE_URL  = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

REQUIRED_VARS = {
    "BOT_TOKEN":    BOT_TOKEN,
    "CHAT_ID":      CHAT_ID,
    "GROQ_API_KEY": GROQ_API_KEY,
}
missing = [k for k, v in REQUIRED_VARS.items() if not v]
if missing:
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

if ALPACA_AVAILABLE and (not ALPACA_API_KEY or not ALPACA_SECRET):
    log.warning("[CONFIG] ALPACA_API_KEY/SECRET not set - stock/forex disabled")
    ALPACA_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ════════════════════════════════════════════════════════════════════════════
MIN_SCORES = {
    'crypto': {'confidence': 6, 'profitability': 6},
    'stock':  {'confidence': 5, 'profitability': 5},
    'forex':  {'confidence': 5, 'profitability': 5},
}
HIGH_QUALITY_SCORES = {
    'crypto': {'confidence': 7, 'profitability': 7},
    'stock':  {'confidence': 6, 'profitability': 6},
    'forex':  {'confidence': 6, 'profitability': 6},
}

# ── Signal tiers ────────────────────────────────────────────────────────────
# STANDARD  : passes base threshold (meets HIGH_QUALITY_SCORES exactly)
# STRONG    : confidence >= 7 AND profitability >= 7 (any asset type)
# ELITE     : confidence >= 8 AND profitability >= 8 AND MTF confirmed AND RR >= 2
STRONG_THRESHOLD = {'confidence': 7, 'profitability': 7}
ELITE_THRESHOLD  = {'confidence': 8, 'profitability': 8}
ELITE_MIN_RR     = 2.0

def classify_signal_tier(confidence, profitability, rr, mtf_label):
    """
    Returns 'elite', 'strong', or 'standard'.
    Elite:    conf>=8, profit>=8, RR>=2, MTF confirmed
    Strong:   conf>=7, profit>=7
    Standard: anything else that passed the quality filter
    """
    if (confidence >= ELITE_THRESHOLD['confidence']
            and profitability >= ELITE_THRESHOLD['profitability']
            and float(rr or 0) >= ELITE_MIN_RR
            and mtf_label == 'CONFIRMED'):
        return 'elite'
    if (confidence >= STRONG_THRESHOLD['confidence']
            and profitability >= STRONG_THRESHOLD['profitability']):
        return 'strong'
    return 'standard'

# Paper trading config
MAX_DAILY_LOSS_PCT  = 0.05   # 5 % of portfolio
POSITION_RISK_PCT   = 0.02   # 2 % per trade
TRAILING_STOP_PCT   = 0.015  # 1.5 % trail
PARTIAL_TP_PCT      = 0.5    # Take 50 % off at first TP

# ════════════════════════════════════════════════════════════════════════════
# CSV / FILE PATHS
# ════════════════════════════════════════════════════════════════════════════
HISTORY_FILE  = "signal_history.csv"
TRADE_LOG     = "trade_journal.csv"
UNIT_TEST_LOG = "unit_test_results.txt"

HISTORY_FIELDS = [
    'date', 'time', 'symbol', 'asset_type', 'signal', 'price',
    'rsi', 'macd', 'atr', 'trend_strength', 'rr_ratio',
    'support', 'resistance', 'volume_spike', 'breakout',
    'confidence', 'profitability', 'safety', 'risk',
    'entry', 'stop_loss', 'take_profit', 'reason',
    'outcome', 'pnl'
]
TRADE_FIELDS = [
    'date', 'time', 'symbol', 'direction', 'qty', 'entry',
    'stop_loss', 'take_profit', 'status', 'pnl', 'notes'
]

# ════════════════════════════════════════════════════════════════════════════
# THREAD LOCKS & SHARED STATE
# ════════════════════════════════════════════════════════════════════════════
signal_lock   = threading.Lock()
csv_lock      = threading.Lock()
cache_lock    = threading.Lock()
trade_lock    = threading.Lock()

all_signals         = []
last_scan_time      = "Never"
total_signals_found = 0
daily_pnl           = 0.0
scan_count          = 0

# AI response cache  { hash(prompt_key) -> (timestamp, scores_dict) }
ai_cache    = {}
CACHE_TTL   = 300   # seconds

# ════════════════════════════════════════════════════════════════════════════
# API CLIENTS
# ════════════════════════════════════════════════════════════════════════════
exchange = ccxt.kraken()
groq_client = Groq(api_key=GROQ_API_KEY)

try:
    exchange.load_markets()
    update_health("exchange_connection", "ok", "Kraken connected and markets loaded")
except Exception as _ex_err:
    update_health("exchange_connection", "error", f"Kraken connection failed: {_ex_err}")

alpaca = None
if ALPACA_AVAILABLE:
    try:
        alpaca = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET, ALPACA_BASE_URL)
        log.info("[ALPACA] Connected to %s", ALPACA_BASE_URL)
        update_health("alpaca_connection", "ok", f"Connected to {ALPACA_BASE_URL}")
    except Exception as e:
        log.warning("[ALPACA] Connection failed: %s", e)
        update_health("alpaca_connection", "error", f"Connection failed: {e}")
        ALPACA_AVAILABLE = False
else:
    update_health("alpaca_connection", "warn", "Alpaca disabled (no API keys or package missing)", increment=False)

# ════════════════════════════════════════════════════════════════════════════
# ASSET LISTS — 50 each, 150 total
# ════════════════════════════════════════════════════════════════════════════
crypto_pairs = [
    # Original 30
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'DOGE/USD',
    'BNB/USD', 'ADA/USD', 'AVAX/USD', 'LINK/USD', 'INJ/USD',
    'FET/USD', 'ARB/USD', 'OP/USD', 'DOT/USD', 'ATOM/USD',
    'LTC/USD', 'UNI/USD', 'NEAR/USD', 'AAVE/USD', 'APT/USD',
    'SUI/USD', 'ZEC/USD', 'BCH/USD', 'ETC/USD', 'XLM/USD',
    'FIL/USD', 'HBAR/USD', 'SEI/USD', 'JUP/USD', 'WLD/USD',
    # Added 20 to reach 50
    'TRX/USD', 'TON/USD', 'SHIB/USD', 'PEPE/USD', 'BONK/USD',
    'IMX/USD', 'GRT/USD', 'LDO/USD', 'SNX/USD', 'CRV/USD',
    'ENS/USD', 'MANA/USD', 'SAND/USD', 'AXS/USD', 'BLUR/USD',
    'PENDLE/USD', 'TIA/USD', 'PYTH/USD', 'ALT/USD', 'STRK/USD',
]  # 50 total

stock_pairs = [
    # Original 30
    'AAPL', 'TSLA', 'NVDA', 'AMZN', 'META',
    'GOOGL', 'MSFT', 'AMD', 'NFLX', 'COIN',
    'SPY', 'QQQ', 'DIA', 'GLD', 'SLV',
    'USO', 'TLT', 'SOFI', 'RIOT', 'MARA',
    'MSTR', 'ARKK', 'XLK', 'XLF', 'XLE',
    'VTI', 'VOO', 'UPRO', 'TQQQ', 'BRK.B',
    # Added 20 to reach 50
    'PLTR', 'HOOD', 'RBLX', 'SNAP', 'UBER',
    'LYFT', 'ABNB', 'PINS', 'SHOP', 'PYPL',
    'SQ', 'ROKU', 'ZM', 'SNOW', 'DKNG',
    'LCID', 'RIVN', 'NIO', 'BIDU', 'BILI',
]  # 50 total

forex_pairs = [
    # Original 20
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD',
    'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY',
    'AUDJPY', 'CADJPY', 'EURAUD', 'GBPAUD', 'NZDCAD',
    'NZDCHF', 'GBPCAD', 'CADCHF', 'AUDCAD', 'AUDCHF',
    # Added 30 to reach 50
    'EURCAD', 'EURCHF', 'EURNZD', 'GBPCHF', 'GBPNZD',
    'AUDNZD', 'NZDJPY', 'CHFJPY', 'SGDJPY', 'USDHKD',
    'USDSGD', 'USDMXN', 'USDZAR', 'USDNOK', 'USDSEK',
    'USDDKK', 'USDPLN', 'USDCZK', 'USDHUF', 'USDTRY',
    'EURPLN', 'EURSEK', 'EURNOK', 'EURDKK', 'EURCZK',
    'GBPSEK', 'GBPNOK', 'GBPDKK', 'GBPPLN', 'AUDSGD',
]  # 50 total

# ════════════════════════════════════════════════════════════════════════════
# CSV HELPERS
# ════════════════════════════════════════════════════════════════════════════
def init_csv():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writeheader()
    if not os.path.exists(TRADE_LOG):
        with open(TRADE_LOG, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=TRADE_FIELDS).writeheader()
    update_health("csv_write", "ok", "CSV files initialised", increment=False)

def save_signal_to_csv(signal_data):
    try:
        with csv_lock:
            with open(HISTORY_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
                writer.writerow({k: signal_data.get(k, '') for k in HISTORY_FIELDS})
        update_health("csv_write", "ok", f"Wrote signal for {signal_data.get('symbol','?')}")
    except Exception as e:
        update_health("csv_write", "error", f"Write failed: {e}")

def save_trade_to_journal(trade_data):
    try:
        with csv_lock:
            with open(TRADE_LOG, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
                writer.writerow({k: trade_data.get(k, '') for k in TRADE_FIELDS})
        update_health("csv_write", "ok", f"Wrote trade for {trade_data.get('symbol','?')}")
    except Exception as e:
        update_health("csv_write", "error", f"Trade write failed: {e}")

def load_history():
    try:
        if not os.path.exists(HISTORY_FILE):
            return []
        with csv_lock:
            with open(HISTORY_FILE, 'r') as f:
                data = list(csv.DictReader(f))
        update_health("csv_read", "ok", f"Loaded {len(data)} history rows")
        return data
    except Exception as e:
        update_health("csv_read", "error", f"History read failed: {e}")
        return []

def load_trade_journal():
    try:
        if not os.path.exists(TRADE_LOG):
            return []
        with csv_lock:
            with open(TRADE_LOG, 'r') as f:
                data = list(csv.DictReader(f))
        update_health("csv_read", "ok", f"Loaded {len(data)} journal rows")
        return data
    except Exception as e:
        update_health("csv_read", "error", f"Journal read failed: {e}")
        return []

# ════════════════════════════════════════════════════════════════════════════
# WIN RATE / STATS HELPERS
# ════════════════════════════════════════════════════════════════════════════
def compute_stats(history):
    total = len(history)
    if total == 0:
        return {'win_rate': 0, 'total': 0, 'wins': 0, 'losses': 0, 'avg_pnl': 0, 'top_assets': []}

    wins   = [h for h in history if str(h.get('outcome', '')).upper() == 'WIN']
    losses = [h for h in history if str(h.get('outcome', '')).upper() == 'LOSS']

    pnl_vals = []
    for h in history:
        try:
            pnl_vals.append(float(h.get('pnl', 0) or 0))
        except (ValueError, TypeError):
            pass

    asset_counts = defaultdict(int)
    for h in history:
        sym = h.get('symbol', '')
        if sym:
            asset_counts[sym] += 1
    top_assets = sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        'win_rate':   round(len(wins) / total * 100, 1) if total else 0,
        'total':      total,
        'wins':       len(wins),
        'losses':     len(losses),
        'avg_pnl':    round(sum(pnl_vals) / len(pnl_vals), 4) if pnl_vals else 0,
        'top_assets': [{'symbol': a[0], 'count': a[1]} for a in top_assets],
    }

# ════════════════════════════════════════════════════════════════════════════
# API RETRY WRAPPER
# ════════════════════════════════════════════════════════════════════════════
def with_retry(fn, retries=3, delay=2, label=""):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt < retries - 1:
                log.warning("[RETRY] Attempt %d/%d for %s: %s", attempt+1, retries, label, e)
                time.sleep(delay * (attempt + 1))
            else:
                log.error("[RETRY] All retries failed for %s: %s", label, e)
                return None

# ════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ════════════════════════════════════════════════════════════════════════════
def get_crypto_ohlcv(symbol, timeframe='1h', limit=500):
    def _fetch():
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    result = with_retry(_fetch, label=symbol)
    if result is not None:
        update_health("crypto_fetch", "ok", f"Fetched {symbol} ({timeframe})")
    else:
        update_health("crypto_fetch", "warn", f"Failed to fetch {symbol} ({timeframe})")
    return result

def get_crypto_ohlcv_multi(symbol, timeframes=('1h', '4h')):
    results = {}
    for tf in timeframes:
        df = get_crypto_ohlcv(symbol, timeframe=tf)
        if df is not None and not df.empty:
            results[tf] = df
    return results

def get_alpaca_ohlcv(symbol):
    if not alpaca:
        return None
    def _fetch():
        bars = alpaca.get_bars(symbol, tradeapi.rest.TimeFrame.Hour, limit=500).df
        col_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        bars = bars.rename(columns={k: v for k, v in col_map.items() if k in bars.columns})
        return bars
    result = with_retry(_fetch, label=symbol)
    if result is not None:
        update_health("stock_fetch", "ok", f"Fetched stock {symbol}")
    else:
        update_health("stock_fetch", "warn", f"Failed to fetch stock {symbol}")
    return result

def get_forex_ohlcv(symbol):
    if not alpaca:
        return None
    def _fetch():
        bars = alpaca.get_bars(symbol, tradeapi.rest.TimeFrame.Hour, limit=500).df
        if bars.empty:
            return None
        col_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        bars = bars.rename(columns={k: v for k, v in col_map.items() if k in bars.columns})
        return bars
    result = with_retry(_fetch, label=symbol)
    if result is not None:
        update_health("forex_fetch", "ok", f"Fetched forex {symbol}")
    else:
        update_health("forex_fetch", "warn", f"Failed to fetch forex {symbol}")
    return result

# ════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ════════════════════════════════════════════════════════════════════════════
def add_indicators(df):
    try:
        df = df.copy()
        df['ema20']  = ta.trend.ema_indicator(df['close'], window=20)
        df['ema50']  = ta.trend.ema_indicator(df['close'], window=50)
        df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
        df['rsi']    = ta.momentum.rsi(df['close'], window=14)
        df['vol_avg'] = df['volume'].rolling(window=20).mean()

        macd_obj          = ta.trend.MACD(df['close'])
        df['macd']        = macd_obj.macd()
        df['macd_signal'] = macd_obj.macd_signal()
        df['macd_diff']   = macd_obj.macd_diff()

        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid']   = bb.bollinger_mavg()

        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx']     = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()

        df['vol_spike']   = df['volume'] / df['vol_avg']
        df['support']     = df['low'].rolling(20).min()
        df['resistance']  = df['high'].rolling(20).max()

        update_health("indicators", "ok", "Indicators computed successfully")
        return df
    except Exception as e:
        update_health("indicators", "error", f"Indicator error: {e}")
        raise

def calc_trend_strength(adx, ema20, ema50, ema200, rsi, macd_diff):
    score = 0
    if adx > 25:   score += 2
    if adx > 40:   score += 1
    if ema20 > ema50 > ema200 or ema20 < ema50 < ema200:
        score += 3
    if rsi > 50 or rsi < 50:  score += 1
    if abs(macd_diff) > 0:    score += 1
    score += min(2, adx / 30)
    return round(min(score, 10), 1)

def calc_rr_ratio(price, stop_loss, take_profit, signal):
    try:
        if signal == 'BUY':
            risk   = abs(price - stop_loss)
            reward = abs(take_profit - price)
        else:
            risk   = abs(stop_loss - price)
            reward = abs(price - take_profit)
        return round(reward / risk, 2) if risk > 0 else 0
    except Exception:
        return 0

def detect_breakout(price, bb_upper, bb_lower, resistance, support):
    if price > bb_upper and price > resistance:
        return "BULLISH_BREAKOUT"
    if price < bb_lower and price < support:
        return "BEARISH_BREAKOUT"
    return "NONE"

def multi_timeframe_confirm(symbol, primary_signal, asset_type):
    """
    Fetch 4h data and check if higher-timeframe trend agrees with signal.
    Returns True if confirmed, False if contradicted, None if unavailable.
    NOTE: MTF rejection is a normal INFO event — not a warning.
    """
    try:
        if asset_type == 'crypto':
            df4h = get_crypto_ohlcv(symbol, timeframe='4h', limit=300)
        else:
            update_health("mtf_confirm", "ok", f"{symbol}: N/A (non-crypto)")
            return None
        if df4h is None or df4h.empty or len(df4h) < 60:
            update_health("mtf_confirm", "warn", f"{symbol}: insufficient 4h data")
            return None
        df4h = add_indicators(df4h)
        last = df4h.iloc[-1]
        if pd.isna(last['ema50']) or pd.isna(last['ema200']):
            update_health("mtf_confirm", "warn", f"{symbol}: NaN in 4h EMAs")
            return None
        htf_bull = last['ema50'] > last['ema200'] and last['rsi'] > 45
        htf_bear = last['ema50'] < last['ema200'] and last['rsi'] < 55
        if primary_signal == 'BUY' and htf_bull:
            update_health("mtf_confirm", "ok", f"{symbol}: CONFIRMED (BUY aligns 4h)")
            return True
        if primary_signal == 'SELL' and htf_bear:
            update_health("mtf_confirm", "ok", f"{symbol}: CONFIRMED (SELL aligns 4h)")
            return True
        # MTF rejection is NORMAL and expected — log at INFO, not WARNING
        update_health("mtf_confirm", "ok",
                      f"{symbol}: rejected - {primary_signal} on 1h contradicts 4h trend (normal filter)")
        return False
    except Exception as e:
        update_health("mtf_confirm", "error", f"{symbol}: exception - {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS  (with cache)
# ════════════════════════════════════════════════════════════════════════════
def _cache_key(symbol, signal, rsi, macd):
    raw = f"{symbol}|{signal}|{round(rsi,1)}|{round(macd,6)}"
    return hashlib.md5(raw.encode()).hexdigest()

def get_ai_analysis(symbol, signal, price, ema20, ema50, ema200,
                    rsi, macd, macd_signal, volume, vol_avg,
                    atr, trend_strength, support, resistance, breakout):
    key = _cache_key(symbol, signal, rsi, macd)
    now = time.time()

    with cache_lock:
        if key in ai_cache:
            ts, cached = ai_cache[key]
            if now - ts < CACHE_TTL:
                log.info("[AI] Cache hit for %s", symbol)
                update_health("ai_cache", "ok", f"Cache hit for {symbol}", increment=True)
                return cached

    prompt = f"""You are a professional trading analyst.

Analyze this trade setup and respond in EXACTLY this format (numbers only for scores):

Pair: {symbol}
Signal: {signal}
Price: {price}
EMA20: {ema20:.4f} | EMA50: {ema50:.4f} | EMA200: {ema200:.4f}
RSI: {rsi:.1f}
MACD: {macd:.6f} (Signal: {macd_signal:.6f})
ATR: {atr:.4f}
Trend Strength: {trend_strength}/10
Volume: {volume:.0f} (Avg: {vol_avg:.0f})
Support: {support:.4f} | Resistance: {resistance:.4f}
Breakout: {breakout}

Respond EXACTLY:
Confidence: X/10
Profitability: X/10
Safety: X/10
Risk: X/10
Entry: X
Stop Loss: X
Take Profit: X
Reason: One sentence only"""

    def _call():
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3,
        )
        return resp.choices[0].message.content

    raw = with_retry(_call, label=f"AI:{symbol}")
    if raw is None:
        update_health("ai_analysis", "error", f"AI call failed for {symbol}")
        return None

    scores = parse_ai_scores(raw)
    update_health("ai_analysis", "ok", f"AI scored {symbol}: C={scores['confidence']} P={scores['profitability']}")

    with cache_lock:
        ai_cache[key] = (now, scores)
    update_health("ai_cache", "ok", f"Cached result for {symbol}", increment=False)

    return scores

def parse_ai_scores(ai_text):
    scores = {
        'confidence': 5, 'profitability': 5,
        'safety': 5, 'risk': 5,
        'entry': 0, 'stop_loss': 0,
        'take_profit': 0, 'reason': ''
    }
    if not ai_text:
        return scores
    try:
        for line in ai_text.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, _, val = line.partition(':')
            key = key.strip()
            val = val.strip()
            if key == 'Confidence':
                scores['confidence'] = int(val.split('/')[0])
            elif key == 'Profitability':
                scores['profitability'] = int(val.split('/')[0])
            elif key == 'Safety':
                scores['safety'] = int(val.split('/')[0])
            elif key == 'Risk':
                scores['risk'] = int(val.split('/')[0])
            elif key == 'Entry':
                scores['entry'] = val
            elif key == 'Stop Loss':
                scores['stop_loss'] = val
            elif key == 'Take Profit':
                scores['take_profit'] = val
            elif key == 'Reason':
                scores['reason'] = val
    except Exception:
        pass
    return scores

# ════════════════════════════════════════════════════════════════════════════
# PAPER TRADING — Alpaca auto-trade
# ════════════════════════════════════════════════════════════════════════════
def get_portfolio_value():
    if not alpaca:
        return 1000.0
    try:
        acct = alpaca.get_account()
        return float(acct.portfolio_value)
    except Exception:
        return 1000.0

def calc_position_size(price, stop_loss, portfolio_value):
    risk_per_share = abs(price - stop_loss)
    if risk_per_share <= 0:
        return 1
    dollar_risk = portfolio_value * POSITION_RISK_PCT
    qty = int(dollar_risk / risk_per_share)
    return max(qty, 1)

def place_paper_trade(signal_data):
    global daily_pnl
    if not alpaca or signal_data['asset_type'] != 'stock':
        return

    with trade_lock:
        portfolio_value = get_portfolio_value()

        if daily_pnl < -(portfolio_value * MAX_DAILY_LOSS_PCT):
            log.warning("[TRADE] Max daily loss reached - no new trades today")
            update_health("paper_trade", "warn", "Max daily loss protection triggered")
            send_telegram("*[TRADE]* Max daily loss protection triggered. No new paper trades today.")
            return

        symbol = signal_data['symbol']
        try:
            price = float(signal_data['price'])
            sl    = float(str(signal_data['stop_loss']).replace(',', '') or 0)
            tp    = float(str(signal_data['take_profit']).replace(',', '') or 0)
        except (ValueError, TypeError):
            update_health("paper_trade", "warn", f"Invalid price data for {symbol}")
            return

        if sl <= 0 or tp <= 0:
            update_health("paper_trade", "warn", f"Invalid SL/TP for {symbol}")
            return

        qty = calc_position_size(price, sl, portfolio_value)
        side = 'buy' if signal_data['signal'] == 'BUY' else 'sell'

        try:
            trail_pct = TRAILING_STOP_PCT * 100
            alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day',
                trail_percent=trail_pct,
            )
            log.info("[TRADE] Paper %s %dx %s @ ~%s", side.upper(), qty, symbol, price)
            update_health("paper_trade", "ok", f"Opened {side.upper()} {qty}x {symbol} @ {price}")

            trade_rec = {
                'date':       datetime.now().strftime('%Y-%m-%d'),
                'time':       datetime.now().strftime('%H:%M:%S'),
                'symbol':     symbol,
                'direction':  side.upper(),
                'qty':        qty,
                'entry':      price,
                'stop_loss':  sl,
                'take_profit':tp,
                'status':     'OPEN',
                'pnl':        '',
                'notes':      f'Auto paper trade | trail {trail_pct}%',
            }
            save_trade_to_journal(trade_rec)

            send_telegram(
                f"*Paper Trade Opened*\n"
                f"`{side.upper()} {qty}x {symbol}`\n"
                f"Entry: `{price}` | SL: `{sl}` | TP: `{tp}`\n"
                f"Trail stop: `{trail_pct}%`"
            )
        except Exception as e:
            log.error("[TRADE] Paper trade error for %s: %s", symbol, e)
            update_health("paper_trade", "error", f"Order failed for {symbol}: {e}")

# ════════════════════════════════════════════════════════════════════════════
# SIGNAL CHECK — full pipeline
# ════════════════════════════════════════════════════════════════════════════
def check_signal(symbol, df, asset_type):
    global total_signals_found

    if df is None or df.empty or len(df) < 210:
        return None

    try:
        df = add_indicators(df)
    except Exception as e:
        log.warning("[SIGNAL] Indicator error %s: %s", symbol, e)
        return None

    latest = df.iloc[-1]

    if any(pd.isna(latest[col]) for col in ['ema200', 'rsi', 'macd', 'atr', 'adx']):
        return None

    price      = float(latest['close'])
    ema20      = float(latest['ema20'])
    ema50      = float(latest['ema50'])
    ema200     = float(latest['ema200'])
    rsi        = float(latest['rsi'])
    volume     = float(latest['volume'])
    vol_avg    = float(latest['vol_avg'])
    macd       = float(latest['macd'])
    macd_sig   = float(latest['macd_signal'])
    macd_diff  = float(latest['macd_diff'])
    atr        = float(latest['atr'])
    adx        = float(latest['adx'])
    vol_spike  = float(latest['vol_spike']) if not pd.isna(latest['vol_spike']) else 1.0
    support    = float(latest['support'])
    resistance = float(latest['resistance'])
    bb_upper   = float(latest['bb_upper'])
    bb_lower   = float(latest['bb_lower'])

    trend_strength = calc_trend_strength(adx, ema20, ema50, ema200, rsi, macd_diff)
    breakout       = detect_breakout(price, bb_upper, bb_lower, resistance, support)
    volume_spike   = vol_spike >= 1.5

    signal = None

    buy_score = 0
    if ema20 > ema50:                              buy_score += 2
    if ema50 > ema200:                             buy_score += 1
    if rsi > 50 and rsi < 70:                      buy_score += 2
    if macd > macd_sig:                            buy_score += 2
    if volume > vol_avg * 0.8:                     buy_score += 1
    if adx > 20:                                   buy_score += 1
    if breakout == 'BULLISH_BREAKOUT':             buy_score += 1
    if volume_spike:                               buy_score += 1
    if buy_score >= 7:
        signal = 'BUY'

    sell_score = 0
    if ema20 < ema50:                              sell_score += 2
    if ema50 < ema200:                             sell_score += 1
    if rsi < 50 and rsi > 30:                      sell_score += 2
    if macd < macd_sig:                            sell_score += 2
    if volume > vol_avg * 0.8:                     sell_score += 1
    if adx > 20:                                   sell_score += 1
    if breakout == 'BEARISH_BREAKOUT':             sell_score += 1
    if volume_spike:                               sell_score += 1
    if sell_score >= 7:
        signal = 'SELL'

    if not signal:
        return None

    update_health("signal_check", "ok", f"{symbol}: {signal} candidate (score B={buy_score} S={sell_score})")

    # MTF confirmation — rejection is a normal INFO outcome, not an error
    mtf = multi_timeframe_confirm(symbol, signal, asset_type)
    mtf_label = {True: 'CONFIRMED', False: 'REJECTED', None: 'N/A'}[mtf]
    if mtf is False:
        # This is expected behaviour — log at INFO level (not warning)
        log.info("[MTF] %s: %s signal filtered out - 1h vs 4h disagreement (normal)", symbol, signal)
        return None

    scores = get_ai_analysis(
        symbol, signal, price, ema20, ema50, ema200,
        rsi, macd, macd_sig, volume, vol_avg,
        atr, trend_strength, support, resistance, breakout
    )
    if scores is None:
        return None

    try:
        sl_price = float(str(scores['stop_loss']).replace(',', '') or 0)
        tp_price = float(str(scores['take_profit']).replace(',', '') or 0)
        rr = calc_rr_ratio(price, sl_price, tp_price, signal)
    except (ValueError, TypeError):
        rr = 0

    threshold     = HIGH_QUALITY_SCORES[asset_type]
    confidence    = scores['confidence']
    profitability = scores['profitability']

    if confidence < threshold['confidence'] or profitability < threshold['profitability']:
        log.info("[SIGNAL] %s below quality threshold (conf=%d, profit=%d)", symbol, confidence, profitability)
        return None

    tier = classify_signal_tier(confidence, profitability, rr, mtf_label)

    signal_data = {
        'timestamp':     datetime.now().strftime('%H:%M:%S'),
        'date':          datetime.now().strftime('%Y-%m-%d'),
        'time':          datetime.now().strftime('%H:%M:%S'),
        'symbol':        symbol,
        'asset_type':    asset_type,
        'signal':        signal,
        'tier':          tier,
        'price':         round(price, 4),
        'rsi':           round(rsi, 2),
        'macd':          round(macd, 6),
        'atr':           round(atr, 6),
        'trend_strength':trend_strength,
        'rr_ratio':      rr,
        'support':       round(support, 4),
        'resistance':    round(resistance, 4),
        'volume_spike':  volume_spike,
        'breakout':      breakout,
        'mtf':           mtf_label,
        'confidence':    confidence,
        'profitability': profitability,
        'safety':        scores['safety'],
        'risk':          scores['risk'],
        'entry':         scores['entry'],
        'stop_loss':     scores['stop_loss'],
        'take_profit':   scores['take_profit'],
        'reason':        scores['reason'],
        'outcome':       '',
        'pnl':           '',
    }

    with signal_lock:
        total_signals_found += 1

    save_signal_to_csv(signal_data)
    place_paper_trade(signal_data)

    # Tier-tagged log lines — easy to grep
    tier_tag = {"elite": "[ELITE]", "strong": "[STRONG]", "standard": "[SIGNAL]"}[tier]
    log.info("%s PASS %s %s | P:%d C:%d RR:%.2f MTF:%s",
             tier_tag, signal, symbol, profitability, confidence, rr, mtf_label)
    return signal_data

# ════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ════════════════════════════════════════════════════════════════════════════
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=10
        )
        if r.status_code == 200:
            update_health("telegram_send", "ok", f"Sent message ({len(message)} chars)")
        else:
            update_health("telegram_send", "warn", f"HTTP {r.status_code} from Telegram")
    except Exception as e:
        log.error("[TELEGRAM] Send failed: %s", e)
        update_health("telegram_send", "error", f"Send failed: {e}")

def _fmt_signal_block(s, idx):
    """Format a single signal block for Telegram (shared by all tiers)."""
    rr_str  = f" | RR `{s['rr_ratio']}`" if s.get('rr_ratio') else ''
    mtf_str = f" | MTF `{s.get('mtf', 'N/A')}`"
    return (
        f"*#{idx} {s['signal']} {s['symbol']}*\n"
        f"P:`{s['profitability']}/10` S:`{s['safety']}/10` "
        f"R:`{s['risk']}/10` C:`{s['confidence']}/10`\n"
        f"Entry:`{s['entry']}` SL:`{s['stop_loss']}` TP:`{s['take_profit']}`"
        f"{rr_str}{mtf_str}\n"
        f"Trend:`{s.get('trend_strength','?')}/10` "
        f"Breakout:`{s.get('breakout','?')}`\n"
        f"_{s['reason']}_\n\n"
    )

def send_telegram_summary(signals):
    if not signals:
        return

    elite    = [s for s in signals if s.get('tier') == 'elite']
    strong   = [s for s in signals if s.get('tier') == 'strong']
    standard = [s for s in signals if s.get('tier') == 'standard']

    ts = datetime.now().strftime('%d %b %Y %H:%M')

    # ── ELITE signals — one dedicated alert per signal ──────────────────────
    for s in sorted(elite, key=lambda x: x['profitability'], reverse=True):
        dir_arrow = "BUY  (LONG)" if s['signal'] == 'BUY' else "SELL (SHORT)"
        msg = (
            f"*=== ELITE SIGNAL ===*\n"
            f"*{dir_arrow}: {s['symbol']}*\n"
            f"_{ts}_\n\n"
            f"*Scores*\n"
            f"Profitability: `{s['profitability']}/10`\n"
            f"Confidence:    `{s['confidence']}/10`\n"
            f"Safety:        `{s['safety']}/10`\n"
            f"Risk:          `{s['risk']}/10`\n"
            f"Trend:         `{s.get('trend_strength','?')}/10`\n"
            f"RR Ratio:      `{s.get('rr_ratio','?')}`\n\n"
            f"*Levels*\n"
            f"Entry:      `{s['entry']}`\n"
            f"Stop Loss:  `{s['stop_loss']}`\n"
            f"Take Profit:`{s['take_profit']}`\n\n"
            f"MTF: `{s.get('mtf','?')}` | Breakout: `{s.get('breakout','?')}`\n\n"
            f"*Reason:* _{s['reason']}_\n\n"
            f"*This is a top-tier setup. Consider sizing up.*"
        )
        send_telegram(msg)

    # ── STRONG signals — grouped in one message ──────────────────────────────
    if strong:
        strong_sorted = sorted(strong, key=lambda x: x['profitability'], reverse=True)
        msg = (
            f"*-- STRONG SIGNALS --*\n"
            f"_{ts} | {len(strong)} setup{'s' if len(strong)!=1 else ''}_\n\n"
        )
        for i, s in enumerate(strong_sorted[:6], 1):
            msg += _fmt_signal_block(s, i)
        msg += "_Strong setups — confirmed trend alignment and solid scores._"
        send_telegram(msg)

    # ── STANDARD signals — compact digest ───────────────────────────────────
    if standard:
        standard_sorted = sorted(standard, key=lambda x: x['profitability'], reverse=True)
        msg = (
            f"*Trade Grid — Scan Results*\n"
            f"_{ts} | {len(standard)} standard signal{'s' if len(standard)!=1 else ''}_\n\n"
        )
        for i, s in enumerate(standard_sorted[:5], 1):
            rr_str = f" RR:`{s['rr_ratio']}`" if s.get('rr_ratio') else ''
            msg += (
                f"*{s['signal']} {s['symbol']}* — "
                f"P:`{s['profitability']}/10` C:`{s['confidence']}/10`{rr_str}\n"
                f"Entry:`{s['entry']}` SL:`{s['stop_loss']}` TP:`{s['take_profit']}`\n"
                f"_{s['reason']}_\n\n"
            )
        msg += "_Open the dashboard for the full list._"
        send_telegram(msg)

    # ── Summary line if multiple tiers fired ────────────────────────────────
    tier_parts = []
    if elite:    tier_parts.append(f"{len(elite)} elite")
    if strong:   tier_parts.append(f"{len(strong)} strong")
    if standard: tier_parts.append(f"{len(standard)} standard")
    if len(tier_parts) > 1:
        send_telegram(
            f"*Scan complete:* {' | '.join(tier_parts)} signal{'s' if len(signals)!=1 else ''} found."
        )

def handle_telegram_commands():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    last_update_id = None
    update_health("telegram_commands", "ok", "Telegram command listener started", increment=False)

    while True:
        try:
            params = {'timeout': 30, 'offset': last_update_id}
            response = requests.get(url, params=params, timeout=35)
            data = response.json()
            update_health("telegram_commands", "ok",
                          f"Polling active ({len(data.get('result',[]))} updates)")

            for update in data.get('result', []):
                last_update_id = update['update_id'] + 1
                message = update.get('message', {})
                text    = message.get('text', '').strip()
                chat_id = message.get('chat', {}).get('id')
                if not chat_id:
                    continue

                cmd = text.lower().split()[0] if text else ''

                if cmd == '/status':
                    reply = (
                        f"*Bot Status*\n"
                        f"Last scan: `{last_scan_time}`\n"
                        f"Assets: `{len(crypto_pairs) + len(stock_pairs) + len(forex_pairs)}`\n"
                        f"Current signals: `{len(all_signals)}`\n"
                        f"Total found: `{total_signals_found}`\n"
                        f"Daily P&L: `{daily_pnl:.4f}`\n"
                        f"Scans run: `{scan_count}`\n"
                        f"Status: Running"
                    )
                    send_telegram(reply)

                elif cmd == '/topsignals':
                    if not all_signals:
                        send_telegram("No signals in current scan. Wait for next scan.")
                    else:
                        top = sorted(all_signals, key=lambda x: x['profitability'], reverse=True)[:5]
                        reply = "*Top Signals*\n\n"
                        for i, s in enumerate(top, 1):
                            reply += (
                                f"*#{i} {s['signal']} {s['symbol']}*\n"
                                f"P:`{s['profitability']}/10` C:`{s['confidence']}/10` "
                                f"RR:`{s.get('rr_ratio','?')}`\n"
                                f"Entry:`{s['entry']}` SL:`{s['stop_loss']}` TP:`{s['take_profit']}`\n"
                                f"_{s['reason']}_\n\n"
                            )
                        send_telegram(reply)

                elif cmd == '/besttoday':
                    history = load_history()
                    today = datetime.now().strftime('%Y-%m-%d')
                    today_signals = [h for h in history if h.get('date') == today]
                    if not today_signals:
                        send_telegram("No signals recorded today yet.")
                    else:
                        best = sorted(today_signals, key=lambda x: float(x.get('profitability', 0) or 0), reverse=True)[:5]
                        reply = f"*Best Signals Today ({today})*\n\n"
                        for i, s in enumerate(best, 1):
                            reply += f"*#{i} {s.get('signal')} {s.get('symbol')}* - P:`{s.get('profitability')}/10`\n"
                        send_telegram(reply)

                elif cmd == '/wins':
                    history = load_history()
                    wins = [h for h in history if str(h.get('outcome', '')).upper() == 'WIN']
                    reply = f"*Wins: {len(wins)} total*\n"
                    for s in wins[-5:]:
                        reply += f"`{s.get('symbol')}` {s.get('signal')} P&L:`{s.get('pnl','?')}`\n"
                    send_telegram(reply)

                elif cmd == '/losses':
                    history = load_history()
                    losses = [h for h in history if str(h.get('outcome', '')).upper() == 'LOSS']
                    reply = f"*Losses: {len(losses)} total*\n"
                    for s in losses[-5:]:
                        reply += f"`{s.get('symbol')}` {s.get('signal')} P&L:`{s.get('pnl','?')}`\n"
                    send_telegram(reply)

                elif cmd == '/history':
                    history = load_history()
                    if not history:
                        send_telegram("No history yet.")
                    else:
                        stats = compute_stats(history)
                        reply = (
                            f"*Signal History*\n"
                            f"Total: `{stats['total']}`\n"
                            f"Win rate: `{stats['win_rate']}%`\n"
                            f"Wins: `{stats['wins']}` | Losses: `{stats['losses']}`\n"
                            f"Avg P&L: `{stats['avg_pnl']}`\n"
                            f"Top assets: {', '.join(a['symbol'] for a in stats['top_assets'][:3])}"
                        )
                        send_telegram(reply)

                elif cmd == '/help':
                    reply = (
                        "*Trade Grid Analysis*\n\n"
                        "`/status` - Bot status\n"
                        "`/topsignals` - Current top signals\n"
                        "`/besttoday` - Best signals today\n"
                        "`/wins` - Win history\n"
                        "`/losses` - Loss history\n"
                        "`/history` - Stats summary\n"
                        "`/help` - This message\n\n"
                        "Or type any asset symbol (e.g. BTC/USD) to look it up."
                    )
                    send_telegram(reply)

                else:
                    sym = text.strip().upper()
                    if sym:
                        history = load_history()
                        matches = [h for h in history if h.get('symbol','').upper() == sym]
                        if matches:
                            recent = matches[-3:]
                            reply = f"*{sym} - last {len(recent)} signal(s)*\n\n"
                            for s in reversed(recent):
                                reply += (
                                    f"`{s.get('date')} {s.get('time')}` "
                                    f"{s.get('signal')} P:`{s.get('profitability')}/10`\n"
                                )
                            send_telegram(reply)
                        else:
                            send_telegram(f"No signals found for `{sym}` yet.")

        except Exception as e:
            log.error("[TELEGRAM] Command listener error: %s", e)
            update_health("telegram_commands", "error", f"Listener error: {e}")
            time.sleep(5)

# ════════════════════════════════════════════════════════════════════════════
# MAIN SCANNER
# ════════════════════════════════════════════════════════════════════════════
def scan_asset(args):
    symbol, asset_type, fetch_fn = args
    try:
        df = fetch_fn(symbol)
        if df is None or df.empty:
            return None
        return check_signal(symbol, df, asset_type)
    except Exception as e:
        log.warning("[SCAN] Error scanning %s: %s", symbol, e)
        return None

def run_scanner():
    global all_signals, last_scan_time, scan_count, daily_pnl

    if datetime.now().hour == 0 and datetime.now().minute < 3:
        daily_pnl = 0.0

    scan_count += 1
    last_scan_time = datetime.now().strftime('%d %b %Y %H:%M')
    log.info("[SCAN] Starting scan #%d at %s", scan_count, last_scan_time)
    update_health("scanner", "ok", f"Scan #{scan_count} started")

    tasks = []
    for sym in crypto_pairs:
        tasks.append((sym, 'crypto', get_crypto_ohlcv))
    if ALPACA_AVAILABLE:
        for sym in stock_pairs:
            tasks.append((sym, 'stock', get_alpaca_ohlcv))
        for sym in forex_pairs:
            tasks.append((sym, 'forex', get_forex_ohlcv))

    new_signals = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(scan_asset, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                new_signals.append(result)

    with signal_lock:
        all_signals = new_signals

    if all_signals:
        send_telegram_summary(all_signals)
        log.info("[SCAN] Complete - %d high quality signals found", len(all_signals))
        update_health("scanner", "ok",
                      f"Scan #{scan_count} complete - {len(all_signals)} signals from {len(tasks)} assets")
    else:
        log.info("[SCAN] Complete - no high quality signals this scan")
        update_health("scanner", "ok",
                      f"Scan #{scan_count} complete - 0 signals (normal, filters are strict)")

    log.info("[SCAN] Monitored %d assets total", len(tasks))

# ════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ════════════════════════════════════════════════════════════════════════════
def run_unit_tests():
    results = []
    try:
        import numpy as np
        n     = 300
        close = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        high  = close + abs(np.random.randn(n))
        low   = close - abs(np.random.randn(n))
        vol   = pd.Series(np.random.randint(1000, 5000, n), dtype=float)
        df = pd.DataFrame({'open': close, 'high': high, 'low': low,
                           'close': close, 'volume': vol})
        df = add_indicators(df)

        def test(name, cond):
            status = "PASS" if cond else "FAIL"
            results.append({'name': name, 'status': status})
            return cond

        test("EMA20 computed",      not df['ema20'].isnull().all())
        test("EMA50 computed",      not df['ema50'].isnull().all())
        test("EMA200 computed",     not df['ema200'].isnull().all())
        test("RSI range 0-100",     df['rsi'].dropna().between(0, 100).all())
        test("MACD computed",       not df['macd'].isnull().all())
        test("ATR positive",        (df['atr'].dropna() >= 0).all())
        test("ADX positive",        (df['adx'].dropna() >= 0).all())
        test("Support <= close",    (df['support'].dropna() <= df['close'][df['support'].notna()]).all())
        test("Resistance >= close", (df['resistance'].dropna() >= df['close'][df['resistance'].notna()]).all())
        test("Vol spike computed",  not df['vol_spike'].isnull().all())
        test("BB upper > lower",    (df['bb_upper'].dropna() > df['bb_lower'].dropna()).all())

        passed = sum(1 for r in results if r['status'] == 'PASS')
        total  = len(results)
        log.info("[TESTS] %d/%d passed", passed, total)

        if passed == total:
            update_health("unit_tests", "ok", f"All {total} tests passed")
        else:
            fails = [r['name'] for r in results if r['status'] == 'FAIL']
            update_health("unit_tests", "warn", f"{passed}/{total} passed. Failed: {', '.join(fails)}")

        with open(UNIT_TEST_LOG, 'w') as f:
            f.write(f"Unit test run: {datetime.now()}\n")
            for r in results:
                f.write(f"{r['status']}  {r['name']}\n")

        return results

    except Exception as e:
        log.error("[TESTS] Unit test exception: %s", e)
        update_health("unit_tests", "error", f"Test runner exception: {e}")
        return []

# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD HTML
# ════════════════════════════════════════════════════════════════════════════
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade Grid Analysis</title>
<style>
:root{
  --bg:#0a0a0f;--surface:#111118;--surface2:#16161f;--border:#1e1e2e;
  --accent:#00d4ff;--accent2:#7c3aed;--green:#00c48c;--red:#ff4d6d;
  --yellow:#f59e0b;--text:#e2e8f0;--muted:#4a5568;--muted2:#718096;
}
.light{
  --bg:#f0f4f8;--surface:#fff;--surface2:#f7fafc;--border:#e2e8f0;
  --accent:#0077cc;--accent2:#6d28d9;--green:#059669;--red:#dc2626;
  --yellow:#d97706;--text:#1a202c;--muted:#a0aec0;--muted2:#718096;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;transition:background .2s,color .2s}
.header{border-bottom:1px solid var(--border);padding:0 16px;display:flex;align-items:center;justify-content:space-between;height:60px;background:var(--surface)}
.logo{display:flex;align-items:center;gap:10px;font-size:14px;font-weight:700;letter-spacing:.5px}
.logo-dot{width:8px;height:8px;background:var(--accent);border-radius:50%;box-shadow:0 0 8px var(--accent)}
.header-right{display:flex;align-items:center;gap:12px}
.status-badge{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--muted2)}
.status-dot{width:6px;height:6px;background:var(--green);border-radius:50%;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.last-update{font-size:11px;color:var(--muted)}
.theme-btn{padding:4px 10px;background:var(--surface2);border:1px solid var(--border);color:var(--muted2);border-radius:6px;cursor:pointer;font-size:11px}
.theme-btn:hover{border-color:var(--accent);color:var(--accent)}
.main{padding:16px}
.stats-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 16px}
.stat-label{font-size:10px;color:var(--muted2);text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px}
.stat-value{font-size:20px;font-weight:700;color:var(--text)}
.stat-sub{font-size:10px;color:var(--muted2);margin-top:2px}
.stat-good{color:var(--green)}.stat-bad{color:var(--red)}.stat-mid{color:var(--yellow)}
.tabs{display:flex;gap:0;margin-bottom:16px;border-bottom:1px solid var(--border);overflow-x:auto}
.tab{padding:10px 12px;font-size:12px;color:var(--muted2);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;transition:all .15s;background:none;border-top:none;border-left:none;border-right:none;white-space:nowrap}
.tab:hover{color:var(--text)}.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-content{display:none}.tab-content.active{display:block}
.controls{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap;align-items:center}
.filter-label{font-size:11px;color:var(--muted2);margin-right:4px}
.sort-btn{padding:4px 10px;background:var(--surface);border:1px solid var(--border);color:var(--muted2);border-radius:6px;cursor:pointer;font-size:11px;transition:all .15s;white-space:nowrap}
.sort-btn:hover{border-color:var(--accent);color:var(--text)}.sort-btn.active{border-color:var(--accent);color:var(--accent);background:rgba(0,212,255,.05)}
.filter-select{padding:4px 8px;background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:6px;font-size:11px;cursor:pointer}
.refresh-btn{padding:4px 10px;background:rgba(0,212,255,.1);border:1px solid var(--accent);color:var(--accent);border-radius:6px;cursor:pointer;font-size:11px;margin-left:auto}
.refresh-btn:hover{background:rgba(0,212,255,.2)}
.table-wrap{background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12px}
thead{background:var(--surface2)}
th{padding:8px 10px;text-align:left;font-size:10px;font-weight:600;color:var(--muted2);text-transform:uppercase;letter-spacing:.6px;cursor:pointer;white-space:nowrap;border-bottom:1px solid var(--border)}
th:hover{color:var(--text)}
td{padding:8px 10px;border-bottom:1px solid var(--border);color:var(--text);font-size:12px}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--surface2)}
.symbol{font-weight:600;font-size:12px}.asset-type{font-size:9px;color:var(--muted2);text-transform:uppercase;letter-spacing:.5px}
.buy{color:var(--green);font-weight:600;font-size:11px}.sell{color:var(--red);font-weight:600;font-size:11px}
.score-pill{display:inline-flex;align-items:center;justify-content:center;width:38px;height:20px;border-radius:4px;font-size:10px;font-weight:700}
.score-high{background:rgba(0,196,140,.15);color:var(--green)}.score-mid{background:rgba(245,158,11,.15);color:var(--yellow)}.score-low{background:rgba(255,77,109,.15);color:var(--red)}
.badge-pill{display:inline-block;padding:2px 6px;border-radius:10px;font-size:9px;font-weight:600}
.badge-bull{background:rgba(0,196,140,.15);color:var(--green)}.badge-bear{background:rgba(255,77,109,.15);color:var(--red)}.badge-none{background:var(--surface2);color:var(--muted2)}
.reason-text{font-size:10px;color:var(--muted2);max-width:160px;line-height:1.3}
.no-signals{text-align:center;padding:40px 16px;color:var(--muted);font-size:12px}
.no-signals-icon{font-size:28px;margin-bottom:8px}
.search-input{padding:6px 10px;background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:6px;font-size:12px;flex:1;min-width:150px}
.search-input:focus{outline:none;border-color:var(--accent)}.search-input::placeholder{color:var(--muted)}
.wr-bar{height:8px;background:var(--surface2);border-radius:4px;overflow:hidden;margin-top:4px}
.wr-fill{height:100%;background:var(--green);border-radius:4px;transition:width .4s}

/* ── SIGNAL TIER STYLES ────────────────────────────────────────────── */
/* Tier badge pill */
.tier-badge{display:inline-flex;align-items:center;gap:4px;padding:2px 7px;border-radius:10px;font-size:9px;font-weight:700;letter-spacing:.4px;text-transform:uppercase}
.tier-elite   {background:linear-gradient(90deg,rgba(255,215,0,.18),rgba(255,165,0,.12));border:1px solid rgba(255,200,0,.45);color:#ffd700}
.tier-strong  {background:rgba(124,58,237,.14);border:1px solid rgba(124,58,237,.4);color:#a78bfa}
.tier-standard{background:rgba(113,128,150,.1);border:1px solid rgba(113,128,150,.25);color:var(--muted2)}

/* Elite row — golden shimmer left border + subtle glow */
tr.row-elite{
  background:linear-gradient(90deg,rgba(255,200,0,.06) 0%,transparent 60%) !important;
  border-left:3px solid #ffd700;
  box-shadow:inset 0 0 20px rgba(255,200,0,.04);
}
tr.row-elite td:first-child{padding-left:9px}
tr.row-elite:hover{background:linear-gradient(90deg,rgba(255,200,0,.1) 0%,rgba(255,255,255,.02) 60%) !important}

/* Strong row — purple accent left border */
tr.row-strong{
  background:linear-gradient(90deg,rgba(124,58,237,.06) 0%,transparent 60%) !important;
  border-left:3px solid #7c3aed;
}
tr.row-strong td:first-child{padding-left:9px}
tr.row-strong:hover{background:linear-gradient(90deg,rgba(124,58,237,.11) 0%,rgba(255,255,255,.02) 60%) !important}

/* Standard rows stay default */

/* Elite signal pulse animation on the tier badge */
@keyframes elite-pulse{
  0%,100%{box-shadow:0 0 0 0 rgba(255,200,0,.0)}
  50%{box-shadow:0 0 6px 2px rgba(255,200,0,.35)}
}
.tier-elite{animation:elite-pulse 2.8s ease-in-out infinite}

/* Tier filter bar */
.tier-filter-btn{padding:4px 10px;background:var(--surface);border:1px solid var(--border);color:var(--muted2);border-radius:20px;cursor:pointer;font-size:11px;transition:all .15s;white-space:nowrap}
.tier-filter-btn:hover{border-color:var(--accent);color:var(--text)}
.tier-filter-btn.active-all     {border-color:var(--accent);color:var(--accent);background:rgba(0,212,255,.06)}
.tier-filter-btn.active-elite   {border-color:#ffd700;color:#ffd700;background:rgba(255,200,0,.08)}
.tier-filter-btn.active-strong  {border-color:#7c3aed;color:#a78bfa;background:rgba(124,58,237,.08)}
.tier-filter-btn.active-standard{border-color:var(--muted);color:var(--muted2);background:rgba(113,128,150,.06)}

/* Tier count badges in the stat row */
.tier-count-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.tier-count-card{display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:8px;border:1px solid;flex:1;min-width:100px}
.tcc-elite  {background:rgba(255,200,0,.06);border-color:rgba(255,200,0,.25)}
.tcc-strong {background:rgba(124,58,237,.06);border-color:rgba(124,58,237,.25)}
.tcc-standard{background:rgba(113,128,150,.05);border-color:rgba(113,128,150,.2)}
.tcc-num{font-size:22px;font-weight:700}
.tcc-elite   .tcc-num{color:#ffd700}
.tcc-strong  .tcc-num{color:#a78bfa}
.tcc-standard .tcc-num{color:var(--muted2)}
.tcc-lbl{font-size:10px;color:var(--muted2);text-transform:uppercase;letter-spacing:.6px}
.guide-full{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:16px}
.guide-full h2{font-size:14px;font-weight:600;color:var(--accent);margin-bottom:10px}
.guide-full p{font-size:12px;color:var(--muted2);line-height:1.6;margin-bottom:8px}
.guide-grid{display:grid;grid-template-columns:1fr;gap:16px}
@media(min-width:768px){.guide-grid{grid-template-columns:1fr 1fr}}
.guide-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px}
.guide-card h3{font-size:13px;font-weight:600;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.guide-card p,.guide-card li{font-size:12px;color:var(--muted2);line-height:1.6}
.guide-card ul{padding-left:16px}.guide-card li{margin-bottom:4px}.guide-card li strong{color:var(--text)}
.platform-row{display:grid;grid-template-columns:1fr;gap:12px;margin-bottom:16px}
@media(min-width:768px){.platform-row{grid-template-columns:repeat(3,1fr)}}
.platform-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px}
.platform-card h4{font-size:13px;font-weight:600;margin-bottom:6px}
.platform-card p{font-size:11px;color:var(--muted2);line-height:1.5}
.badge{display:inline-block;padding:2px 6px;border-radius:4px;font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.badge-green{background:rgba(0,196,140,.15);color:var(--green)}.badge-yellow{background:rgba(245,158,11,.15);color:var(--yellow)}
.step-list{display:flex;flex-direction:column;gap:10px}
.step{display:flex;gap:12px;align-items:flex-start}
.step-num{min-width:24px;height:24px;background:rgba(0,212,255,.1);border:1px solid var(--accent);color:var(--accent);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;margin-top:2px}
.step-text{font-size:12px;color:var(--muted2);line-height:1.5}.step-text strong{color:var(--text)}
.rules-grid{display:grid;grid-template-columns:1fr;gap:8px}
@media(min-width:768px){.rules-grid{grid-template-columns:1fr 1fr}}
.rule{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:10px 12px;font-size:11px;color:var(--muted2);line-height:1.4}
.rule strong{color:var(--text);display:block;margin-bottom:3px}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:20px}
.big-stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px}
.big-stat-val{font-size:28px;font-weight:700;margin-bottom:4px}
.big-stat-lbl{font-size:11px;color:var(--muted2);text-transform:uppercase;letter-spacing:.6px}
.top-assets-table td{padding:6px 10px;font-size:12px}

/* ── SYSTEM HEALTH TAB ─────────────────────────────────────────── */
.health-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:12px}
.health-card{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:10px;
  padding:14px 16px;
  display:flex;
  align-items:flex-start;
  gap:14px;
  transition:border-color .2s;
}
.health-card:hover{border-color:var(--muted)}
.health-card.status-ok    {border-left:3px solid var(--green)}
.health-card.status-warn  {border-left:3px solid var(--yellow)}
.health-card.status-error {border-left:3px solid var(--red)}
.health-card.status-idle  {border-left:3px solid var(--muted)}
.health-icon{
  width:34px;height:34px;border-radius:8px;
  display:flex;align-items:center;justify-content:center;
  font-size:16px;flex-shrink:0;margin-top:1px;
}
.health-card.status-ok   .health-icon{background:rgba(0,196,140,.12)}
.health-card.status-warn .health-icon{background:rgba(245,158,11,.12)}
.health-card.status-error .health-icon{background:rgba(255,77,109,.12)}
.health-card.status-idle .health-icon{background:rgba(113,128,150,.1)}
.health-body{flex:1;min-width:0}
.health-name{font-size:12px;font-weight:600;color:var(--text);margin-bottom:3px;display:flex;align-items:center;gap:8px}
.health-msg{font-size:11px;color:var(--muted2);line-height:1.4;word-break:break-word}
.health-meta{display:flex;align-items:center;gap:10px;margin-top:6px}
.health-time{font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums}
.health-count{font-size:10px;color:var(--muted2);background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:1px 5px}
.status-dot-sm{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.status-dot-sm.ok   {background:var(--green);box-shadow:0 0 5px var(--green)}
.status-dot-sm.warn {background:var(--yellow);box-shadow:0 0 5px var(--yellow)}
.status-dot-sm.error{background:var(--red);box-shadow:0 0 5px var(--red)}
.status-dot-sm.idle {background:var(--muted)}
.health-banner{
  display:flex;align-items:center;gap:12px;
  background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:14px 18px;margin-bottom:16px;
}
.health-banner-icon{font-size:22px}
.health-banner-text h3{font-size:14px;font-weight:700;margin-bottom:2px}
.health-banner-text p{font-size:11px;color:var(--muted2)}
.health-summary-pills{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.h-pill{
  display:flex;align-items:center;gap:6px;
  padding:5px 12px;border-radius:20px;
  font-size:11px;font-weight:600;border:1px solid;
}
.h-pill-ok   {background:rgba(0,196,140,.08);border-color:rgba(0,196,140,.3);color:var(--green)}
.h-pill-warn {background:rgba(245,158,11,.08);border-color:rgba(245,158,11,.3);color:var(--yellow)}
.h-pill-error{background:rgba(255,77,109,.08);border-color:rgba(255,77,109,.3);color:var(--red)}
.h-pill-idle {background:rgba(113,128,150,.08);border-color:rgba(113,128,150,.3);color:var(--muted2)}
.health-section-title{font-size:10px;color:var(--muted2);text-transform:uppercase;letter-spacing:.8px;margin:16px 0 8px;font-weight:600}

@media(max-width:480px){.header{padding:0 12px}.logo{font-size:12px}.main{padding:12px}.stat-value{font-size:18px}table{font-size:11px}th,td{padding:6px 8px}.reason-text{max-width:100px;font-size:9px}.health-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
  <div class="logo"><div class="logo-dot"></div>Trade Grid Analysis</div>
  <div class="header-right">
    <div class="status-badge"><div class="status-dot"></div>Live</div>
    <div class="last-update" id="lastUpdate">Loading...</div>
    <button class="theme-btn" onclick="toggleTheme()">Light / Dark</button>
  </div>
</div>

<div class="main">
  <div class="stats-row">
    <div class="stat-card"><div class="stat-label">Current Signals</div><div class="stat-value" id="statCurrent">0</div><div class="stat-sub">This scan</div></div>
    <div class="stat-card"><div class="stat-label">Total History</div><div class="stat-value" id="statTotal">0</div><div class="stat-sub">All time</div></div>
    <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value" id="statWinRate">—</div><div class="wr-bar"><div class="wr-fill" id="wrFill" style="width:0%"></div></div></div>
    <div class="stat-card"><div class="stat-label">Total P&amp;L</div><div class="stat-value" id="statPnl">—</div><div class="stat-sub">Paper trades</div></div>
    <div class="stat-card"><div class="stat-label">Assets Monitored</div><div class="stat-value" id="statAssets">150</div><div class="stat-sub">50 Crypto + 50 Stocks + 50 Forex</div></div>
    <div class="stat-card"><div class="stat-label">Scan Interval</div><div class="stat-value">2.5m</div><div class="stat-sub">Auto-refresh 30s</div></div>
  </div>

  <div class="tabs">
    <button class="tab active"  onclick="switchTab('signals',event)">Live Signals</button>
    <button class="tab"         onclick="switchTab('history',event)">History</button>
    <button class="tab"         onclick="switchTab('performance',event)">Performance</button>
    <button class="tab"         onclick="switchTab('journal',event)">Trade Journal</button>
    <button class="tab"         onclick="switchTab('health',event)">System Health</button>
    <button class="tab"         onclick="switchTab('guide',event)">How To Trade</button>
  </div>

  <!-- ══ LIVE SIGNALS ══════════════════════════════════════════════ -->
  <div id="tab-signals" class="tab-content active">
    <div class="controls">
      <span class="filter-label">Sort:</span>
      <button class="sort-btn active" onclick="sortTable('profitability',event)">Profit</button>
      <button class="sort-btn" onclick="sortTable('safety',event)">Safety</button>
      <button class="sort-btn" onclick="sortTable('risk',event)">Risk</button>
      <button class="sort-btn" onclick="sortTable('confidence',event)">Confidence</button>
      <button class="sort-btn" onclick="sortTable('rr_ratio',event)">RR</button>
      <button class="sort-btn" onclick="sortTable('trend_strength',event)">Trend</button>
      <select class="filter-select" id="typeFilter" onchange="renderSignals()">
        <option value="">All types</option>
        <option value="crypto">Crypto</option>
        <option value="stock">Stock</option>
        <option value="forex">Forex</option>
      </select>
      <select class="filter-select" id="signalFilter" onchange="renderSignals()">
        <option value="">BUY &amp; SELL</option>
        <option value="BUY">BUY only</option>
        <option value="SELL">SELL only</option>
      </select>
      <button class="refresh-btn" onclick="loadSignals()">Refresh</button>
    </div>
    <div class="table-wrap"><table>
      <thead><tr>
        <th onclick="sortTable('symbol',event)">Symbol</th>
        <th onclick="sortTable('signal',event)">Signal</th>
        <th onclick="sortTable('price',event)">Price</th>
        <th onclick="sortTable('profitability',event)">Profit</th>
        <th onclick="sortTable('safety',event)">Safety</th>
        <th onclick="sortTable('risk',event)">Risk</th>
        <th onclick="sortTable('confidence',event)">Conf</th>
        <th onclick="sortTable('rr_ratio',event)">RR</th>
        <th onclick="sortTable('trend_strength',event)">Trend</th>
        <th>Breakout</th><th>MTF</th>
        <th>Entry</th><th>Stop Loss</th><th>Take Profit</th>
        <th>RSI</th><th>MACD</th><th>ATR</th>
        <th>Reason</th>
        <th onclick="sortTable('timestamp',event)">Time</th>
      </tr></thead>
      <tbody id="tableBody"><tr><td colspan="19" class="no-signals"><div class="no-signals-icon">📡</div>Scanning markets…</td></tr></tbody>
    </table></div>
  </div>

  <!-- ══ HISTORY ═══════════════════════════════════════════════════ -->
  <div id="tab-history" class="tab-content">
    <div class="controls">
      <input class="search-input" type="text" id="historySearch" placeholder="Search symbol…" oninput="renderHistory()">
      <select class="filter-select" id="histTypeFilter" onchange="renderHistory()">
        <option value="">All types</option>
        <option value="crypto">Crypto</option>
        <option value="stock">Stock</option>
        <option value="forex">Forex</option>
      </select>
      <button class="sort-btn" onclick="sortHistory('profitability',event)">Profit</button>
      <button class="sort-btn" onclick="sortHistory('confidence',event)">Confidence</button>
      <button class="sort-btn" onclick="sortHistory('date',event)">Date</button>
      <button class="refresh-btn" onclick="loadHistory()">Refresh</button>
    </div>
    <div class="table-wrap"><table>
      <thead><tr>
        <th>Date</th><th>Time</th><th>Symbol</th><th>Type</th><th>Signal</th><th>Price</th>
        <th>Profit</th><th>Safety</th><th>Risk</th><th>Conf</th><th>RR</th><th>Trend</th>
        <th>Breakout</th><th>Entry</th><th>SL</th><th>TP</th><th>Outcome</th><th>P&amp;L</th><th>Reason</th>
      </tr></thead>
      <tbody id="historyBody"><tr><td colspan="19" class="no-signals"><div class="no-signals-icon">📂</div>No history yet</td></tr></tbody>
    </table></div>
  </div>

  <!-- ══ PERFORMANCE ════════════════════════════════════════════════ -->
  <div id="tab-performance" class="tab-content">
    <div class="stats-grid">
      <div class="big-stat"><div class="big-stat-val" id="perfWR">—</div><div class="big-stat-lbl">Win Rate</div></div>
      <div class="big-stat"><div class="big-stat-val" id="perfWins">—</div><div class="big-stat-lbl">Total Wins</div></div>
      <div class="big-stat"><div class="big-stat-val" id="perfLosses">—</div><div class="big-stat-lbl">Total Losses</div></div>
      <div class="big-stat"><div class="big-stat-val" id="perfAvgPnl">—</div><div class="big-stat-lbl">Avg P&amp;L per Trade</div></div>
      <div class="big-stat"><div class="big-stat-val" id="perfTotal">—</div><div class="big-stat-lbl">Total Signals</div></div>
    </div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px">
      <div class="stat-label" style="margin-bottom:10px">Top Performing Assets</div>
      <table style="width:100%;border-collapse:collapse">
        <thead><tr><th style="text-align:left;font-size:10px;color:var(--muted2);padding:6px 10px">Symbol</th><th style="text-align:left;font-size:10px;color:var(--muted2)">Count</th></tr></thead>
        <tbody id="topAssetsBody"><tr><td colspan="2" style="color:var(--muted2);padding:10px;font-size:12px">No data yet</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- ══ TRADE JOURNAL ══════════════════════════════════════════════ -->
  <div id="tab-journal" class="tab-content">
    <div class="controls">
      <input class="search-input" type="text" id="journalSearch" placeholder="Search symbol…" oninput="renderJournal()">
      <button class="refresh-btn" onclick="loadJournal()">Refresh</button>
    </div>
    <div class="table-wrap"><table>
      <thead><tr>
        <th>Date</th><th>Time</th><th>Symbol</th><th>Dir</th><th>Qty</th>
        <th>Entry</th><th>Stop Loss</th><th>Take Profit</th><th>Status</th><th>P&amp;L</th><th>Notes</th>
      </tr></thead>
      <tbody id="journalBody"><tr><td colspan="11" class="no-signals"><div class="no-signals-icon">📒</div>No paper trades yet</td></tr></tbody>
    </table></div>
  </div>

  <!-- ══ SYSTEM HEALTH ══════════════════════════════════════════════ -->
  <div id="tab-health" class="tab-content">

    <!-- Overall banner -->
    <div class="health-banner" id="healthBanner">
      <div class="health-banner-icon" id="healthBannerIcon">⬤</div>
      <div class="health-banner-text">
        <h3 id="healthBannerTitle">Checking system…</h3>
        <p id="healthBannerSub">Loading health data from backend</p>
      </div>
      <button class="refresh-btn" style="margin-left:auto" onclick="loadHealth()">Refresh</button>
    </div>

    <!-- Summary pills -->
    <div class="health-summary-pills" id="healthPills"></div>

    <!-- Cards -->
    <div class="health-section-title">All Functions</div>
    <div class="health-grid" id="healthGrid">
      <div style="color:var(--muted2);font-size:12px;padding:20px">Loading health data…</div>
    </div>

    <!-- Auto-refresh note -->
    <div style="margin-top:16px;font-size:10px;color:var(--muted);text-align:right">
      Auto-refreshes every 15 seconds &nbsp;|&nbsp; Last fetched: <span id="healthLastFetch">—</span>
    </div>
  </div>

  <!-- ══ HOW TO TRADE ════════════════════════════════════════════════ -->
  <div id="tab-guide" class="tab-content">
    <div class="guide-full"><h2>What This Bot Does</h2>
      <p>Trade Grid Analysis scans 150 assets every 2.5 minutes across crypto, stocks, and forex using EMA, RSI, MACD, ATR, ADX, Bollinger Bands, Support/Resistance and multi-timeframe confirmation. Only setups passing the technical filter are sent to AI for scoring. High quality signals (crypto 7/10, stocks/forex 6/10) appear here and in your Telegram.</p>
    </div>
    <div class="platform-row">
      <div class="platform-card"><div class="badge badge-green">Recommended</div><h4>Plus500</h4><p>Best for beginners. Crypto, stocks, forex, gold, oil and indices in one place. Simple interface, no commissions, CFDs. Great for small accounts.</p></div>
      <div class="platform-card"><div class="badge badge-yellow">Crypto Only</div><h4>Binance</h4><p>Best for crypto trading only. Very low fees, huge selection. Focus purely on crypto signals.</p></div>
      <div class="platform-card"><div class="badge badge-yellow">Practice First</div><h4>Alpaca Paper</h4><p>Free paper trading with real market data. Practice with fake money before risking real funds. Highly recommended.</p></div>
    </div>
    <div class="guide-full"><h2>New Indicators Explained</h2>
      <div class="guide-grid" style="margin-top:12px">
        <div><ul style="list-style:none;padding:0">
          <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">ATR (Avg True Range)</strong> — Measures volatility. Higher ATR = bigger price swings. Used to size stop losses.</li>
          <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">ADX / Trend Strength</strong> — Scores trend power 0-10. Above 25 = strong trend. Bot requires trend for signals.</li>
          <li style="padding:6px 0;font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">Risk/Reward (RR)</strong> — Potential profit divided by potential loss. Aim for RR 2 or higher.</li>
        </ul></div>
        <div><ul style="list-style:none;padding:0">
          <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">Breakout</strong> — Price breaking above resistance or below support. Bullish/Bearish breakout = strong signal.</li>
          <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">Volume Spike</strong> — Volume 50% above average. Confirms real momentum behind a move.</li>
          <li style="padding:6px 0;font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">MTF (Multi-Timeframe)</strong> — CONFIRMED = 4h trend agrees with 1h signal. REJECTED = contradicts (normal filter).</li>
        </ul></div>
      </div>
    </div>
    <div class="guide-full"><h2>Step By Step — How To Place A Trade</h2>
      <div class="step-list" style="margin-top:12px">
        <div class="step"><div class="step-num">1</div><div class="step-text"><strong>Wait for CONFIRMED + high RR signals</strong> — Look for MTF: CONFIRMED, RR 2.0 or higher, Confidence 7/10 or higher.</div></div>
        <div class="step"><div class="step-num">2</div><div class="step-text"><strong>Open your platform and search the asset</strong> — e.g. BTC/USD on Plus500, AAPL on Alpaca.</div></div>
        <div class="step"><div class="step-num">3</div><div class="step-text"><strong>Size your trade</strong> — Risk 2-5% of account per trade. With R500 that is R10-R25.</div></div>
        <div class="step"><div class="step-num">4</div><div class="step-text"><strong>Set Stop Loss and Take Profit</strong> — Use the exact values the bot provides. Always. No exceptions.</div></div>
        <div class="step"><div class="step-num">5</div><div class="step-text"><strong>Open the trade and let it run</strong> — Trust your levels. Do not move your stop loss.</div></div>
        <div class="step"><div class="step-num">6</div><div class="step-text"><strong>Journal every trade</strong> — The Trade Journal tab auto-records paper trades. Add manual trades too.</div></div>
      </div>
    </div>
    <div class="guide-full"><h2>Golden Rules</h2>
      <div class="rules-grid">
        <div class="rule"><strong>Max 5% per trade</strong>Never risk more than 5% of your account on one trade</div>
        <div class="rule"><strong>Always set a stop loss</strong>No exceptions. Ever. Protects your account.</div>
        <div class="rule"><strong>Only trade CONFIRMED signals</strong>MTF confirmed + RR 2 or higher = much higher win rate</div>
        <div class="rule"><strong>Never chase losses</strong>If a trade goes wrong, step back. Do not revenge trade.</div>
        <div class="rule"><strong>Aim for RR 2 or higher</strong>You can be wrong 40% of the time and still be profitable</div>
        <div class="rule"><strong>Quality over quantity</strong>2 great trades a week beats 20 bad ones every time</div>
      </div>
    </div>
  </div>
</div>

<script>
let signals=[], historyData=[], journalData=[], healthData={};
let sortKey='profitability', sortAsc=false;
let historySortKey='date', historySortAsc=false;
let theme='dark';

function toggleTheme(){
  theme = theme==='dark' ? 'light' : 'dark';
  document.body.classList.toggle('light', theme==='light');
}

function switchTab(tab, e){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-'+tab).classList.add('active');
  if(e) e.target.classList.add('active');
  if(tab==='history')     loadHistory();
  if(tab==='performance') loadPerformance();
  if(tab==='journal')     loadJournal();
  if(tab==='health')      loadHealth();
}

function sc(score){
  score=parseInt(score)||0;
  let cls = score>=7?'score-high':score>=4?'score-mid':'score-low';
  return `<span class="score-pill ${cls}">${score}/10</span>`;
}

function breakoutBadge(b){
  if(b==='BULLISH_BREAKOUT') return '<span class="badge-pill badge-bull">Bull</span>';
  if(b==='BEARISH_BREAKOUT') return '<span class="badge-pill badge-bear">Bear</span>';
  return '<span class="badge-pill badge-none">—</span>';
}

function mtfBadge(m){
  if(m==='CONFIRMED') return '<span class="badge-pill badge-bull">Confirmed</span>';
  if(m==='REJECTED')  return '<span class="badge-pill badge-bear">Rejected</span>';
  return '<span class="badge-pill badge-none">N/A</span>';
}

function sortTable(key, e){
  if(sortKey===key) sortAsc=!sortAsc; else{sortKey=key; sortAsc=false;}
  document.querySelectorAll('#tab-signals .sort-btn').forEach(b=>b.classList.remove('active'));
  if(e&&e.target.classList.contains('sort-btn')) e.target.classList.add('active');
  renderSignals();
}

function sortHistory(key, e){
  if(historySortKey===key) historySortAsc=!historySortAsc; else{historySortKey=key; historySortAsc=false;}
  renderHistory();
}

function renderSignals(){
  const typeF = document.getElementById('typeFilter').value;
  const sigF  = document.getElementById('signalFilter').value;
  let filtered = signals.filter(s=>
    (!typeF || s.asset_type===typeF) &&
    (!sigF  || s.signal===sigF)
  );
  filtered.sort((a,b)=>{
    let av=a[sortKey], bv=b[sortKey];
    if(typeof av==='string') av=av.toLowerCase();
    if(typeof bv==='string') bv=bv.toLowerCase();
    return sortAsc?(av>bv?1:-1):(av<bv?1:-1);
  });
  const tbody=document.getElementById('tableBody');
  if(!filtered.length){
    tbody.innerHTML=`<tr><td colspan="19" class="no-signals"><div class="no-signals-icon">📡</div>No high quality signals yet — scanning every 2.5 mins</td></tr>`;
    return;
  }
  tbody.innerHTML=filtered.map(s=>`<tr>
    <td><div class="symbol">${s.symbol}</div><div class="asset-type">${s.asset_type}</div></td>
    <td class="${s.signal==='BUY'?'buy':'sell'}">${s.signal}</td>
    <td>${s.price}</td>
    <td>${sc(s.profitability)}</td><td>${sc(s.safety)}</td><td>${sc(s.risk)}</td><td>${sc(s.confidence)}</td>
    <td style="font-size:11px;font-weight:600;color:${parseFloat(s.rr_ratio||0)>=2?'var(--green)':'var(--yellow)'}">${s.rr_ratio||'—'}</td>
    <td>${sc(s.trend_strength)}</td>
    <td>${breakoutBadge(s.breakout)}</td>
    <td>${mtfBadge(s.mtf)}</td>
    <td style="font-size:11px">${s.entry}</td>
    <td style="font-size:11px">${s.stop_loss}</td>
    <td style="font-size:11px">${s.take_profit}</td>
    <td style="font-size:11px;color:var(--muted2)">${s.rsi}</td>
    <td style="font-size:10px;color:var(--muted2)">${s.macd}</td>
    <td style="font-size:10px;color:var(--muted2)">${s.atr}</td>
    <td><div class="reason-text">${s.reason}</div></td>
    <td style="color:var(--muted2);font-size:11px">${s.timestamp}</td>
  </tr>`).join('');
}

function renderHistory(){
  const search = document.getElementById('historySearch').value.toLowerCase();
  const typeF  = document.getElementById('histTypeFilter').value;
  let filtered = historyData.filter(h=>
    (!search || (h.symbol||'').toLowerCase().includes(search)) &&
    (!typeF  || h.asset_type===typeF)
  );
  filtered.sort((a,b)=>{
    let av=a[historySortKey]||'', bv=b[historySortKey]||'';
    if(typeof av==='string') av=av.toLowerCase();
    if(typeof bv==='string') bv=bv.toLowerCase();
    return historySortAsc?(av>bv?1:-1):(av<bv?1:-1);
  });
  const tbody=document.getElementById('historyBody');
  if(!filtered.length){
    tbody.innerHTML=`<tr><td colspan="19" class="no-signals"><div class="no-signals-icon">📂</div>No history yet</td></tr>`;
    return;
  }
  tbody.innerHTML=filtered.map(s=>`<tr>
    <td style="font-size:11px;color:var(--muted2)">${s.date}</td>
    <td style="font-size:11px;color:var(--muted2)">${s.time}</td>
    <td><div class="symbol">${s.symbol}</div></td>
    <td><div class="asset-type">${s.asset_type}</div></td>
    <td class="${(s.signal||'').includes('BUY')?'buy':'sell'}">${s.signal}</td>
    <td>${s.price}</td>
    <td>${sc(s.profitability)}</td><td>${sc(s.safety)}</td><td>${sc(s.risk)}</td><td>${sc(s.confidence)}</td>
    <td style="font-size:11px">${s.rr_ratio||'—'}</td>
    <td>${sc(s.trend_strength)}</td>
    <td>${breakoutBadge(s.breakout)}</td>
    <td style="font-size:11px">${s.entry}</td>
    <td style="font-size:11px">${s.stop_loss}</td>
    <td style="font-size:11px">${s.take_profit}</td>
    <td style="font-size:11px;color:${s.outcome==='WIN'?'var(--green)':s.outcome==='LOSS'?'var(--red)':'var(--muted2)'}">${s.outcome||'—'}</td>
    <td style="font-size:11px">${s.pnl||'—'}</td>
    <td><div class="reason-text">${s.reason}</div></td>
  </tr>`).join('');
}

function renderPerformance(stats){
  document.getElementById('perfWR').textContent     = stats.win_rate+'%';
  document.getElementById('perfWins').textContent   = stats.wins;
  document.getElementById('perfLosses').textContent = stats.losses;
  document.getElementById('perfAvgPnl').textContent = stats.avg_pnl;
  document.getElementById('perfTotal').textContent  = stats.total;
  document.getElementById('statWinRate').textContent = stats.win_rate+'%';
  document.getElementById('wrFill').style.width = stats.win_rate+'%';
  const tbody=document.getElementById('topAssetsBody');
  if(stats.top_assets&&stats.top_assets.length){
    tbody.innerHTML=stats.top_assets.map(a=>`<tr>
      <td style="font-weight:600">${a.symbol}</td>
      <td style="color:var(--muted2)">${a.count} signal${a.count!==1?'s':''}</td>
    </tr>`).join('');
  }
}

function renderJournal(){
  const search = document.getElementById('journalSearch').value.toLowerCase();
  let filtered = journalData.filter(t=>
    !search || (t.symbol||'').toLowerCase().includes(search)
  );
  const tbody=document.getElementById('journalBody');
  if(!filtered.length){
    tbody.innerHTML=`<tr><td colspan="11" class="no-signals"><div class="no-signals-icon">📒</div>No paper trades yet</td></tr>`;
    return;
  }
  tbody.innerHTML=filtered.map(t=>`<tr>
    <td style="font-size:11px;color:var(--muted2)">${t.date}</td>
    <td style="font-size:11px;color:var(--muted2)">${t.time}</td>
    <td><div class="symbol">${t.symbol}</div></td>
    <td class="${t.direction==='BUY'?'buy':'sell'}">${t.direction}</td>
    <td>${t.qty}</td>
    <td>${t.entry}</td><td>${t.stop_loss}</td><td>${t.take_profit}</td>
    <td style="color:${t.status==='OPEN'?'var(--yellow)':t.status==='WIN'?'var(--green)':'var(--red)'}">${t.status}</td>
    <td>${t.pnl||'—'}</td>
    <td><div class="reason-text">${t.notes}</div></td>
  </tr>`).join('');
}

// ── HEALTH ICONS per function ──────────────────────────────────────
const HEALTH_ICONS = {
  scanner:            '🔍',
  crypto_fetch:       '🪙',
  stock_fetch:        '📈',
  forex_fetch:        '💱',
  indicators:         '📊',
  ai_analysis:        '🤖',
  ai_cache:           '⚡',
  mtf_confirm:        '🕐',
  signal_check:       '🎯',
  telegram_send:      '📤',
  telegram_commands:  '📱',
  paper_trade:        '📝',
  csv_write:          '💾',
  csv_read:           '📂',
  unit_tests:         '🧪',
  web_server:         '🌐',
  alpaca_connection:  '🦙',
  exchange_connection:'🔗',
};

const HEALTH_LABELS = {
  scanner:            'Market Scanner',
  crypto_fetch:       'Crypto Data Fetch',
  stock_fetch:        'Stock Data Fetch',
  forex_fetch:        'Forex Data Fetch',
  indicators:         'Technical Indicators',
  ai_analysis:        'AI Analysis (Groq)',
  ai_cache:           'AI Response Cache',
  mtf_confirm:        'Multi-Timeframe Check',
  signal_check:       'Signal Filter',
  telegram_send:      'Telegram Send',
  telegram_commands:  'Telegram Commands',
  paper_trade:        'Paper Trading',
  csv_write:          'CSV Write',
  csv_read:           'CSV Read',
  unit_tests:         'Unit Tests',
  web_server:         'Web Server',
  alpaca_connection:  'Alpaca Connection',
  exchange_connection:'Exchange Connection',
};

function renderHealth(data){
  healthData = data;
  const now = new Date().toLocaleTimeString();
  document.getElementById('healthLastFetch').textContent = now;

  const counts = {ok:0, warn:0, error:0, idle:0};
  Object.values(data).forEach(v => { counts[v.status] = (counts[v.status]||0)+1; });

  // Banner
  const total = Object.keys(data).length;
  const banner = document.getElementById('healthBanner');
  const bannerIcon = document.getElementById('healthBannerIcon');
  const bannerTitle = document.getElementById('healthBannerTitle');
  const bannerSub = document.getElementById('healthBannerSub');
  if(counts.error > 0){
    banner.style.borderColor = 'var(--red)';
    bannerIcon.textContent = '✕';
    bannerIcon.style.color = 'var(--red)';
    bannerTitle.textContent = `${counts.error} function${counts.error>1?'s':''} in error state`;
    bannerSub.textContent   = `${counts.ok} OK, ${counts.warn} warnings, ${counts.idle} idle of ${total} functions`;
  } else if(counts.warn > 0){
    banner.style.borderColor = 'var(--yellow)';
    bannerIcon.textContent = '⚠';
    bannerIcon.style.color = 'var(--yellow)';
    bannerTitle.textContent = `${counts.warn} warning${counts.warn>1?'s':''}`;
    bannerSub.textContent   = `${counts.ok} OK, ${counts.idle} idle — system running`;
  } else if(counts.ok > 0){
    banner.style.borderColor = 'var(--green)';
    bannerIcon.textContent = '✓';
    bannerIcon.style.color = 'var(--green)';
    bannerTitle.textContent = 'All systems operating normally';
    bannerSub.textContent   = `${counts.ok} functions OK, ${counts.idle} idle`;
  } else {
    banner.style.borderColor = 'var(--muted)';
    bannerIcon.textContent = '○';
    bannerIcon.style.color = 'var(--muted)';
    bannerTitle.textContent = 'System starting up';
    bannerSub.textContent   = 'Functions will activate on first scan';
  }

  // Summary pills
  const pills = document.getElementById('healthPills');
  pills.innerHTML = [
    counts.ok    ? `<div class="h-pill h-pill-ok"><span class="status-dot-sm ok"></span>${counts.ok} OK</div>` : '',
    counts.warn  ? `<div class="h-pill h-pill-warn"><span class="status-dot-sm warn"></span>${counts.warn} Warning</div>` : '',
    counts.error ? `<div class="h-pill h-pill-error"><span class="status-dot-sm error"></span>${counts.error} Error</div>` : '',
    counts.idle  ? `<div class="h-pill h-pill-idle"><span class="status-dot-sm idle"></span>${counts.idle} Idle</div>` : '',
  ].join('');

  // Cards — errors first, then warns, then ok, then idle
  const order = ['error','warn','ok','idle'];
  const entries = Object.entries(data).sort((a,b)=>{
    return order.indexOf(a[1].status) - order.indexOf(b[1].status);
  });

  const grid = document.getElementById('healthGrid');
  grid.innerHTML = entries.map(([key, v]) => {
    const icon  = HEALTH_ICONS[key]  || '⬤';
    const label = HEALTH_LABELS[key] || key;
    return `
    <div class="health-card status-${v.status}">
      <div class="health-icon">${icon}</div>
      <div class="health-body">
        <div class="health-name">
          <span class="status-dot-sm ${v.status}"></span>
          ${label}
        </div>
        <div class="health-msg">${v.message || '—'}</div>
        <div class="health-meta">
          <span class="health-time">Last: ${v.last_run || 'Never'}</span>
          ${v.count > 0 ? `<span class="health-count">${v.count} calls</span>` : ''}
        </div>
      </div>
    </div>`;
  }).join('');
}

function loadHealth(){
  fetch('/health_full').then(r=>r.json()).then(data=>{
    renderHealth(data);
  }).catch(err=>{
    document.getElementById('healthGrid').innerHTML =
      `<div style="color:var(--red);font-size:12px;padding:20px">Failed to load health data: ${err}</div>`;
  });
}

function loadSignals(){
  fetch('/signals').then(r=>r.json()).then(data=>{
    signals=data.signals||[];
    document.getElementById('statCurrent').textContent=signals.length;
    document.getElementById('statTotal').textContent=data.total||0;
    document.getElementById('statAssets').textContent=data.asset_count||'150';
    document.getElementById('statPnl').textContent=data.daily_pnl!=null?(data.daily_pnl>=0?'+':'')+data.daily_pnl:'—';
    document.getElementById('lastUpdate').textContent='Updated '+new Date().toLocaleTimeString();
    renderSignals();
  }).catch(()=>{});
}

function loadHistory(){
  fetch('/history').then(r=>r.json()).then(data=>{historyData=data; renderHistory();}).catch(()=>{});
}

function loadPerformance(){
  fetch('/stats').then(r=>r.json()).then(data=>{renderPerformance(data);}).catch(()=>{});
}

function loadJournal(){
  fetch('/journal').then(r=>r.json()).then(data=>{journalData=data; renderJournal();}).catch(()=>{});
}

loadSignals();
setInterval(loadSignals, 30000);
setInterval(()=>{
  if(document.getElementById('tab-health').classList.contains('active')){
    loadHealth();
  }
}, 15000);
</script>
</body>
</html>"""

# ════════════════════════════════════════════════════════════════════════════
# WEB SERVER
# ════════════════════════════════════════════════════════════════════════════
class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        routes = {
            '/':            ('text/html',        lambda: DASHBOARD_HTML.encode()),
            '/signals':     ('application/json', self._signals),
            '/history':     ('application/json', self._history),
            '/stats':       ('application/json', self._stats),
            '/journal':     ('application/json', self._journal),
            '/health':      ('application/json', self._health_simple),
            '/health_full': ('application/json', self._health_full),
        }
        if self.path in routes:
            ct, fn = routes[self.path]
            body = fn()
            self.send_response(200)
            self.send_header('Content-type', ct)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes) else body.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _signals(self):
        total = len(crypto_pairs) + (len(stock_pairs) if ALPACA_AVAILABLE else 0) + (len(forex_pairs) if ALPACA_AVAILABLE else 0)
        return json.dumps({
            'signals':     all_signals,
            'total':       total_signals_found,
            'asset_count': total,
            'daily_pnl':   round(daily_pnl, 4),
            'last_scan':   last_scan_time,
        })

    def _history(self):
        return json.dumps(load_history())

    def _stats(self):
        return json.dumps(compute_stats(load_history()))

    def _journal(self):
        return json.dumps(load_trade_journal())

    def _health_simple(self):
        with health_lock:
            return json.dumps({'status': 'ok', 'last_scan': last_scan_time, 'signals': len(all_signals)})

    def _health_full(self):
        with health_lock:
            return json.dumps(dict(system_health))

    def log_message(self, format, *args):
        pass  # suppress default HTTP request logs

def start_dashboard():
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    log.info("[SERVER] Dashboard running on port %d", port)
    update_health("web_server", "ok", f"HTTP server listening on port {port}", increment=False)
    server.serve_forever()

# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    log.info("[BOOT] Trade Grid Analysis starting — 50 crypto + 50 stocks + 50 forex = 150 assets")

    run_unit_tests()
    init_csv()

    threading.Thread(target=start_dashboard, daemon=True).start()
    threading.Thread(target=handle_telegram_commands, daemon=True).start()
    log.info("[BOOT] Telegram listener thread started")

    run_scanner()

    schedule.clear()
    schedule.every(150).seconds.do(run_scanner)  # 150s = 2m30s

    log.info("[BOOT] Scanning every 2m 30s. Press Ctrl+C to stop.")

    while True:
        try:
            schedule.run_pending()
            time.sleep(15)
        except KeyboardInterrupt:
            log.info("[BOOT] Shutting down Trade Grid Analysis")
            break
        except Exception as e:
            log.error("[BOOT] Scheduler error: %s\n%s", e, traceback.format_exc())
            time.sleep(30)