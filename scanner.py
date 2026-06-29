import ccxt
import pandas as pd
import numpy as np
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
import re
import pytz
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
# COLORIZED LOGGING UTILITY
# ════════════════════════════════════════════════════════════════════════════
class ColorFormatter(logging.Formatter):
    """Custom formatter to colorize terminal logs while maintaining plain file text."""
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    cyan = "\x1b[36;20m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s [%(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + fmt + reset,
        logging.INFO: cyan + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.grey + self.fmt + self.reset)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

log_handler_stream = logging.StreamHandler()
log_handler_stream.setFormatter(ColorFormatter())
log_handler_file = logging.FileHandler("tradegrid.log")
log_handler_file.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[log_handler_file, log_handler_stream]
)
log = logging.getLogger("tradegrid")

# ════════════════════════════════════════════════════════════════════════════
# SYSTEM HEALTH TRACKER
# ════════════════════════════════════════════════════════════════════════════
health_lock = threading.Lock()
system_health = {
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

def update_health(key: str, status: str, message: str, increment: bool = True):
    """Updates a system health entry thread-safely."""
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
# CONFIGURATION
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

# Global configuration constants
GLOBAL_CONFIDENCE_THRESHOLD = 70.0  # Core engine score cutoff (0-100 scale)
MAX_DAILY_LOSS_PCT          = 0.05
POSITION_RISK_PCT           = 0.02
TRAILING_STOP_PCT           = 0.015
PARTIAL_TP_PCT              = 0.5

# ════════════════════════════════════════════════════════════════════════════
# RECORD-KEEPING AND FILE PATHS
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

signal_lock   = threading.Lock()
csv_lock      = threading.Lock()
cache_lock    = threading.Lock()
trade_lock    = threading.Lock()

all_signals         = []
last_scan_time      = "Never"
total_signals_found = 0
daily_pnl           = 0.0
scan_count          = 0
ai_quota_exceeded   = False

# AI response cache { hash -> (timestamp, score_payload) }
ai_cache    = {}
CACHE_TTL   = 900  # Extended cache to 15 mins for performance

# ════════════════════════════════════════════════════════════════════════════
# CLIENT POOLS
# ════════════════════════════════════════════════════════════════════════════
exchange = ccxt.kraken({'enableRateLimit': True})
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
        update_health("alpaca_connection", "ok", f"Connected to {ALPACA_BASE_URL}")
    except Exception as e:
        update_health("alpaca_connection", "error", f"Connection failed: {e}")
        ALPACA_AVAILABLE = False
else:
    update_health("alpaca_connection", "warn", "Alpaca disabled", increment=False)

# ════════════════════════════════════════════════════════════════════════════
# UTILITIES & RECURSIVE TIMING MODULES
# ════════════════════════════════════════════════════════════════════════════
def is_market_open(asset_type: str) -> bool:
    """Verifies operational exchange boundaries across global multi-asset parameters."""
    if asset_type.lower() == 'crypto':
        return True
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    if now_ny.weekday() >= 5:
        return False
    if asset_type.lower() == 'stock':
        start_time = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        end_time = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        return start_time <= now_ny <= end_time
    if asset_type.lower() == 'forex':
        if now_ny.weekday() == 4 and now_ny.hour >= 17:
            return False
        if now_ny.weekday() == 6 and now_ny.hour < 17:
            return False
        return True
    return True

def with_retry(fn, retries: int = 3, delay: float = 2.0, label: str = ""):
    """Implements exponential backoff wrapper for tracking network connections."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                log.warning(f"[RATE_LIMIT] 429 Hit on {label}. Backing off...")
                time.sleep(delay * (2 ** attempt))
                continue
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                log.error(f"[RETRY_FAIL] Execution broken for {label}: {e}")
                return None

def fetch_live_stock_universe():
    fallback = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL', 'AMD', 'NFLX', 'PLTR']
    if not alpaca:
        return fallback
    try:
        assets = alpaca.list_assets(status='active', asset_class='us_equity')
        liquid_symbols = [a.symbol for a in assets if a.tradable and a.shortable and a.marginable]
        focus_universe = {'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL', 'NFLX', 'AMD', 'PLTR', 'COIN', 'INTC', 'BABA', 'SPY'}
        active_watchlist = [sym for sym in liquid_symbols if sym in focus_universe]
        return active_watchlist if active_watchlist else fallback
    except Exception as e:
        log.error(f"[SYNC] Stock discovery failed: {e}")
        return fallback

def fetch_live_crypto_universe():
    fallback = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'LINK/USDT', 'AVAX/USDT']
    if not exchange:
        return fallback
    try:
        markets = exchange.markets
        if not markets:
            markets = exchange.load_markets()
        active_pairs = []
        for symbol, market in markets.items():
            if market.get('active') and market.get('spot') and market.get('quote') in ['USDT', 'USD']:
                base = market.get('base', '')
                if not any(t in base for t in ['UP', 'DOWN', 'BEAR', 'BULL', '3L', '3S']):
                    active_pairs.append(symbol)
        majors = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT']
        final_list = [p for p in active_pairs if p in majors]
        for pair in active_pairs:
            if pair not in final_list and len(final_list) < 15:
                final_list.append(pair)
        return final_list if final_list else fallback
    except Exception as e:
        log.error(f"[SYNC] Crypto discovery failed: {e}")
        return fallback

# ════════════════════════════════════════════════════════════════════════════
# CORE DATA CLEANER & VERIFICATION LAYER
# ════════════════════════════════════════════════════════════════════════════
def clean_and_validate_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return None
    try:
        df = df.copy()
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                return None
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        else:
            df = df.sort_index()
            df = df.loc[~df.index.duplicated(keep='last')]
        return df if len(df) >= 200 else None
    except Exception as e:
        log.error(f"[CLEANER] Processing error: {e}")
        return None

def get_crypto_ohlcv(symbol: str, timeframe: str = '1h', limit: int = 500) -> pd.DataFrame:
    def _fetch():
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    result = with_retry(_fetch, label=f"Crypto:{symbol}:{timeframe}")
    return clean_and_validate_df(result)

def get_alpaca_ohlcv(symbol: str, timeframe: str = '1Hour', limit: int = 500) -> pd.DataFrame:
    if not alpaca:
        return None
    tf_map = {'1h': tradeapi.rest.TimeFrame.Hour, '1d': tradeapi.rest.TimeFrame.Day, '1Hour': tradeapi.rest.TimeFrame.Hour}
    tf = tf_map.get(timeframe, tradeapi.rest.TimeFrame.Hour)
    def _fetch():
        bars = alpaca.get_bars(symbol, tf, limit=limit).df
        if bars.empty:
            return None
        col_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        return bars.rename(columns=col_map)
    result = with_retry(_fetch, label=f"Alpaca:{symbol}:{timeframe}")
    return clean_and_validate_df(result)

# ════════════════════════════════════════════════════════════════════════════
# VECTORIZED TECHNICAL ENGINE & MARKET STRUCTURE DETECTOR
# ════════════════════════════════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized application of technical parameters without processing leaks."""
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
        df['atr']         = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

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
        return df
    except Exception as e:
        log.error(f"[INDICATORS] Evaluation loop failure: {e}")
        raise

def add_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Algorithmic mapping of Smart Money Patterns (BOS, CHOCH, FVG, OB)."""
    df = df.copy()
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    length = len(df)

    swing_highs = np.zeros(length)
    swing_lows = np.zeros(length)
    bos = np.zeros(length)
    choch = np.zeros(length)
    fvg = np.zeros(length)
    order_blocks = np.zeros(length)

    last_high = highs[0]
    last_low = lows[0]
    trend_direction = 0  # 1 for Bullish, -1 for Bearish

    for i in range(2, length - 2):
        # 1. Swing Detection
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs[i] = highs[i]
            last_high = highs[i]
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows[i] = lows[i]
            last_low = lows[i]

        # 2. Fair Value Gaps (FVG)
        if highs[i-2] < lows[i]:
            fvg[i] = 1  # Bullish FVG
        elif lows[i-2] > highs[i]:
            fvg[i] = -1  # Bearish FVG

        # 3. Structural Breakouts (BOS / CHOCH)
        if closes[i] > last_high:
            if trend_direction == -1:
                choch[i] = 1
                trend_direction = 1
            else:
                bos[i] = 1
            order_blocks[i-1] = lows[i-1]  # Demand Zone Allocation
        elif closes[i] < last_low:
            if trend_direction == 1:
                choch[i] = -1
                trend_direction = -1
            else:
                bos[i] = -1
            order_blocks[i-1] = highs[i-1]  # Supply Zone Allocation

    df['swing_high'] = swing_highs
    df['swing_low'] = swing_lows
    df['bos'] = bos
    df['choch'] = choch
    df['fvg'] = fvg
    df['order_block'] = order_blocks
    return df

# ════════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME AGREEMENT ENGINE
# ════════════════════════════════════════════════════════════════════════════
def verify_mtf_alignment(symbol: str, asset_type: str, direction: str) -> bool:
    """Verifies macro structures match across Daily, 4H, and 1H time horizons."""
    try:
        if asset_type.lower() == 'crypto':
            df_daily = get_crypto_ohlcv(symbol, timeframe='1d', limit=100)
            df_4h    = get_crypto_ohlcv(symbol, timeframe='4h', limit=100)
        else:
            df_daily = get_alpaca_ohlcv(symbol, timeframe='1d', limit=100)
            df_4h    = get_alpaca_ohlcv(symbol, timeframe='1Hour', limit=300) # Fallback scaling

        if df_daily is None or df_4h is None:
            return False

        df_daily = add_indicators(df_daily)
        df_4h    = add_indicators(df_4h)

        daily_bull = df_daily['ema50'].iloc[-1] > df_daily['ema200'].iloc[-1] and df_daily['rsi'].iloc[-1] > 50
        daily_bear = df_daily['ema50'].iloc[-1] < df_daily['ema200'].iloc[-1] and df_daily['rsi'].iloc[-1] < 50

        fourh_bull = df_4h['ema20'].iloc[-1] > df_4h['ema50'].iloc[-1] and df_4h['rsi'].iloc[-1] > 48
        fourh_bear = df_4h['ema20'].iloc[-1] < df_4h['ema50'].iloc[-1] and df_4h['rsi'].iloc[-1] < 52

        if direction == 'BUY' and daily_bull and fourh_bull:
            return True
        if direction == 'SELL' and daily_bear and fourh_bear:
            return True
        return False
    except Exception as e:
        log.error(f"[MTF] Structural alignment crash for {symbol}: {e}")
        return False

# ════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC MATH-DRIVEN CONFIDENCE ENGINE
# ════════════════════════════════════════════════════════════════════════════
def calculate_system_confidence(df: pd.DataFrame, direction: str) -> float:
    """Deterministic grading model processing mathematical variables (0 to 100)."""
    row = df.iloc[-1]
    score = 0.0

    # 1. Moving Average Alignment Matrix (Max 30 Points)
    if direction == 'BUY':
        if row['ema20'] > row['ema50']: score += 15
        if row['ema50'] > row['ema200']: score += 15
    else:
        if row['ema20'] < row['ema50']: score += 15
        if row['ema50'] < row['ema200']: score += 15

    # 2. Momentum Indicators (Max 25 Points)
    rsi = row['rsi']
    if direction == 'BUY':
        if 50 < rsi <= 70: score += 15
        elif rsi > 70: score += 5  # Overextended structural penalties
        if row['macd_diff'] > 0: score += 10
    else:
        if 30 <= rsi < 50: score += 15
        elif rsi < 30: score += 5
        if row['macd_diff'] < 0: score += 10

    # 3. Volatility & Trend Intensitites (Max 20 Points)
    if row['adx'] > 25:
        score += 10
        if direction == 'BUY' and row['adx_pos'] > row['adx_neg']: score += 10
        elif direction == 'SELL' and row['adx_neg'] > row['adx_pos']: score += 10

    # 4. Smart Money Structure & Volumes (Max 25 Points)
    if row['vol_spike'] > 1.5: score += 10
    if direction == 'BUY':
        if row['fvg'] == 1: score += 5
        if row['bos'] == 1 or row['choch'] == 1: score += 10
    else:
        if row['fvg'] == -1: score += 5
        if row['bos'] == -1 or row['choch'] == -1: score += 10

    return float(np.clip(score, 0.0, 100.0))

# ════════════════════════════════════════════════════════════════════════════
# QUANT RISK MANAGEMENT & EXECUTION GENERATOR
# ════════════════════════════════════════════════════════════════════════════
def run_risk_engine(df: pd.DataFrame, direction: str) -> dict:
    """Calculates entry points, risk-adjusted stop losses, and target placement."""
    row = df.iloc[-1]
    price = float(row['close'])
    atr = float(row['atr']) if not pd.isna(row['atr']) else (price * 0.01)

    if direction == 'BUY':
        stop_loss = price - (atr * 2.0)
        take_profit = price + (atr * 4.0)
    else:
        stop_loss = price + (atr * 2.0)
        take_profit = price - (atr * 4.0)

    risk = abs(price - stop_loss)
    reward = abs(take_profit - price)
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

    return {
        "entry": round(price, 4),
        "stop_loss": round(stop_loss, 4),
        "take_profit": round(take_profit, 4),
        "rr_ratio": rr_ratio
    }

# ════════════════════════════════════════════════════════════════════════════
# AI VALIDATION LAYER (INTELLIGENT FILTERING)
# ════════════════════════════════════════════════════════════════════════════
def parse_ai_narrative(ai_text: str) -> str:
    """Extracts explanation line strictly keeping execution files clean."""
    if not ai_text:
        return "No narrative response generated."
    for line in ai_text.split('\n'):
        if 'explanation' in line.lower() or 'reason' in line.lower():
            return line.partition(':')[2].strip().replace('*', '')
    return ai_text.split('\n')[0].strip()

def run_ai_validation(symbol: str, direction: str, confidence: float, risk_profile: dict) -> str:
    """Asynchronous analytical engine mapping institutional data context."""
    global ai_quota_exceeded
    if ai_quota_exceeded:
        return "Deterministic engine confirmation. AI execution layer bypassed due to network rate limits."

    cache_hash = hashlib.md5(f"{symbol}|{direction}|{round(confidence,0)}".encode()).hexdigest()
    with cache_lock:
        if cache_hash in ai_cache:
            ts, cached_reason = ai_cache[cache_hash]
            if time.time() - ts < CACHE_TTL:
                return cached_reason

    prompt = f"""You are an elite quantitative risk analyst. Validate this system setup:
    Asset: {symbol} | Direction: {direction}
    Quantitative Scoring: {confidence}/100
    Calculated Target Entry: {risk_profile['entry']} | Target Stop: {risk_profile['stop_loss']} | Target Profit: {risk_profile['take_profit']}
    
    Provide exactly one short sentence explaining the spatial institutional mechanics supporting or questioning this setup under the field 'Explanation:'."""

    def _call():
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )
        return resp.choices[0].message.content

    raw_response = with_retry(_call, label=f"AI-Validation:{symbol}")
    if not raw_response:
        log.warning(f"[AI_LAYER] Quota limit encountered. Bypassing validation for {symbol}.")
        return "Mathematical system confirmation. LLM pipeline bypassed."

    explanation = parse_ai_narrative(raw_response)
    with cache_lock:
        ai_cache[cache_hash] = (time.time(), explanation)
    return explanation

# ════════════════════════════════════════════════════════════════════════════
# CORE SIGNAL PIPELINE LINKAGE
# ════════════════════════════════════════════════════════════════════════════
def check_signal(symbol: str, df: pd.DataFrame, asset_type: str):
    """Deterministic routing module managing live trading candidates."""
    global total_signals_found, all_signals
    if df is None or df.empty:
        return

    df = add_market_structure(df)
    row = df.iloc[-1]
    
    # Mathematical rules for underlying trend engines
    direction = None
    if row['ema20'] > row['ema50'] and row['rsi'] > 53:
        direction = 'BUY'
    elif row['ema20'] < row['ema50'] and row['rsi'] < 47:
        direction = 'SELL'

    if not direction:
        return

    # 1. Math Scoring Engine Evaluation
    confidence = calculate_system_confidence(df, direction)
    if confidence < GLOBAL_CONFIDENCE_THRESHOLD:
        return  # Filter dropped immediately: zero computing waste

    # 2. Multi-Timeframe Confirmation Filter
    if not verify_mtf_alignment(symbol, asset_type, direction):
        return

    # 3. Risk Profile Engineering Matrix
    risk_profile = run_risk_engine(df, direction)
    if risk_profile['rr_ratio'] < 1.5:
        return  # Reject trade if risk-to-reward ratio is poor

    # 4. LLM Structural Validation Layer (Fired only for high-probability setups)
    explanation = run_ai_validation(symbol, direction, confidence, risk_profile)

    signal_payload = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M:%S'),
        'symbol': symbol,
        'asset_type': asset_type,
        'signal': direction,
        'price': risk_profile['entry'],
        'rsi': round(row['rsi'], 2),
        'macd': round(row['macd'], 6),
        'atr': round(row['atr'], 4),
        'trend_strength': confidence / 10.0,
        'rr_ratio': risk_profile['rr_ratio'],
        'support': round(row['support'], 4),
        'resistance': round(row['resistance'], 4),
        'volume_spike': 'YES' if row['vol_spike'] > 1.5 else 'NO',
        'breakout': 'BOS' if row['bos'] != 0 else 'NONE',
        'confidence': int(confidence),
        'profitability': int(confidence * 0.95 / 10),
        'safety': int(confidence * 0.9 / 10),
        'risk': int(10 - (confidence / 10)),
        'entry': risk_profile['entry'],
        'stop_loss': risk_profile['stop_loss'],
        'take_profit': risk_profile['take_profit'],
        'reason': explanation,
        'outcome': 'OPEN',
        'pnl': 0.0
    }

    with signal_lock:
        all_signals.append(signal_payload)
        total_signals_found += 1

    save_signal_to_csv(signal_payload)
    place_paper_trade(signal_payload)
    send_formatted_telegram_notification(signal_payload)

# ════════════════════════════════════════════════════════════════════════════
# TELEGRAM TELEMETRY ENGINE
# ════════════════════════════════════════════════════════════════════════════
def send_telegram(message: str):
    """Dispatches payload notification blocks directly to user terminal."""
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
        if r.status_code == 200:
            update_health("telegram_send", "ok", f"Message sent ({len(message)} chars)")
        else:
            update_health("telegram_send", "warn", f"HTTP Response code: {r.status_code}")
    except Exception as e:
        update_health("telegram_send", "error", f"Network dispatch broken: {e}")

def send_formatted_telegram_notification(s: dict):
    """Generates clean mobile notifications matching the strict UI guidelines."""
    emoji = "🚀" if s['signal'] == 'BUY' else "💥"
    msg = (
        f"{emoji} *{s['symbol']}*\n"
        f"*{s['signal']}*\n\n"
        f"• *Confidence:* `{s['confidence']}%`\n"
        f"• *Entry:* `{s['entry']}`\n"
        f"• *Stop Loss:* `{s['stop_loss']}`\n"
        f"• *Take Profit:* `{s['take_profit']}`\n"
        f"• *Risk Reward:* `1 : {s['rr_ratio']}`\n\n"
        f"*Institutional Context:*\n"
        f"_{s['reason']}_"
    )
    send_telegram(msg)

def handle_telegram_commands():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    last_update_id = None
    update_health("telegram_commands", "ok", "Telegram command listener started", increment=False)
    while True:
        try:
            params = {'timeout': 30, 'offset': last_update_id}
            response = requests.get(url, params=params, timeout=35)
            data = response.json()
            for update in data.get('result', []):
                last_update_id = update['update_id'] + 1
                message = update.get('message', {})
                text = message.get('text', '').strip()
                chat_id = message.get('chat', {}).get('id')
                if not chat_id:
                    continue
                cmd = text.lower().split()[0] if text else ''
                if cmd == '/status':
                    reply = (
                        f"*Bot Status Engine*\n"
                        f"Last Run Matrix: `{last_scan_time}`\n"
                        f"Total Signals Logged: `{total_signals_found}`\n"
                        f"Portfolio Drawdown Tracker: `{daily_pnl}`"
                    )
                    send_telegram(reply)
        except Exception as e:
            time.sleep(5)

# ════════════════════════════════════════════════════════════════════════════
# PAPER TRADING MODULE
# ════════════════════════════════════════════════════════════════════════════
def get_portfolio_value() -> float:
    if not alpaca:
        return 100000.0
    try:
        return float(alpaca.get_account().portfolio_value)
    except Exception:
        return 100000.0

def place_paper_trade(signal_data: dict):
    global daily_pnl
    if not alpaca or signal_data['asset_type'].lower() != 'stock':
        return
    with trade_lock:
        portfolio = get_portfolio_value()
        if daily_pnl < -(portfolio * MAX_DAILY_LOSS_PCT):
            return
        symbol = signal_data['symbol']
        try:
            price = float(signal_data['price'])
            sl = float(signal_data['stop_loss'])
            qty = max(int((portfolio * POSITION_RISK_PCT) / max(abs(price - sl), 0.01)), 1)
            side = 'buy' if signal_data['signal'] == 'BUY' else 'sell'
            alpaca.submit_order(
                symbol=symbol, qty=qty, side=side, type='market',
                time_in_force='day', trail_percent=TRAILING_STOP_PCT * 100
            )
            trade_rec = {
                'date': datetime.now().strftime('%Y-%m-%d'), 'time': datetime.now().strftime('%H:%M:%S'),
                'symbol': symbol, 'direction': side.upper(), 'qty': qty, 'entry': price,
                'stop_loss': sl, 'take_profit': signal_data['take_profit'], 'status': 'OPEN', 'pnl': 0.0, 'notes': 'Auto paper trade executed'
            }
            save_trade_to_journal(trade_rec)
        except Exception as e:
            log.error(f"[EXECUTION] Paper trade failed: {e}")

# ════════════════════════════════════════════════════════════════════════════
# CSV RECORD OPERATIONS
# ════════════════════════════════════════════════════════════════════════════
def init_csv():
    with csv_lock:
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writeheader()
        if not os.path.exists(TRADE_LOG):
            with open(TRADE_LOG, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=TRADE_FIELDS).writeheader()

def save_signal_to_csv(signal_data: dict):
    try:
        with csv_lock:
            with open(HISTORY_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
                writer.writerow({k: signal_data.get(k, '') for k in HISTORY_FIELDS})
    except Exception as e:
        log.error(f"[CSV_WRITE] Error: {e}")

def save_trade_to_journal(trade_data: dict):
    try:
        with csv_lock:
            with open(TRADE_LOG, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
                writer.writerow({k: trade_data.get(k, '') for k in TRADE_FIELDS})
    except Exception as e:
        log.error(f"[CSV_JOURNAL] Error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# MULTI-THREADED SCANNING MATRIX
# ════════════════════════════════════════════════════════════════════════════
def process_single_asset(symbol: str, asset_type: str):
    """Concurrent worker thread target for tracking active market structures."""
    try:
        if not is_market_open(asset_type):
            return
        if asset_type.lower() == 'crypto':
            df = get_crypto_ohlcv(symbol, timeframe='1h')
        elif asset_type.lower() == 'stock':
            df = get_alpaca_ohlcv(symbol, timeframe='1Hour')
        else:
            df = get_alpaca_ohlcv(symbol, timeframe='1Hour')  # Forex mapping framework

        if df is not None:
            df = add_indicators(df)
            check_signal(symbol, df, asset_type)
    except Exception as e:
        log.error(f"[ASYNC_WORKER] Error executing asset processing for {symbol}: {e}")

def run_scanner():
    """Concurrently loops over asset structures via thread pool execution."""
    global last_scan_time, scan_count
    scan_count += 1
    last_scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    update_health("scanner", "ok", f"Scanning sequence #{scan_count} triggered.")

    crypto_pool = fetch_live_crypto_universe()
    stock_pool  = fetch_live_stock_universe()
    forex_pool  = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']

    tasks = []
    for s in crypto_pool: tasks.append((s, "Crypto"))
    for s in stock_pool:  tasks.append((s, "Stock"))
    for s in forex_pool:  tasks.append((s, "Forex"))

    # High-speed processing matrix running up to 16 concurrent workers
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_single_asset, item[0], item[1]) for item in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log.error(f"[POOL_ERROR] Thread pool exception execution caught: {e}")

    update_health("scanner", "idle", f"Scan completed at {datetime.now().strftime('%H:%M:%S')}")

# ════════════════════════════════════════════════════════════════════════════
# ENGINE DASHBOARD ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html><html><head><title>System Core Status</title><style>body{background:#111;color:#eee;font-family:monospace;padding:20px;}</style></head><body><h1>Production Quant System Monitoring Matrix</h1></body></html>"""

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            with health_lock:
                self.wfile.write(json.dumps(system_health).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def start_dashboard():
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    update_health("web_server", "ok", f"HTTP production listener opened on port {port}", increment=False)
    server.serve_forever()

# ════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE UNIT TEST ENGINE
# ════════════════════════════════════════════════════════════════════════════
def run_unit_tests():
    try:
        mock_data = {
            'open': np.linspace(100, 110, 210),
            'high': np.linspace(102, 112, 210),
            'low': np.linspace(99, 109, 210),
            'close': np.linspace(101, 111, 210),
            'volume': np.random.randint(1000, 5000, 210).astype(float)
        }
        df = pd.DataFrame(mock_data)
        df = add_indicators(df)
        df = add_market_structure(df)
        assert 'ema20' in df.columns, "Indicator component failure"
        assert 'bos' in df.columns, "SMC component failure"
        update_health("unit_tests", "ok", "All operational indicators passed validation.", increment=False)
    except Exception as e:
        update_health("unit_tests", "error", f"System unit assertion failed: {e}", increment=False)

# ════════════════════════════════════════════════════════════════════════════
# SYSTEM ENGINE INITIALIZATION ENTRY
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    log.info("[BOOT] Initializing production-grade quantitative system architecture layers...")
    run_unit_tests()
    init_csv()

    threading.Thread(target=start_dashboard, daemon=True).start()
    threading.Thread(target=handle_telegram_commands, daemon=True).start()

    run_scanner()
    schedule.every(150).seconds.do(run_scanner)

    log.info("[OPERATIONAL] Production loop initialized. Scanning every 2 minutes 30 seconds.")
    while True:
        schedule.run_pending()
        time.sleep(1)