"""
Trade Grid Analysis — scanner.py
Full upgrade: Signal Quality, AI Improvements, Dashboard, Telegram, Paper Trading, Reliability
"""

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
    print("⚠️  alpaca-trade-api not installed — stock/forex scanning disabled")

# ════════════════════════════════════════════════════════════════════════════
# LOGGING
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
    log.warning("ALPACA_API_KEY/SECRET not set — stock/forex disabled")
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
    'outcome', 'pnl'                          # filled later by paper trade tracking
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
CACHE_TTL   = 300   # seconds — reuse AI response for same setup within 5 min

# ════════════════════════════════════════════════════════════════════════════
# API CLIENTS
# ════════════════════════════════════════════════════════════════════════════
exchange = ccxt.kraken()
groq_client = Groq(api_key=GROQ_API_KEY)

alpaca = None
if ALPACA_AVAILABLE:
    try:
        alpaca = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET, ALPACA_BASE_URL)
        log.info("✅ Alpaca connected")
    except Exception as e:
        log.warning(f"Alpaca connection failed: {e}")
        ALPACA_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════
# ASSET LISTS
# ════════════════════════════════════════════════════════════════════════════
crypto_pairs = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'DOGE/USD',
    'BNB/USD', 'ADA/USD', 'AVAX/USD', 'LINK/USD', 'INJ/USD',
    'FET/USD', 'ARB/USD', 'OP/USD', 'DOT/USD', 'ATOM/USD',
    'LTC/USD', 'UNI/USD', 'NEAR/USD', 'AAVE/USD', 'APT/USD',
    'SUI/USD', 'ZEC/USD', 'BCH/USD', 'ETC/USD', 'XLM/USD',
    'FIL/USD', 'HBAR/USD', 'SEI/USD', 'JUP/USD', 'WLD/USD',
]

stock_pairs = [
    'AAPL', 'TSLA', 'NVDA', 'AMZN', 'META',
    'GOOGL', 'MSFT', 'AMD', 'NFLX', 'COIN',
    'SPY', 'QQQ', 'DIA', 'GLD', 'SLV',
    'USO', 'TLT', 'SOFI', 'RIOT', 'MARA',
    'MSTR', 'ARKK', 'XLK', 'XLF', 'XLE',
    'VTI', 'VOO', 'UPRO', 'TQQQ', 'BRK.B',
]

forex_pairs = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD',
    'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY',
    'AUDJPY', 'CADJPY', 'EURAUD', 'GBPAUD', 'NZDCAD',
    'NZDCHF', 'GBPCAD', 'CADCHF', 'AUDCAD', 'AUDCHF',
]

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

def save_signal_to_csv(signal_data):
    with csv_lock:
        with open(HISTORY_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            writer.writerow({k: signal_data.get(k, '') for k in HISTORY_FIELDS})

def save_trade_to_journal(trade_data):
    with csv_lock:
        with open(TRADE_LOG, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
            writer.writerow({k: trade_data.get(k, '') for k in TRADE_FIELDS})

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with csv_lock:
        with open(HISTORY_FILE, 'r') as f:
            return list(csv.DictReader(f))

def load_trade_journal():
    if not os.path.exists(TRADE_LOG):
        return []
    with csv_lock:
        with open(TRADE_LOG, 'r') as f:
            return list(csv.DictReader(f))

# ════════════════════════════════════════════════════════════════════════════
# WIN RATE / STATS HELPERS
# ════════════════════════════════════════════════════════════════════════════
def compute_stats(history):
    """Return dict of win_rate, total, wins, losses, avg_pnl, top_assets."""
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

    # Top assets by count
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
                log.warning(f"Retry {attempt+1}/{retries} for {label}: {e}")
                time.sleep(delay * (attempt + 1))
            else:
                log.error(f"All retries failed for {label}: {e}")
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
    return with_retry(_fetch, label=symbol)

def get_crypto_ohlcv_multi(symbol, timeframes=('1h', '4h')):
    """Fetch multiple timeframes for confirmation."""
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
    return with_retry(_fetch, label=symbol)

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
    return with_retry(_fetch, label=symbol)

# ════════════════════════════════════════════════════════════════════════════
# INDICATORS (including new: ATR, Support/Resistance, Trend Strength)
# ════════════════════════════════════════════════════════════════════════════
def add_indicators(df):
    df = df.copy()

    # Core
    df['ema20']  = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50']  = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['rsi']    = ta.momentum.rsi(df['close'], window=14)
    df['vol_avg'] = df['volume'].rolling(window=20).mean()

    # MACD
    macd_obj        = ta.trend.MACD(df['close'])
    df['macd']      = macd_obj.macd()
    df['macd_signal'] = macd_obj.macd_signal()
    df['macd_diff'] = macd_obj.macd_diff()

    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    # Bollinger Bands (for breakout)
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mid']   = bb.bollinger_mavg()

    # ADX — trend strength
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx']    = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()

    # Volume spike — current vol vs 20-bar average
    df['vol_spike'] = df['volume'] / df['vol_avg']

    # Support / Resistance — rolling 20-bar min/max
    df['support']    = df['low'].rolling(20).min()
    df['resistance'] = df['high'].rolling(20).max()

    return df

def calc_trend_strength(adx, ema20, ema50, ema200, rsi, macd_diff):
    """Return a 0–10 trend strength score."""
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
    """Return risk:reward ratio."""
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
    """Detect if price is breaking out of Bollinger or S/R levels."""
    if price > bb_upper and price > resistance:
        return "BULLISH_BREAKOUT"
    if price < bb_lower and price < support:
        return "BEARISH_BREAKOUT"
    return "NONE"

def multi_timeframe_confirm(symbol, primary_signal, asset_type):
    """
    Fetch 4h data and check if higher-timeframe trend agrees with signal.
    Returns True if confirmed, False if contradicted, None if unavailable.
    """
    try:
        if asset_type == 'crypto':
            df4h = get_crypto_ohlcv(symbol, timeframe='4h', limit=300)
        else:
            return None   # Alpaca doesn't support easy multi-TF for now
        if df4h is None or df4h.empty or len(df4h) < 60:
            return None
        df4h = add_indicators(df4h)
        last = df4h.iloc[-1]
        if pd.isna(last['ema50']) or pd.isna(last['ema200']):
            return None
        htf_bull = last['ema50'] > last['ema200'] and last['rsi'] > 45
        htf_bear = last['ema50'] < last['ema200'] and last['rsi'] < 55
        if primary_signal == 'BUY'  and htf_bull: return True
        if primary_signal == 'SELL' and htf_bear: return True
        return False
    except Exception:
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
                log.info(f"  [CACHE HIT] {symbol}")
                return cached    # return cached scores dict directly

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
        return None

    scores = parse_ai_scores(raw)

    with cache_lock:
        ai_cache[key] = (now, scores)

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
        return 1000.0   # fallback
    try:
        acct = alpaca.get_account()
        return float(acct.portfolio_value)
    except Exception:
        return 1000.0

def calc_position_size(price, stop_loss, portfolio_value):
    """Risk POSITION_RISK_PCT of portfolio per trade."""
    risk_per_share = abs(price - stop_loss)
    if risk_per_share <= 0:
        return 1
    dollar_risk = portfolio_value * POSITION_RISK_PCT
    qty = int(dollar_risk / risk_per_share)
    return max(qty, 1)

def place_paper_trade(signal_data):
    global daily_pnl
    if not alpaca or signal_data['asset_type'] != 'stock':
        return   # paper trade only stocks via Alpaca for now

    with trade_lock:
        portfolio_value = get_portfolio_value()

        # Max daily loss check
        if daily_pnl < -(portfolio_value * MAX_DAILY_LOSS_PCT):
            log.warning("⛔ Max daily loss reached — no new trades today")
            send_telegram("⛔ *Max daily loss protection triggered.* No new paper trades today.")
            return

        symbol = signal_data['symbol']
        try:
            price = float(signal_data['price'])
            sl    = float(str(signal_data['stop_loss']).replace(',', '') or 0)
            tp    = float(str(signal_data['take_profit']).replace(',', '') or 0)
        except (ValueError, TypeError):
            return

        if sl <= 0 or tp <= 0:
            return

        qty = calc_position_size(price, sl, portfolio_value)
        side = 'buy' if signal_data['signal'] == 'BUY' else 'sell'

        try:
            # Trailing stop order
            trail_pct = TRAILING_STOP_PCT * 100
            alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day',
                trail_percent=trail_pct,
            )
            log.info(f"📤 Paper trade: {side.upper()} {qty}x {symbol} @ ~{price}")

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
                f"📤 *Paper Trade Opened*\n"
                f"`{side.upper()} {qty}x {symbol}`\n"
                f"Entry ≈ `{price}` | SL `{sl}` | TP `{tp}`\n"
                f"Trail stop: `{trail_pct}%`"
            )
        except Exception as e:
            log.error(f"Paper trade error for {symbol}: {e}")

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
        log.warning(f"Indicator error {symbol}: {e}")
        return None

    latest = df.iloc[-1]

    # Guard NaN
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
    volume_spike   = vol_spike >= 1.5   # 50 % above average

    signal = None

    # ── BUY conditions ──────────────────────────────────────────────────
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

    # ── SELL conditions ─────────────────────────────────────────────────
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

    # ── Multi-timeframe confirmation ─────────────────────────────────────
    mtf = multi_timeframe_confirm(symbol, signal, asset_type)
    mtf_label = {True: 'CONFIRMED', False: 'REJECTED', None: 'N/A'}[mtf]
    if mtf is False:
        log.info(f"  ↩ {symbol} MTF rejected ({signal} on 1h but opposite on 4h)")
        return None

    # ── AI analysis (only for signals passing technical filter) ──────────
    scores = get_ai_analysis(
        symbol, signal, price, ema20, ema50, ema200,
        rsi, macd, macd_sig, volume, vol_avg,
        atr, trend_strength, support, resistance, breakout
    )
    if scores is None:
        return None

    # ── Risk/reward ratio ────────────────────────────────────────────────
    try:
        sl_price = float(str(scores['stop_loss']).replace(',', '') or 0)
        tp_price = float(str(scores['take_profit']).replace(',', '') or 0)
        rr = calc_rr_ratio(price, sl_price, tp_price, signal)
    except (ValueError, TypeError):
        rr = 0

    # ── Quality filter ───────────────────────────────────────────────────
    threshold   = HIGH_QUALITY_SCORES[asset_type]
    confidence  = scores['confidence']
    profitability = scores['profitability']

    if confidence < threshold['confidence'] or profitability < threshold['profitability']:
        log.info(f"  ↩ {symbol} below quality threshold (conf={confidence}, profit={profitability})")
        return None

    # ── Build signal record ──────────────────────────────────────────────
    signal_data = {
        'timestamp':     datetime.now().strftime('%H:%M:%S'),
        'date':          datetime.now().strftime('%Y-%m-%d'),
        'time':          datetime.now().strftime('%H:%M:%S'),
        'symbol':        symbol,
        'asset_type':    asset_type,
        'signal':        signal,
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

    # Auto paper trade
    place_paper_trade(signal_data)

    log.info(f"✅ {signal} {symbol} | P:{profitability} C:{confidence} RR:{rr} MTF:{mtf_label}")
    return signal_data

# ════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ════════════════════════════════════════════════════════════════════════════
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        log.error(f"Telegram send failed: {e}")

def send_telegram_summary(signals):
    if not signals:
        return
    sorted_signals = sorted(signals, key=lambda x: x['profitability'], reverse=True)
    msg = (f"*📊 TRADE GRID ANALYSIS*\n"
           f"_High quality signals — {datetime.now().strftime('%d %b %Y %H:%M')}_\n\n")
    for i, s in enumerate(sorted_signals[:8], 1):
        rr_str = f" | RR `{s['rr_ratio']}`" if s.get('rr_ratio') else ''
        mtf_str = f" | MTF `{s.get('mtf', 'N/A')}`"
        msg += (
            f"*#{i} {s['signal']} {s['symbol']}*\n"
            f"💰 P:`{s['profitability']}/10` 🛡 S:`{s['safety']}/10` "
            f"⚠️ R:`{s['risk']}/10` 🎯 C:`{s['confidence']}/10`\n"
            f"📍 Entry:`{s['entry']}` SL:`{s['stop_loss']}` TP:`{s['take_profit']}`"
            f"{rr_str}{mtf_str}\n"
            f"📈 Trend:`{s.get('trend_strength','?')}/10` "
            f"Breakout:`{s.get('breakout','?')}`\n"
            f"💡 _{s['reason']}_\n\n"
        )
    msg += "🌐 _Open Trade Grid dashboard for full rankings_"
    send_telegram(msg)

def handle_telegram_commands():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    last_update_id = None

    while True:
        try:
            params = {'timeout': 30, 'offset': last_update_id}
            response = requests.get(url, params=params, timeout=35)
            data = response.json()

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
                        f"*📊 Bot Status*\n"
                        f"Last scan: `{last_scan_time}`\n"
                        f"Assets: `{len(crypto_pairs) + len(stock_pairs) + len(forex_pairs)}`\n"
                        f"Current signals: `{len(all_signals)}`\n"
                        f"Total found: `{total_signals_found}`\n"
                        f"Daily P&L: `{daily_pnl:.4f}`\n"
                        f"Scans run: `{scan_count}`\n"
                        f"Status: ✅ Running"
                    )
                    send_telegram(reply)

                elif cmd == '/topsignals':
                    if not all_signals:
                        send_telegram("No signals in current scan. Wait for next scan.")
                    else:
                        top = sorted(all_signals, key=lambda x: x['profitability'], reverse=True)[:5]
                        reply = "*🏆 Top Signals*\n\n"
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
                        reply = f"*🌟 Best Signals Today ({today})*\n\n"
                        for i, s in enumerate(best, 1):
                            reply += f"*#{i} {s.get('signal')} {s.get('symbol')}* — P:`{s.get('profitability')}/10`\n"
                        send_telegram(reply)

                elif cmd == '/wins':
                    history = load_history()
                    wins = [h for h in history if str(h.get('outcome', '')).upper() == 'WIN']
                    reply = f"*✅ Wins: {len(wins)} total*\n"
                    for s in wins[-5:]:
                        reply += f"`{s.get('symbol')}` {s.get('signal')} P&L:`{s.get('pnl','?')}`\n"
                    send_telegram(reply)

                elif cmd == '/losses':
                    history = load_history()
                    losses = [h for h in history if str(h.get('outcome', '')).upper() == 'LOSS']
                    reply = f"*❌ Losses: {len(losses)} total*\n"
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
                            f"*📁 Signal History*\n"
                            f"Total: `{stats['total']}`\n"
                            f"Win rate: `{stats['win_rate']}%`\n"
                            f"Wins: `{stats['wins']}` | Losses: `{stats['losses']}`\n"
                            f"Avg P&L: `{stats['avg_pnl']}`\n"
                            f"Top assets: {', '.join(a['symbol'] for a in stats['top_assets'][:3])}"
                        )
                        send_telegram(reply)

                elif cmd == '/help':
                    reply = (
                        "*🤖 Trade Grid Analysis*\n\n"
                        "`/status` — Bot status\n"
                        "`/topsignals` — Current top signals\n"
                        "`/besttoday` — Best signals today\n"
                        "`/wins` — Win history\n"
                        "`/losses` — Loss history\n"
                        "`/history` — Stats summary\n"
                        "`/help` — This message\n\n"
                        "Or type any asset symbol (e.g. `BTC/USD`) to look it up."
                    )
                    send_telegram(reply)

                else:
                    # Individual asset lookup
                    sym = text.strip().upper()
                    if sym:
                        history = load_history()
                        matches = [h for h in history if h.get('symbol','').upper() == sym]
                        if matches:
                            recent = matches[-3:]
                            reply = f"*🔍 {sym} — last {len(recent)} signal(s)*\n\n"
                            for s in reversed(recent):
                                reply += (
                                    f"`{s.get('date')} {s.get('time')}` "
                                    f"{s.get('signal')} P:`{s.get('profitability')}/10`\n"
                                )
                            send_telegram(reply)
                        else:
                            send_telegram(f"No signals found for `{sym}` yet.")

        except Exception as e:
            log.error(f"Telegram command error: {e}")
            time.sleep(5)

# ════════════════════════════════════════════════════════════════════════════
# MAIN SCANNER
# ════════════════════════════════════════════════════════════════════════════
def scan_asset(args):
    """Worker function for thread pool."""
    symbol, asset_type, fetch_fn = args
    try:
        df = fetch_fn(symbol)
        if df is None or df.empty:
            return None
        return check_signal(symbol, df, asset_type)
    except Exception as e:
        log.warning(f"Scan error {symbol}: {e}")
        return None

def run_scanner():
    global all_signals, last_scan_time, scan_count, daily_pnl

    # Reset daily P&L at midnight
    if datetime.now().hour == 0 and datetime.now().minute < 3:
        daily_pnl = 0.0

    scan_count += 1
    last_scan_time = datetime.now().strftime('%d %b %Y %H:%M')
    log.info(f"\n🔍 Scan #{scan_count} — {last_scan_time}")

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
        log.info(f"✅ Scan complete — {len(all_signals)} signals")
    else:
        log.info(f"✅ Scan complete — No high quality signals")

    log.info(f"   Monitored: {len(tasks)} assets total")

# ════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — indicator sanity checks
# ════════════════════════════════════════════════════════════════════════════
def run_unit_tests():
    results = []
    try:
        import numpy as np
        # Synthetic OHLCV data
        n = 300
        close = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        high  = close + abs(np.random.randn(n))
        low   = close - abs(np.random.randn(n))
        vol   = pd.Series(np.random.randint(1000, 5000, n), dtype=float)
        df = pd.DataFrame({'open': close, 'high': high, 'low': low,
                           'close': close, 'volume': vol})

        df = add_indicators(df)

        def test(name, cond):
            status = "PASS" if cond else "FAIL"
            results.append(f"{status}  {name}")
            return cond

        test("EMA20 computed",     not df['ema20'].isnull().all())
        test("EMA50 computed",     not df['ema50'].isnull().all())
        test("EMA200 computed",    not df['ema200'].isnull().all())
        test("RSI range 0-100",    df['rsi'].dropna().between(0, 100).all())
        test("MACD computed",      not df['macd'].isnull().all())
        test("ATR positive",       (df['atr'].dropna() >= 0).all())
        test("ADX positive",       (df['adx'].dropna() >= 0).all())
        test("Support <= close",   (df['support'].dropna() <= df['close'][df['support'].notna()]).all())
        test("Resistance >= close",(df['resistance'].dropna() >= df['close'][df['resistance'].notna()]).all())
        test("Vol spike computed", not df['vol_spike'].isnull().all())
        test("BB upper > lower",   (df['bb_upper'].dropna() > df['bb_lower'].dropna()).all())

        passed = sum(1 for r in results if r.startswith("PASS"))
        log.info(f"🧪 Unit tests: {passed}/{len(results)} passed")
        with open(UNIT_TEST_LOG, 'w') as f:
            f.write(f"Unit test run: {datetime.now()}\n")
            f.write('\n'.join(results))
    except Exception as e:
        log.error(f"Unit test error: {e}")

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
/* Win rate bar */
.wr-bar{height:8px;background:var(--surface2);border-radius:4px;overflow:hidden;margin-top:4px}
.wr-fill{height:100%;background:var(--green);border-radius:4px;transition:width .4s}
/* Guide */
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
/* Stats tab */
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:20px}
.big-stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px}
.big-stat-val{font-size:28px;font-weight:700;margin-bottom:4px}
.big-stat-lbl{font-size:11px;color:var(--muted2);text-transform:uppercase;letter-spacing:.6px}
.top-assets-table td{padding:6px 10px;font-size:12px}
@media(max-width:480px){.header{padding:0 12px}.logo{font-size:12px}.main{padding:12px}.stat-value{font-size:18px}table{font-size:11px}th,td{padding:6px 8px}.reason-text{max-width:100px;font-size:9px}}
</style>
</head>
<body>
<div class="header">
  <div class="logo"><div class="logo-dot"></div>Trade Grid Analysis</div>
  <div class="header-right">
    <div class="status-badge"><div class="status-dot"></div>Live</div>
    <div class="last-update" id="lastUpdate">Loading...</div>
    <button class="theme-btn" onclick="toggleTheme()">☀ / ☾</button>
  </div>
</div>

<div class="main">
  <!-- STATS ROW -->
  <div class="stats-row">
    <div class="stat-card"><div class="stat-label">Current Signals</div><div class="stat-value" id="statCurrent">0</div><div class="stat-sub">This scan</div></div>
    <div class="stat-card"><div class="stat-label">Total History</div><div class="stat-value" id="statTotal">0</div><div class="stat-sub">All time</div></div>
    <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value" id="statWinRate">—</div><div class="wr-bar"><div class="wr-fill" id="wrFill" style="width:0%"></div></div></div>
    <div class="stat-card"><div class="stat-label">Total P&amp;L</div><div class="stat-value" id="statPnl">—</div><div class="stat-sub">Paper trades</div></div>
    <div class="stat-card"><div class="stat-label">Assets Monitored</div><div class="stat-value" id="statAssets">—</div><div class="stat-sub">Crypto · Stocks · Forex</div></div>
    <div class="stat-card"><div class="stat-label">Scan Interval</div><div class="stat-value">2.5m</div><div class="stat-sub">Auto-refresh 30s</div></div>
  </div>

  <!-- TABS -->
  <div class="tabs">
    <button class="tab active" onclick="switchTab('signals',event)">📊 Live Signals</button>
    <button class="tab" onclick="switchTab('history',event)">📁 History</button>
    <button class="tab" onclick="switchTab('performance',event)">📈 Performance</button>
    <button class="tab" onclick="switchTab('journal',event)">📒 Trade Journal</button>
    <button class="tab" onclick="switchTab('guide',event)">📖 How To Trade</button>
  </div>

  <!-- LIVE SIGNALS TAB -->
  <div id="tab-signals" class="tab-content active">
    <div class="controls">
      <span class="filter-label">Sort:</span>
      <button class="sort-btn active" onclick="sortTable('profitability',event)">💰 Profit</button>
      <button class="sort-btn" onclick="sortTable('safety',event)">🛡 Safety</button>
      <button class="sort-btn" onclick="sortTable('risk',event)">⚠️ Risk</button>
      <button class="sort-btn" onclick="sortTable('confidence',event)">🎯 Conf</button>
      <button class="sort-btn" onclick="sortTable('rr_ratio',event)">⚖️ RR</button>
      <button class="sort-btn" onclick="sortTable('trend_strength',event)">📈 Trend</button>
      <select class="filter-select" id="typeFilter" onchange="renderSignals()">
        <option value="">All types</option>
        <option value="crypto">Crypto</option>
        <option value="stock">Stock</option>
        <option value="forex">Forex</option>
      </select>
      <select class="filter-select" id="signalFilter" onchange="renderSignals()">
        <option value="">BUY & SELL</option>
        <option value="BUY">BUY only</option>
        <option value="SELL">SELL only</option>
      </select>
      <button class="refresh-btn" onclick="loadSignals()">↻ Refresh</button>
    </div>
    <div class="table-wrap"><table>
      <thead><tr>
        <th onclick="sortTable('symbol',event)">Symbol ↕</th>
        <th onclick="sortTable('signal',event)">Signal ↕</th>
        <th onclick="sortTable('price',event)">Price ↕</th>
        <th onclick="sortTable('profitability',event)">Profit ↕</th>
        <th onclick="sortTable('safety',event)">Safety ↕</th>
        <th onclick="sortTable('risk',event)">Risk ↕</th>
        <th onclick="sortTable('confidence',event)">Conf ↕</th>
        <th onclick="sortTable('rr_ratio',event)">RR ↕</th>
        <th onclick="sortTable('trend_strength',event)">Trend ↕</th>
        <th>Breakout</th>
        <th>MTF</th>
        <th>Entry</th><th>Stop Loss</th><th>Take Profit</th>
        <th>RSI</th><th>MACD</th><th>ATR</th>
        <th>Reason</th>
        <th onclick="sortTable('timestamp',event)">Time ↕</th>
      </tr></thead>
      <tbody id="tableBody"><tr><td colspan="19" class="no-signals"><div class="no-signals-icon">📡</div>Scanning markets…</td></tr></tbody>
    </table></div>
  </div>

  <!-- HISTORY TAB -->
  <div id="tab-history" class="tab-content">
    <div class="controls">
      <input class="search-input" type="text" id="historySearch" placeholder="Search symbol…" oninput="renderHistory()">
      <select class="filter-select" id="histTypeFilter" onchange="renderHistory()">
        <option value="">All types</option>
        <option value="crypto">Crypto</option>
        <option value="stock">Stock</option>
        <option value="forex">Forex</option>
      </select>
      <button class="sort-btn" onclick="sortHistory('profitability',event)">💰 Profit</button>
      <button class="sort-btn" onclick="sortHistory('confidence',event)">🎯 Conf</button>
      <button class="sort-btn" onclick="sortHistory('date',event)">📅 Date</button>
      <button class="refresh-btn" onclick="loadHistory()">↻ Refresh</button>
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

  <!-- PERFORMANCE TAB -->
  <div id="tab-performance" class="tab-content">
    <div class="stats-grid" id="perfStats">
      <div class="big-stat"><div class="big-stat-val" id="perfWR">—</div><div class="big-stat-lbl">Win Rate</div></div>
      <div class="big-stat"><div class="big-stat-val" id="perfWins">—</div><div class="big-stat-lbl">Total Wins</div></div>
      <div class="big-stat"><div class="big-stat-val" id="perfLosses">—</div><div class="big-stat-lbl">Total Losses</div></div>
      <div class="big-stat"><div class="big-stat-val" id="perfAvgPnl">—</div><div class="big-stat-lbl">Avg P&amp;L per Trade</div></div>
      <div class="big-stat"><div class="big-stat-val" id="perfTotal">—</div><div class="big-stat-lbl">Total Signals</div></div>
    </div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px">
      <div class="stat-label" style="margin-bottom:10px">Top Performing Assets</div>
      <table class="top-assets-table" style="width:100%;border-collapse:collapse">
        <thead><tr><th style="text-align:left;font-size:10px;color:var(--muted2);padding:6px 10px">Symbol</th><th style="text-align:left;font-size:10px;color:var(--muted2)">Signal Count</th></tr></thead>
        <tbody id="topAssetsBody"><tr><td colspan="2" style="color:var(--muted2);padding:10px;font-size:12px">No data yet</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- TRADE JOURNAL TAB -->
  <div id="tab-journal" class="tab-content">
    <div class="controls">
      <input class="search-input" type="text" id="journalSearch" placeholder="Search symbol…" oninput="renderJournal()">
      <button class="refresh-btn" onclick="loadJournal()">↻ Refresh</button>
    </div>
    <div class="table-wrap"><table>
      <thead><tr>
        <th>Date</th><th>Time</th><th>Symbol</th><th>Dir</th><th>Qty</th>
        <th>Entry</th><th>Stop Loss</th><th>Take Profit</th><th>Status</th><th>P&amp;L</th><th>Notes</th>
      </tr></thead>
      <tbody id="journalBody"><tr><td colspan="11" class="no-signals"><div class="no-signals-icon">📒</div>No paper trades yet</td></tr></tbody>
    </table></div>
  </div>

  <!-- HOW TO TRADE TAB -->
  <div id="tab-guide" class="tab-content">
    <div class="guide-full"><h2>What This Bot Does</h2>
      <p>Trade Grid Analysis scans assets every 2.5 minutes across crypto, stocks, commodities and indices using EMA, RSI, MACD, ATR, ADX, Bollinger Bands, Support/Resistance and multi-timeframe confirmation. Only setups passing the technical filter are sent to AI for scoring. High quality signals (crypto 7/10, stocks 6/10) appear here and in your Telegram.</p>
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
          <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">ADX / Trend Strength</strong> — Scores trend power 0–10. Above 25 = strong trend. Bot requires trend for signals.</li>
          <li style="padding:6px 0;font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">Risk/Reward (RR)</strong> — Potential profit ÷ potential loss. Aim for RR ≥ 2 (earn twice what you risk).</li>
        </ul></div>
        <div><ul style="list-style:none;padding:0">
          <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">Breakout</strong> — Price breaking above resistance or below support. Bullish/Bearish breakout = strong signal.</li>
          <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">Volume Spike</strong> — Volume 50%+ above average. Confirms real momentum behind a move.</li>
          <li style="padding:6px 0;font-size:12px;color:var(--muted2)"><strong style="color:var(--text)">MTF (Multi-Timeframe)</strong> — CONFIRMED = 4h trend agrees with 1h signal. REJECTED = contradicts.</li>
        </ul></div>
      </div>
    </div>
    <div class="guide-full"><h2>Step By Step — How To Place A Trade</h2>
      <div class="step-list" style="margin-top:12px">
        <div class="step"><div class="step-num">1</div><div class="step-text"><strong>Wait for CONFIRMED + high RR signals</strong> — Look for MTF: CONFIRMED, RR ≥ 2.0, Confidence ≥ 7/10.</div></div>
        <div class="step"><div class="step-num">2</div><div class="step-text"><strong>Open your platform and search the asset</strong> — E.g. BTC/USD on Plus500, AAPL on Alpaca.</div></div>
        <div class="step"><div class="step-num">3</div><div class="step-text"><strong>Size your trade</strong> — Risk 2–5% of account per trade. With R500 that's R10–R25.</div></div>
        <div class="step"><div class="step-num">4</div><div class="step-text"><strong>Set Stop Loss and Take Profit</strong> — Use the exact values the bot provides. Always. No exceptions.</div></div>
        <div class="step"><div class="step-num">5</div><div class="step-text"><strong>Open the trade and let it run</strong> — Trust your levels. Don't move your stop loss.</div></div>
        <div class="step"><div class="step-num">6</div><div class="step-text"><strong>Journal every trade</strong> — The Trade Journal tab auto-records paper trades. Add manual trades too.</div></div>
      </div>
    </div>
    <div class="guide-full"><h2>Golden Rules</h2>
      <div class="rules-grid">
        <div class="rule"><strong>Max 5% per trade</strong>Never risk more than 5% of your account on one trade</div>
        <div class="rule"><strong>Always set a stop loss</strong>No exceptions. Ever. Protects your account.</div>
        <div class="rule"><strong>Only trade CONFIRMED signals</strong>MTF confirmed + RR ≥ 2 = much higher win rate</div>
        <div class="rule"><strong>Never chase losses</strong>If a trade goes wrong, step back. Don't revenge trade.</div>
        <div class="rule"><strong>Aim for RR ≥ 2</strong>You can be wrong 40% of the time and still be profitable</div>
        <div class="rule"><strong>Quality over quantity</strong>2 great trades a week beats 20 bad ones every time</div>
      </div>
    </div>
  </div>
</div>

<script>
let signals=[], historyData=[], journalData=[];
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
  if(tab==='history') loadHistory();
  if(tab==='performance') loadPerformance();
  if(tab==='journal') loadJournal();
}

function sc(score){
  score=parseInt(score)||0;
  let cls = score>=7?'score-high':score>=4?'score-mid':'score-low';
  return `<span class="score-pill ${cls}">${score}/10</span>`;
}

function breakoutBadge(b){
  if(b==='BULLISH_BREAKOUT') return '<span class="badge-pill badge-bull">▲ Bull</span>';
  if(b==='BEARISH_BREAKOUT') return '<span class="badge-pill badge-bear">▼ Bear</span>';
  return '<span class="badge-pill badge-none">—</span>';
}

function mtfBadge(m){
  if(m==='CONFIRMED') return '<span class="badge-pill badge-bull">✓ Conf</span>';
  if(m==='REJECTED')  return '<span class="badge-pill badge-bear">✗ Rej</span>';
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

function loadSignals(){
  fetch('/signals').then(r=>r.json()).then(data=>{
    signals=data.signals||[];
    document.getElementById('statCurrent').textContent=signals.length;
    document.getElementById('statTotal').textContent=data.total||0;
    document.getElementById('statAssets').textContent=data.asset_count||'—';
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
setInterval(loadSignals,30000);
</script>
</body>
</html>"""

# ════════════════════════════════════════════════════════════════════════════
# WEB SERVER
# ════════════════════════════════════════════════════════════════════════════
class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        routes = {
            '/':        ('text/html',         lambda: DASHBOARD_HTML.encode()),
            '/signals': ('application/json',  self._signals),
            '/history': ('application/json',  self._history),
            '/stats':   ('application/json',  self._stats),
            '/journal': ('application/json',  self._journal),
            '/health':  ('application/json',  self._health),
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

    def _health(self):
        return json.dumps({'status': 'ok', 'last_scan': last_scan_time, 'signals': len(all_signals)})

    def log_message(self, format, *args):
        pass   # suppress default request logs

def start_dashboard():
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    log.info(f"🌐 Dashboard running on port {port}")
    server.serve_forever()

# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    log.info("🚀 Trade Grid Analysis starting...")

    # Run unit tests on startup
    run_unit_tests()

    # Initialise CSV files
    init_csv()

    # Start dashboard thread
    threading.Thread(target=start_dashboard, daemon=True).start()

    # Start Telegram command listener thread
    threading.Thread(target=handle_telegram_commands, daemon=True).start()
    log.info("📱 Telegram listener started")

    # First scan immediately
    run_scanner()

    # Schedule every 2 minutes 30 seconds
    schedule.every(2).minutes.do(run_scanner)
    schedule.every(30).seconds.do(run_scanner)   # offset trick → runs at 0:00, 0:30, 2:00, 2:30, 4:00...
    # Simpler direct approach:
    schedule.clear()
    schedule.every(150).seconds.do(run_scanner)  # 150s = 2m30s exactly

    log.info("⏰ Scanning every 2m 30s. Press Ctrl+C to stop.")

    while True:
        try:
            schedule.run_pending()
            time.sleep(15)
        except KeyboardInterrupt:
            log.info("👋 Shutting down Trade Grid Analysis")
            break
        except Exception as e:
            log.error(f"Scheduler error: {e}\n{traceback.format_exc()}")
            time.sleep(30)