import ccxt
import pandas as pd
import ta
import requests
import schedule
import time
import json
import csv
import os
from groq import Groq
import alpaca_trade_api as tradeapi
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from datetime import datetime

# ---- CONFIG ----
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET")
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# ---- MINIMUM SCORES PER ASSET TYPE ----
MIN_SCORES = {
    'crypto': {'confidence': 8, 'profitability': 8},
    'stock':  {'confidence': 7, 'profitability': 7},
    'forex':  {'confidence': 7, 'profitability': 7},
}

# ---- CSV HISTORY FILE ----
HISTORY_FILE = 'signal_history.csv'
HISTORY_FIELDS = [
    'date', 'time', 'symbol', 'asset_type', 'signal',
    'price', 'rsi', 'macd', 'confidence', 'profitability',
    'safety', 'risk', 'entry', 'stop_loss', 'take_profit', 'reason'
]

# Initialize clients
exchange = ccxt.kraken()
client = Groq(api_key=GROQ_API_KEY)
alpaca = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET, ALPACA_BASE_URL)

all_signals = []
last_scan_time = "Never"
total_signals_found = 0

# ---- ASSET LISTS ----
crypto_pairs = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'DOGE/USD',
    'BNB/USD', 'ADA/USD', 'AVAX/USD', 'LINK/USD', 'INJ/USD',
    'FET/USD', 'ARB/USD', 'OP/USD', 'TIA/USD', 'POLYGON/USD',
    'DOT/USD', 'ATOM/USD', 'LTC/USD', 'UNI/USD', 'NEAR/USD',
    'RENDER/USD', 'BLUR/USD', 'WLD/USD', 'LDO/USD', 'PEPE/USD',
    'SHIB/USD', 'BONK/USD', 'WIF/USD', 'FLOKI/USD', 'AAVE/USD',
    'CRVE/USD', 'GMX/USD', 'APT/USD', 'SUI/USD', 'ZEC/USD',
    'XMR/USD', 'BCH/USD', 'ETC/USD', 'XLM/USD', 'VET/USD',
    'KSM/USD', 'THETA/USD', 'ONE/USD', 'AGIX/USD', 'RAIN/USD',
    'CHZ/USD', 'JASMY/USD', 'JUP/USD', 'SEI/USD', 'HBAR/USD'
]

stock_pairs = [
    'AAPL', 'TSLA', 'NVDA', 'AMZN', 'META',
    'GOOGL', 'MSFT', 'AMD', 'NFLX', 'COIN',
    'SPY', 'QQQ', 'DIA', 'GLD', 'SLV',
    'USO', 'BNO', 'TLT', 'AGG', 'VIX',
    'SOFI', 'RIOT', 'MARA', 'MSTR', 'CLSK',
    'UPRO', 'TQQQ', 'SSO', 'EEM', 'ARKK',
    'ARKW', 'XLK', 'XLV', 'XLF', 'XLE',
    'XLI', 'XLY', 'XLP', 'XLRE', 'XLU',
    'SCHX', 'SCHB', 'SCHF', 'SCHE', 'SCHP',
    'VTSAX', 'VTI', 'VOO', 'VTIAX', 'BRK.B'
]

forex_pairs = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD',
    'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'EURCHF',
    'GBPJPY', 'GBPCHF', 'AUDJPY', 'CADJPY', 'CHFJPY',
    'EURAUD', 'EURNZD', 'GBPAUD', 'AUDNZD', 'USDSEK',
    'USDNOK', 'USDDKK', 'EURSEK', 'EURNOK', 'EURDKK',
    'GBPSEK', 'GBPNOK', 'USDSGD', 'USDHKD', 'AUDSGD',
    'EURSGD', 'GBPSGD', 'EURHKD', 'AUDHKD', 'NZDCAD',
    'NZDCHF', 'NZDJPY', 'GBPCAD', 'GBPNZD', 'CADCHF',
    'AUDNZD', 'AUDCAD', 'AUDCHF', 'NZDSGD', 'EURCAD',
    'GBPNZD', 'CHFSGD', 'CADSGD', 'JPYSGD', 'HKDJPY'
]

# ---- CSV HISTORY ----
def init_csv():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            writer.writeheader()

def save_signal_to_csv(signal_data):
    with open(HISTORY_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writerow({k: signal_data.get(k, '') for k in HISTORY_FIELDS})

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

# ---- TELEGRAM ----
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=payload)

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
                text = message.get('text', '').strip().lower()
                chat_id = message.get('chat', {}).get('id')

                if not chat_id:
                    continue

                if text == '/status':
                    reply = (
                        f"*📊 Bot Status*\n"
                        f"Last scan: `{last_scan_time}`\n"
                        f"Assets monitored: `{len(crypto_pairs) + len(stock_pairs)}`\n"
                        f"Signals this scan: `{len(all_signals)}`\n"
                        f"Total signals found: `{total_signals_found}`\n"
                        f"Scanning every: `2 minutes`\n"
                        f"Status: `✅ Running`"
                    )
                    send_telegram(reply)

                elif text == '/topsignals':
                    if not all_signals:
                        send_telegram("No high quality signals in the current scan. Wait for the next scan.")
                    else:
                        sorted_signals = sorted(all_signals, key=lambda x: x['profitability'], reverse=True)
                        reply = "*🏆 Top Signals Right Now*\n\n"
                        for i, s in enumerate(sorted_signals[:5], 1):
                            reply += (
                                f"*#{i} {s['signal']} {s['symbol']}*\n"
                                f"💰 Profit: `{s['profitability']}/10` | 🛡 Safety: `{s['safety']}/10`\n"
                                f"📍 Entry: `{s['entry']}` | SL: `{s['stop_loss']}` | TP: `{s['take_profit']}`\n"
                                f"💡 _{s['reason']}_\n\n"
                            )
                        send_telegram(reply)

                elif text == '/help':
                    reply = (
                        "*🤖 Trade Grid Analysis Bot*\n\n"
                        "Available commands:\n"
                        "`/status` — Bot status and scan info\n"
                        "`/topsignals` — Top signals from latest scan\n"
                        "`/help` — Show this message"
                    )
                    send_telegram(reply)

        except Exception as e:
            print(f"Telegram command error: {e}")
            time.sleep(5)

# ---- DATA FETCHING ----
def get_crypto_ohlcv(symbol, timeframe='1h', limit=200):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def get_alpaca_ohlcv(symbol):
    bars = alpaca.get_bars(symbol, tradeapi.rest.TimeFrame.Hour, limit=200).df
    bars = bars.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    return bars

def get_forex_ohlcv(symbol):
    bars = alpaca.get_bars(symbol, tradeapi.rest.TimeFrame.Hour, limit=200).df
    bars = bars.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    return bars

# ---- INDICATORS INCLUDING MACD ----
def add_indicators(df):
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['vol_avg'] = df['volume'].rolling(window=20).mean()

    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    return df

# ---- AI ANALYSIS ----
def get_ai_analysis(symbol, signal, price, ema20, ema50, ema200, rsi, macd, macd_signal, volume, vol_avg):
    prompt = f"""
You are a professional trading analyst covering crypto, stocks, forex and commodities.

Analyze this trade setup and respond in EXACTLY this format with numbers only for scores:

Pair: {symbol}
Signal: {signal}
Price: {price}
EMA20: {ema20}
EMA50: {ema50}
EMA200: {ema200}
RSI: {rsi}
MACD: {macd}
MACD Signal: {macd_signal}
Volume: {volume} (Average: {vol_avg})

Respond in EXACTLY this format:
Confidence: X/10
Profitability: X/10
Safety: X/10
Risk: X/10
Entry: X
Stop Loss: X
Take Profit: X
Reason: One sentence explanation
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250
    )
    return response.choices[0].message.content

def parse_ai_scores(ai_text):
    scores = {
        'confidence': 5, 'profitability': 5,
        'safety': 5, 'risk': 5,
        'entry': 0, 'stop_loss': 0,
        'take_profit': 0, 'reason': ''
    }
    try:
        for line in ai_text.split('\n'):
            line = line.strip()
            if line.startswith('Confidence:'):
                scores['confidence'] = int(line.split(':')[1].strip().split('/')[0])
            elif line.startswith('Profitability:'):
                scores['profitability'] = int(line.split(':')[1].strip().split('/')[0])
            elif line.startswith('Safety:'):
                scores['safety'] = int(line.split(':')[1].strip().split('/')[0])
            elif line.startswith('Risk:'):
                scores['risk'] = int(line.split(':')[1].strip().split('/')[0])
            elif line.startswith('Entry:'):
                scores['entry'] = line.split(':')[1].strip()
            elif line.startswith('Stop Loss:'):
                scores['stop_loss'] = line.split(':')[1].strip()
            elif line.startswith('Take Profit:'):
                scores['take_profit'] = line.split(':')[1].strip()
            elif line.startswith('Reason:'):
                scores['reason'] = line.split(':', 1)[1].strip()
    except:
        pass
    return scores

# ---- SIGNAL CHECK ----
def check_signal(symbol, df, asset_type):
    global total_signals_found
    df = add_indicators(df)
    latest = df.iloc[-1]

    price = latest['close']
    ema20 = latest['ema20']
    ema50 = latest['ema50']
    ema200 = latest['ema200']
    rsi = latest['rsi']
    volume = latest['volume']
    vol_avg = latest['vol_avg']
    macd = round(latest['macd'], 6)
    macd_signal = round(latest['macd_signal'], 6)
    macd_diff = latest['macd_diff']

    signal = None

    # BUY: price above EMA20, RSI not overbought, volume strong, MACD positive
    if price > ema20 and rsi < 70 and volume > vol_avg * 0.8 and macd_diff > 0:
        signal = "BUY 🟢"
    # SELL: price below EMA20, RSI not oversold, volume strong, MACD negative
    elif price < ema20 and rsi > 30 and volume > vol_avg * 0.8 and macd_diff < 0:
        signal = "SELL 🔴"

    if signal:
        print(f"{symbol} — {signal} detected, getting AI analysis...")
        ai_text = get_ai_analysis(symbol, signal, price, ema20, ema50, ema200, rsi, macd, macd_signal, volume, vol_avg)
        scores = parse_ai_scores(ai_text)

        # Per asset type minimum filter
        min_conf = MIN_SCORES.get(asset_type, {}).get('confidence', 7)
        min_prof = MIN_SCORES.get(asset_type, {}).get('profitability', 7)

        if scores['confidence'] >= min_conf and scores['profitability'] >= min_prof:
            now = datetime.now()
            signal_data = {
                'symbol': symbol,
                'signal': signal,
                'price': round(price, 4),
                'rsi': round(rsi, 2),
                'macd': macd,
                'asset_type': asset_type,
                'confidence': scores['confidence'],
                'profitability': scores['profitability'],
                'safety': scores['safety'],
                'risk': scores['risk'],
                'entry': scores['entry'],
                'stop_loss': scores['stop_loss'],
                'take_profit': scores['take_profit'],
                'reason': scores['reason'],
                'timestamp': now.strftime('%H:%M:%S'),
                'date': now.strftime('%Y-%m-%d'),
                'time': now.strftime('%H:%M:%S'),
            }
            all_signals.append(signal_data)
            save_signal_to_csv(signal_data)
            total_signals_found += 1
            print(f"{symbol} — ✅ High quality signal! (Conf: {scores['confidence']}/10, Prof: {scores['profitability']}/10)")
        else:
            print(f"{symbol} — Below threshold for {asset_type} (Conf: {scores['confidence']}/10, Prof: {scores['profitability']}/10)")
    else:
        print(f"{symbol} — No signal")

# ---- TELEGRAM SUMMARY ----
def send_telegram_summary(signals):
    if not signals:
        return

    sorted_signals = sorted(signals, key=lambda x: x['profitability'], reverse=True)
    message = "*📊 TRADE GRID ANALYSIS*\n"
    message += f"_High quality signals — {datetime.now().strftime('%d %b %Y %H:%M')}_\n\n"

    for i, s in enumerate(sorted_signals[:10], 1):
        message += (
            f"*#{i} {s['signal']} {s['symbol']}*\n"
            f"💰 Profit: `{s['profitability']}/10` | "
            f"🛡 Safety: `{s['safety']}/10` | "
            f"⚠️ Risk: `{s['risk']}/10`\n"
            f"📍 Entry: `{s['entry']}` | SL: `{s['stop_loss']}` | TP: `{s['take_profit']}`\n"
            f"💡 _{s['reason']}_\n\n"
        )

    message += "🌐 _Open Trade Grid dashboard for full rankings_"
    send_telegram(message)

# ---- DASHBOARD HTML ----
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trade Grid Analysis</title>
    <style>
        :root {
            --bg: #0a0a0f;
            --surface: #111118;
            --surface2: #16161f;
            --border: #1e1e2e;
            --accent: #00d4ff;
            --accent2: #7c3aed;
            --green: #00c48c;
            --red: #ff4d6d;
            --yellow: #f59e0b;
            --text: #e2e8f0;
            --muted: #4a5568;
            --muted2: #718096;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }

        /* HEADER */
        .header {
            border-bottom: 1px solid var(--border);
            padding: 0 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 60px;
            background: var(--surface);
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 0.5px;
            color: var(--text);
        }
        .logo-dot {
            width: 8px; height: 8px;
            background: var(--accent);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent);
        }
        .header-right {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: var(--muted2);
        }
        .status-dot {
            width: 6px; height: 6px;
            background: var(--green);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .last-update {
            font-size: 12px;
            color: var(--muted);
        }

        /* MAIN LAYOUT */
        .main { padding: 24px 32px; }

        /* STATS ROW */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px 20px;
        }
        .stat-label {
            font-size: 11px;
            color: var(--muted2);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 6px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--text);
        }
        .stat-sub {
            font-size: 11px;
            color: var(--muted2);
            margin-top: 4px;
        }

        /* TABS */
        .tabs {
            display: flex;
            gap: 2px;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border);
        }
        .tab {
            padding: 10px 20px;
            font-size: 13px;
            color: var(--muted2);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
            transition: all 0.15s;
            background: none;
            border-top: none;
            border-left: none;
            border-right: none;
        }
        .tab:hover { color: var(--text); }
        .tab.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* CONTROLS */
        .controls {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            align-items: center;
        }
        .filter-label {
            font-size: 12px;
            color: var(--muted2);
            margin-right: 4px;
        }
        .sort-btn {
            padding: 6px 14px;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--muted2);
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.15s;
        }
        .sort-btn:hover { border-color: var(--accent); color: var(--text); }
        .sort-btn.active { border-color: var(--accent); color: var(--accent); background: rgba(0,212,255,0.05); }
        .refresh-btn {
            padding: 6px 14px;
            background: rgba(0,212,255,0.1);
            border: 1px solid var(--accent);
            color: var(--accent);
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            margin-left: auto;
        }
        .refresh-btn:hover { background: rgba(0,212,255,0.2); }

        /* TABLE */
        .table-wrap {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        thead { background: var(--surface2); }
        th {
            padding: 11px 14px;
            text-align: left;
            font-size: 11px;
            font-weight: 600;
            color: var(--muted2);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            cursor: pointer;
            white-space: nowrap;
            border-bottom: 1px solid var(--border);
        }
        th:hover { color: var(--text); }
        td {
            padding: 11px 14px;
            border-bottom: 1px solid var(--border);
            color: var(--text);
        }
        tbody tr:last-child td { border-bottom: none; }
        tbody tr:hover { background: var(--surface2); }

        .symbol { font-weight: 600; font-size: 13px; }
        .asset-type {
            font-size: 10px;
            color: var(--muted2);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .buy { color: var(--green); font-weight: 600; font-size: 12px; }
        .sell { color: var(--red); font-weight: 600; font-size: 12px; }

        .score-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 42px;
            height: 22px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 700;
        }
        .score-high { background: rgba(0,196,140,0.15); color: var(--green); }
        .score-mid { background: rgba(245,158,11,0.15); color: var(--yellow); }
        .score-low { background: rgba(255,77,109,0.15); color: var(--red); }

        .reason-text { font-size: 11px; color: var(--muted2); max-width: 200px; line-height: 1.4; }
        .no-signals {
            text-align: center;
            padding: 60px 20px;
            color: var(--muted);
            font-size: 13px;
        }
        .no-signals-icon { font-size: 32px; margin-bottom: 10px; }

        /* HISTORY TABLE */
        .history-controls {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            align-items: center;
        }
        .search-input {
            padding: 6px 12px;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            border-radius: 6px;
            font-size: 12px;
            width: 200px;
        }
        .search-input:focus {
            outline: none;
            border-color: var(--accent);
        }
        .search-input::placeholder { color: var(--muted); }

        /* GUIDE */
        .guide-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .guide-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }
        .guide-card h3 {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .guide-card p, .guide-card li {
            font-size: 13px;
            color: var(--muted2);
            line-height: 1.7;
        }
        .guide-card ul { padding-left: 16px; }
        .guide-card li { margin-bottom: 4px; }
        .guide-card li strong { color: var(--text); }
        .guide-full {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .guide-full h2 {
            font-size: 15px;
            font-weight: 600;
            color: var(--accent);
            margin-bottom: 12px;
        }
        .guide-full p { font-size: 13px; color: var(--muted2); line-height: 1.7; margin-bottom: 8px; }

        .step-list { display: flex; flex-direction: column; gap: 10px; }
        .step {
            display: flex;
            gap: 14px;
            align-items: flex-start;
        }
        .step-num {
            min-width: 24px; height: 24px;
            background: rgba(0,212,255,0.1);
            border: 1px solid var(--accent);
            color: var(--accent);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 700;
            margin-top: 2px;
        }
        .step-text { font-size: 13px; color: var(--muted2); line-height: 1.6; }
        .step-text strong { color: var(--text); }

        .platform-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; }
        .platform-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }
        .platform-card h4 { font-size: 14px; font-weight: 600; margin-bottom: 6px; }
        .platform-card p { font-size: 12px; color: var(--muted2); line-height: 1.6; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .badge-green { background: rgba(0,196,140,0.15); color: var(--green); }
        .badge-yellow { background: rgba(245,158,11,0.15); color: var(--yellow); }

        .rules-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .rule {
            background: var(--surface2);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px 14px;
            font-size: 12px;
            color: var(--muted2);
            line-height: 1.5;
        }
        .rule strong { color: var(--text); display: block; margin-bottom: 3px; }
    </style>
</head>
<body>

    <!-- HEADER -->
    <div class="header">
        <div class="logo">
            <div class="logo-dot"></div>
            Trade Grid Analysis
        </div>
        <div class="header-right">
            <div class="status-badge">
                <div class="status-dot"></div>
                Live
            </div>
            <div class="last-update" id="lastUpdate">Loading...</div>
        </div>
    </div>

    <div class="main">

        <!-- STATS ROW -->
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-label">Current Signals</div>
                <div class="stat-value" id="statCurrent">0</div>
                <div class="stat-sub">This scan</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total History</div>
                <div class="stat-value" id="statTotal">0</div>
                <div class="stat-sub">All time</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Assets Monitored</div>
                <div class="stat-value">0</div>
                <div class="stat-sub">Crypto, Stocks, ETFs, Forex</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Scan Interval</div>
                <div class="stat-value">2m</div>
                <div class="stat-sub">Auto-refresh 30s</div>
            </div>
        </div>

        <!-- TABS -->
        <div class="tabs">
            <button class="tab active" onclick="switchTab('signals', event)">📊 Live Signals</button>
            <button class="tab" onclick="switchTab('history', event)">📁 Signal History</button>
            <button class="tab" onclick="switchTab('guide', event)">📖 How To Trade</button>
        </div>

        <!-- LIVE SIGNALS TAB -->
        <div id="tab-signals" class="tab-content active">
            <div class="controls">
                <span class="filter-label">Sort by:</span>
                <button class="sort-btn active" onclick="sortTable('profitability', event)">💰 Profitability</button>
                <button class="sort-btn" onclick="sortTable('safety', event)">🛡 Safety</button>
                <button class="sort-btn" onclick="sortTable('risk', event)">⚠️ Risk</button>
                <button class="sort-btn" onclick="sortTable('confidence', event)">🎯 Confidence</button>
                <button class="sort-btn" onclick="sortTable('symbol', event)">A–Z Symbol</button>
                <button class="refresh-btn" onclick="loadSignals()">↻ Refresh</button>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th onclick="sortTable('symbol', event)">Symbol ↕</th>
                            <th onclick="sortTable('signal', event)">Signal ↕</th>
                            <th onclick="sortTable('price', event)">Price ↕</th>
                            <th onclick="sortTable('profitability', event)">Profit ↕</th>
                            <th onclick="sortTable('safety', event)">Safety ↕</th>
                            <th onclick="sortTable('risk', event)">Risk ↕</th>
                            <th onclick="sortTable('confidence', event)">Conf ↕</th>
                            <th>Entry</th>
                            <th>Stop Loss</th>
                            <th>Take Profit</th>
                            <th>MACD</th>
                            <th>RSI</th>
                            <th>Reason</th>
                            <th onclick="sortTable('timestamp', event)">Time ↕</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody">
                        <tr><td colspan="14" class="no-signals">
                            <div class="no-signals-icon">📡</div>
                            Scanning markets... signals will appear here
                        </td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- HISTORY TAB -->
        <div id="tab-history" class="tab-content">
            <div class="history-controls">
                <input class="search-input" type="text" id="historySearch" placeholder="Search symbol..." oninput="filterHistory()">
                <button class="sort-btn" onclick="sortHistory('profitability', event)">💰 Profit</button>
                <button class="sort-btn" onclick="sortHistory('confidence', event)">🎯 Confidence</button>
                <button class="sort-btn" onclick="sortHistory('date', event)">📅 Date</button>
                <button class="refresh-btn" onclick="loadHistory()">↻ Refresh</button>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Type</th>
                            <th>Signal</th>
                            <th>Price</th>
                            <th>Profit</th>
                            <th>Safety</th>
                            <th>Risk</th>
                            <th>Conf</th>
                            <th>Entry</th>
                            <th>Stop Loss</th>
                            <th>Take Profit</th>
                            <th>Reason</th>
                        </tr>
                    </thead>
                    <tbody id="historyBody">
                        <tr><td colspan="14" class="no-signals">
                            <div class="no-signals-icon">📂</div>
                            No history yet
                        </td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- HOW TO TRADE TAB -->
        <div id="tab-guide" class="tab-content">

            <div class="guide-full">
                <h2>What This Bot Does</h2>
                <p>Trade Grid Analysis scans 37 assets every 5 minutes across crypto, stocks, commodities and indices. Each potential trade is scored by AI on profitability, safety, risk and confidence. Only high quality signals (8/10 for crypto, 7/10 for stocks) are shown here and sent to your Telegram.</p>
                <p>The bot tells you when to trade and gives you entry, stop loss and take profit levels. You then manually place that trade on your platform.</p>
            </div>

            <div class="platform-row">
                <div class="platform-card">
                    <div class="badge badge-green">Recommended</div>
                    <h4>Plus500</h4>
                    <p>Best for beginners. Supports crypto, stocks, forex, gold, oil and indices in one place. Simple interface, no commissions, uses CFDs. Great for small accounts like R500.</p>
                </div>
                <div class="platform-card">
                    <div class="badge badge-yellow">Crypto Only</div>
                    <h4>Binance</h4>
                    <p>Best for crypto trading only. Very low fees, huge selection. Good if you want to focus purely on the crypto signals this bot generates.</p>
                </div>
                <div class="platform-card">
                    <div class="badge badge-yellow">Practice First</div>
                    <h4>Alpaca Paper Trading</h4>
                    <p>Free paper trading with real market data. Practice your strategy with fake money before risking your R500. Highly recommended for beginners.</p>
                </div>
            </div>

            <div class="guide-full">
                <h2>How To Read A Signal</h2>
                <div class="guide-grid" style="margin-top:12px">
                    <div>
                        <ul style="list-style:none;padding:0">
                            <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;color:var(--muted2)"><strong style="color:var(--green)">BUY 🟢</strong> — Price trending up, consider buying</li>
                            <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;color:var(--muted2)"><strong style="color:var(--red)">SELL 🔴</strong> — Price trending down, consider selling</li>
                            <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;color:var(--muted2)"><strong style="color:var(--text)">Entry</strong> — The price to open your trade at</li>
                            <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;color:var(--muted2)"><strong style="color:var(--text)">Stop Loss</strong> — Close here to limit your loss</li>
                            <li style="padding:6px 0;font-size:13px;color:var(--muted2)"><strong style="color:var(--text)">Take Profit</strong> — Close here to lock in profit</li>
                        </ul>
                    </div>
                    <div>
                        <ul style="list-style:none;padding:0">
                            <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;color:var(--muted2)"><strong style="color:var(--text)">Profitability</strong> — Profit potential (7+ is good)</li>
                            <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;color:var(--muted2)"><strong style="color:var(--text)">Safety</strong> — How safe the trade is (7+ is good)</li>
                            <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;color:var(--muted2)"><strong style="color:var(--text)">Risk</strong> — How risky it is (lower is better)</li>
                            <li style="padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;color:var(--muted2)"><strong style="color:var(--text)">Confidence</strong> — AI confidence in the signal</li>
                            <li style="padding:6px 0;font-size:13px;color:var(--muted2)"><strong style="color:var(--text)">MACD</strong> — Confirms trend direction</li>
                        </ul>
                    </div>
                </div>
            </div>

            <div class="guide-full">
                <h2>Step By Step — How To Place A Trade</h2>
                <div class="step-list" style="margin-top:12px">
                    <div class="step"><div class="step-num">1</div><div class="step-text"><strong>Wait for a high quality signal</strong> — Only trade signals scoring 7+ on Confidence and Profitability. The bot already filters these for you.</div></div>
                    <div class="step"><div class="step-num">2</div><div class="step-text"><strong>Open Plus500 and search the asset</strong> — Search the symbol e.g. BTC/USD, AAPL, Gold. Click to open the trading screen.</div></div>
                    <div class="step"><div class="step-num">3</div><div class="step-text"><strong>Set your trade size</strong> — With R500 never risk more than R25–R50 per trade (5–10%). This protects your account.</div></div>
                    <div class="step"><div class="step-num">4</div><div class="step-text"><strong>Set your Stop Loss</strong> — Always set the stop loss at the price the bot suggests. This is the most important step.</div></div>
                    <div class="step"><div class="step-num">5</div><div class="step-text"><strong>Set your Take Profit</strong> — Set take profit at the bot's suggested price to lock in gains automatically.</div></div>
                    <div class="step"><div class="step-num">6</div><div class="step-text"><strong>Open the trade and wait</strong> — Let it run. Don't panic if price moves slightly against you. Trust your stop loss.</div></div>
                </div>
            </div>

            <div class="guide-full">
                <h2>Golden Rules</h2>
                <div class="rules-grid">
                    <div class="rule"><strong>Max 10% per trade</strong>Never risk more than 10% of your account on one trade</div>
                    <div class="rule"><strong>Always set a stop loss</strong>No exceptions. Ever. This protects your account.</div>
                    <div class="rule"><strong>7/10 minimum</strong>Only trade signals scoring 7+ on confidence and profitability</div>
                    <div class="rule"><strong>Never chase losses</strong>If a trade goes wrong, don't immediately open another to recover</div>
                    <div class="rule"><strong>Keep a trading journal</strong>Note every trade, why you took it and what happened</div>
                    <div class="rule"><strong>Quality over quantity</strong>2 great trades a week beats 20 bad ones</div>
                </div>
            </div>

        </div>
    </div>

    <script>
        let signals = [];
        let historyData = [];
        let sortKey = 'profitability';
        let sortAsc = false;
        let historySortKey = 'date';
        let historySortAsc = false;

        function switchTab(tab, e) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.getElementById('tab-' + tab).classList.add('active');
            if (e) e.target.classList.add('active');
            if (tab === 'history') loadHistory();
        }

        function getScoreClass(score) {
            score = parseInt(score);
            if (score >= 7) return 'score-high';
            if (score >= 4) return 'score-mid';
            return 'score-low';
        }

        function scoreHTML(score) {
            return `<span class="score-pill ${getScoreClass(score)}">${score}/10</span>`;
        }

        function sortTable(key, e) {
            if (sortKey === key) sortAsc = !sortAsc;
            else { sortKey = key; sortAsc = false; }
            document.querySelectorAll('#tab-signals .sort-btn').forEach(b => b.classList.remove('active'));
            if (e && e.target.classList.contains('sort-btn')) e.target.classList.add('active');
            renderSignals();
        }

        function sortHistory(key, e) {
            if (historySortKey === key) historySortAsc = !historySortAsc;
            else { historySortKey = key; historySortAsc = false; }
            renderHistory();
        }

        function renderSignals() {
            const sorted = [...signals].sort((a, b) => {
                let av = a[sortKey], bv = b[sortKey];
                if (typeof av === 'string') av = av.toLowerCase();
                if (typeof bv === 'string') bv = bv.toLowerCase();
                return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
            });

            const tbody = document.getElementById('tableBody');
            if (!sorted.length) {
                tbody.innerHTML = `<tr><td colspan="14" class="no-signals"><div class="no-signals-icon">📡</div>No high quality signals yet — scanning every 5 mins</td></tr>`;
                return;
            }

            tbody.innerHTML = sorted.map(s => `
                <tr>
                    <td><div class="symbol">${s.symbol}</div><div class="asset-type">${s.asset_type}</div></td>
                    <td class="${s.signal.includes('BUY') ? 'buy' : 'sell'}">${s.signal}</td>
                    <td>${s.price}</td>
                    <td>${scoreHTML(s.profitability)}</td>
                    <td>${scoreHTML(s.safety)}</td>
                    <td>${scoreHTML(s.risk)}</td>
                    <td>${scoreHTML(s.confidence)}</td>
                    <td>${s.entry}</td>
                    <td>${s.stop_loss}</td>
                    <td>${s.take_profit}</td>
                    <td style="font-size:11px;color:var(--muted2)">${s.macd}</td>
                    <td style="font-size:11px">${s.rsi}</td>
                    <td><div class="reason-text">${s.reason}</div></td>
                    <td style="color:var(--muted2);font-size:12px">${s.timestamp}</td>
                </tr>
            `).join('');
        }

        function filterHistory() {
            renderHistory();
        }

        function renderHistory() {
            const search = document.getElementById('historySearch').value.toLowerCase();
            let filtered = historyData.filter(h => h.symbol && h.symbol.toLowerCase().includes(search));
            filtered.sort((a, b) => {
                let av = a[historySortKey], bv = b[historySortKey];
                if (typeof av === 'string') av = av.toLowerCase();
                if (typeof bv === 'string') bv = bv.toLowerCase();
                return historySortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
            });

            const tbody = document.getElementById('historyBody');
            if (!filtered.length) {
                tbody.innerHTML = `<tr><td colspan="14" class="no-signals"><div class="no-signals-icon">📂</div>No history yet</td></tr>`;
                return;
            }

            tbody.innerHTML = filtered.map(s => `
                <tr>
                    <td style="font-size:12px;color:var(--muted2)">${s.date}</td>
                    <td style="font-size:12px;color:var(--muted2)">${s.time}</td>
                    <td><div class="symbol">${s.symbol}</div></td>
                    <td><div class="asset-type">${s.asset_type}</div></td>
                    <td class="${s.signal && s.signal.includes('BUY') ? 'buy' : 'sell'}">${s.signal}</td>
                    <td>${s.price}</td>
                    <td>${scoreHTML(s.profitability)}</td>
                    <td>${scoreHTML(s.safety)}</td>
                    <td>${scoreHTML(s.risk)}</td>
                    <td>${scoreHTML(s.confidence)}</td>
                    <td>${s.entry}</td>
                    <td>${s.stop_loss}</td>
                    <td>${s.take_profit}</td>
                    <td><div class="reason-text">${s.reason}</div></td>
                </tr>
            `).join('');
        }

        function loadSignals() {
            fetch('/signals')
                .then(r => r.json())
                .then(data => {
                    signals = data.signals || [];
                    document.getElementById('statCurrent').textContent = signals.length;
                    document.getElementById('statTotal').textContent = data.total || 0;
                    document.getElementById('lastUpdate').textContent = `Updated ${new Date().toLocaleTimeString()}`;
                    renderSignals();
                });
        }

        function loadHistory() {
            fetch('/history')
                .then(r => r.json())
                .then(data => {
                    historyData = data;
                    renderHistory();
                });
        }

        loadSignals();
        setInterval(loadSignals, 30000);
    </script>
</body>
</html>
"""

# ---- WEB SERVER ----
class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == '/signals':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            payload = json.dumps({'signals': all_signals, 'total': total_signals_found})
            self.wfile.write(payload.encode())
        elif self.path == '/history':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(load_history()).encode())

    def log_message(self, format, *args):
        pass

def start_dashboard():
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    server.serve_forever()

# ---- MAIN SCANNER ----
def run_scanner():
    global all_signals, last_scan_time
    all_signals = []
    last_scan_time = datetime.now().strftime('%d %b %Y %H:%M')
    print(f"\n🔍 Running full market scan — {last_scan_time}")

    print("\n🪙 Scanning Crypto...")
    for pair in crypto_pairs:
        try:
            df = get_crypto_ohlcv(pair)
            check_signal(pair, df, 'crypto')
        except Exception as e:
            print(f"Error {pair}: {e}")

    print("\n📈 Scanning Stocks, Commodities & Indices...")
    for symbol in stock_pairs:
        try:
            df = get_alpaca_ohlcv(symbol)
            check_signal(symbol, df, 'stock')
        except Exception as e:
            print(f"Error {symbol}: {e}")

    print("\n💱 Scanning Forex...")
    for pair in forex_pairs:
        try:
            df = get_forex_ohlcv(pair)
            check_signal(pair, df, 'forex')
        except Exception as e:
            print(f"Error {pair}: {e}")

    if all_signals:
        send_telegram_summary(all_signals)
        print(f"\n✅ Scan complete — {len(all_signals)} high quality signals found!")
    else:
        print("\n✅ Scan complete — No high quality signals this round")

# ---- START ----
init_csv()

# Start dashboard
dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
dashboard_thread.start()
print("🌐 Trade Grid Analysis running at http://localhost:8080")

# Start Telegram command listener
telegram_thread = threading.Thread(target=handle_telegram_commands, daemon=True)
telegram_thread.start()
print("📱 Telegram command listener started (/status /topsignals /help)")

# Run scanner immediately
run_scanner()

# Schedule every 2 mins
schedule.every(2).minutes.do(run_scanner)

print("\n⏰ Scanning every 2 mins. Press Ctrl+C to stop.")

while True:
    schedule.run_pending()
    time.sleep(30)