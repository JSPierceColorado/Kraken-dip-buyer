import os
import time
import math
import logging
from datetime import datetime, timezone, timedelta

import ccxt
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

# -------- Config --------
EXCHANGE_ID = os.getenv("EXCHANGE", "kraken")
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USD").upper()
DRY_RUN = os.getenv("DRY_RUN", "1") == "1"
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "900"))          # 15m default
MAX_SYMBOLS_PER_RUN = int(os.getenv("MAX_SYMBOLS_PER_RUN", "20"))
ENABLE_LISTING_PRICE = os.getenv("ENABLE_LISTING_PRICE", "0") == "1"
MIN_USD_ORDER = float(os.getenv("MIN_USD_ORDER", "5"))

API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

# -------- Logging --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("kraken-bot")

# -------- Helpers --------
STABLE_LIKE = {
    "USD", "USDT", "USDC", "DAI", "EUR", "GBP", "TUSD", "FDUSD", "PYUSD", "BUSD", "GUSD", "USDP"
}

def is_stablecoin(symbol: str) -> bool:
    base = symbol.split("/")[0]
    return base.upper() in STABLE_LIKE

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Wilder's RSI implementation using EMA with alpha=1/length.
    Returns a Series aligned to `close`.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (EMA with alpha = 1/length)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="backfill")

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    df columns: [ts, open, high, low, close, volume]
    Adds sma60, sma240, rsi14 columns.
    """
    df = df.copy()
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df["sma60"] = df["close"].rolling(window=60, min_periods=60).mean()
    df["sma240"] = df["close"].rolling(window=240, min_periods=240).mean()
    df["rsi14"] = rsi_wilder(df["close"], length=14)
    return df

def had_trades_last_24h(df_15m: pd.DataFrame) -> bool:
    # last 96 * 15m = 24h
    vol_24h = df_15m["volume"].tail(96).sum()
    return bool(vol_24h and vol_24h > 0)

# -------- Exchange --------
def make_exchange() -> ccxt.Exchange:
    klass = getattr(ccxt, EXCHANGE_ID)
    return klass({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def fetch_ohlcv(exchange, symbol, timeframe="15m", limit=300):
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def fetch_ohlcv_since(exchange, symbol, timeframe="1d", since=None, limit=2000):
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

def get_quote_balance(exchange, quote_asset: str) -> float:
    bal = exchange.fetch_balance()
    free = bal.get("free", {}).get(quote_asset, 0.0)
    if not free:
        free = bal.get("total", {}).get(quote_asset, 0.0)
    return float(free or 0.0)

def select_markets(exchange, quote_asset: str):
    markets = exchange.load_markets()
    symbols = []
    for m in markets.values():
        if not m.get("active", True):
            continue
        if not m.get("spot", False):
            continue
        if m.get("quote") != quote_asset:
            continue
        sym = m["symbol"]
        if is_stablecoin(sym):
            continue
        symbols.append(sym)
    return sorted(set(symbols))

def place_market_buy(exchange, market, notional):
    ticker = exchange.fetch_ticker(market["symbol"])
    price = ticker.get("last")
    if not price or price <= 0:
        raise ValueError(f"No valid last price for {market['symbol']}")
    amount = notional / price
    amount = exchange.amount_to_precision(market["symbol"], amount)

    if DRY_RUN:
        log.info(f"[DRY_RUN] Would place market BUY {market['symbol']} amount={amount} notional≈{notional:.2f}")
        return {"id": "dry-run"}

    order = exchange.create_order(symbol=market["symbol"], type="market", side="buy", amount=amount)
    log.info(f"Placed order: {order}")
    return order

def price_above_listing(exchange, symbol, current_price) -> bool:
    # Optional guard: current >= listing_open * 1.05
    try:
        since = int((datetime.now(timezone.utc) - timedelta(days=3650)).timestamp() * 1000)  # ~10y backstop
        candles = fetch_ohlcv_since(exchange, symbol, timeframe="1d", since=since, limit=2000)
        if candles:
            earliest_open = candles[0][1]
            if earliest_open and current_price is not None:
                return current_price >= earliest_open * 1.05
        # If nothing returned, treat as "unknown"
        return not ENABLE_LISTING_PRICE  # pass if disabled; block if enabled
    except Exception as e:
        log.debug(f"Listing price check failed for {symbol}: {e}")
        return not ENABLE_LISTING_PRICE

def evaluate_symbol(exchange, market) -> dict:
    symbol = market["symbol"]
    try:
        ohlcv = fetch_ohlcv(exchange, symbol, timeframe="15m", limit=300)
        if not ohlcv or len(ohlcv) < 240:
            return {"symbol": symbol, "eligible": False, "reason": "insufficient candles"}

        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        ind = compute_indicators(df)
        row = ind.iloc[-1]
        sma60 = row["sma60"]
        sma240 = row["sma240"]
        rsi14 = row["rsi14"]
        last_close = row["close"]

        # RSI14 not zero / valid
        if pd.isna(rsi14) or not (rsi14 > 0):
            return {"symbol": symbol, "eligible": False, "reason": "invalid RSI (0 or NaN)"}

        # RSI14 < 30
        if float(rsi14) >= 30:
            return {"symbol": symbol, "eligible": False, "reason": f"RSI14={float(rsi14):.2f} >= 30"}

        # SMA60 < SMA240
        if pd.isna(sma60) or pd.isna(sma240) or not (float(sma60) < float(sma240)):
            return {"symbol": symbol, "eligible": False, "reason": f"SMA60 !< SMA240"}

        # Trades in last 24h
        if not had_trades_last_24h(df):
            return {"symbol": symbol, "eligible": False, "reason": "no trades last 24h (zero volume)"}

        # Optional: listing price filter
        if ENABLE_LISTING_PRICE:
            if not price_above_listing(exchange, symbol, float(last_close)):
                return {"symbol": symbol, "eligible": False, "reason": "below listing price +5%"}

        return {
            "symbol": symbol,
            "eligible": True,
            "sma60": float(sma60),
            "sma240": float(sma240),
            "rsi14": float(rsi14),
            "last": float(last_close),
        }
    except Exception as e:
        return {"symbol": symbol, "eligible": False, "reason": f"error: {e}"}

def main_loop():
    exchange = make_exchange()
    exchange.load_markets()

    while True:
        try:
            symbols = select_markets(exchange, QUOTE_ASSET)
            log.info(f"Evaluating up to {min(len(symbols), MAX_SYMBOLS_PER_RUN)} of {len(symbols)} symbols (quote={QUOTE_ASSET})")

            quote_bal = get_quote_balance(exchange, QUOTE_ASSET)
            log.info(f"Quote balance {QUOTE_ASSET}: {quote_bal:.8f}")
            notional = quote_bal * 0.05
            if notional < MIN_USD_ORDER:
                log.info(f"Notional {notional:.2f} < MIN_USD_ORDER {MIN_USD_ORDER:.2f} — will skip orders this run.")

            to_check = symbols[:MAX_SYMBOLS_PER_RUN]
            eligible_count = 0
            for sym in to_check:
                market = exchange.market(sym)
                res = evaluate_symbol(exchange, market)
                if res.get("eligible"):
                    eligible_count += 1
                    log.info(f"ELIGIBLE: {sym} | RSI14={res['rsi14']:.2f} SMA60={res['sma60']:.6f} < SMA240={res['sma240']:.6f}")
                    if notional >= MIN_USD_ORDER:
                        try:
                            place_market_buy(exchange, market, notional=notional)
                        except Exception as e:
                            log.error(f"Order failed for {sym}: {e}")
                else:
                    log.debug(f"INELIGIBLE: {sym} — {res.get('reason')}")

            log.info(f"Run summary: {eligible_count} eligible (checked {len(to_check)}).")
        except Exception as e:
            log.error(f"Top-level loop error: {e}")

        log.info(f"Sleeping {INTERVAL_SEC}s…")
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main_loop()
