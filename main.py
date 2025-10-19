import os
import time
import math
import logging
from datetime import datetime, timezone, timedelta

import ccxt
import pandas as pd
import pandas_ta as ta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

# -------- Config --------
EXCHANGE_ID = os.getenv("EXCHANGE", "kraken")
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USD").upper()
DRY_RUN = os.getenv("DRY_RUN", "1") == "1"
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "900"))
MAX_SYMBOLS_PER_RUN = int(os.getenv("MAX_SYMBOLS_PER_RUN", "20"))
ENABLE_LISTING_PRICE = os.getenv("ENABLE_LISTING_PRICE", "0") == "1"
MIN_USD_ORDER = float(os.getenv("MIN_USD_ORDER", "5"))

API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("kraken-bot")

STABLE_LIKE = {
    "USD", "USDT", "USDC", "DAI", "EUR", "GBP", "TUSD", "FDUSD", "PYUSD", "BUSD", "GUSD", "USDP"
}

def is_stablecoin(symbol: str) -> bool:
    base = symbol.split("/")[0]
    return base.upper() in STABLE_LIKE

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def fetch_ohlcv_safe(exchange, symbol, timeframe="15m", limit=300):
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df["sma60"] = ta.sma(df["close"], length=60)
    df["sma240"] = ta.sma(df["close"], length=240)
    df["rsi14"] = ta.rsi(df["close"], length=14)
    return df

def had_trades_last_24h(df_15m: pd.DataFrame) -> bool:
    vol_24h = df_15m["volume"].tail(96).sum()
    return vol_24h and vol_24h > 0

def make_exchange() -> ccxt.Exchange:
    klass = getattr(ccxt, EXCHANGE_ID)
    return klass({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })

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
        if is_stablecoin(m["symbol"]):
            continue
        symbols.append(m["symbol"])
    return sorted(set(symbols))

def place_market_buy(exchange, market, notional):
    ticker = exchange.fetch_ticker(market["symbol"])
    price = ticker["last"]
    amount = notional / price if price else 0
    amount = exchange.amount_to_precision(market["symbol"], amount)

    if DRY_RUN:
        log.info(f"[DRY_RUN] Would buy {market['symbol']} worth {notional:.2f}")
        return {"id": "dry-run"}

    order = exchange.create_order(symbol=market["symbol"], type="market", side="buy", amount=amount)
    log.info(f"Order placed: {order}")
    return order

def evaluate_symbol(exchange, market):
    symbol = market["symbol"]
    try:
        ohlcv = fetch_ohlcv_safe(exchange, symbol, timeframe="15m", limit=300)
        if not ohlcv or len(ohlcv) < 240:
            return {"symbol": symbol, "eligible": False, "reason": "insufficient data"}

        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        ind = compute_indicators(df)
        row = ind.iloc[-1]
        sma60, sma240, rsi14, last_close = row["sma60"], row["sma240"], row["rsi14"], row["close"]

        if rsi14 is None or rsi14 == 0 or pd.isna(rsi14):
            return {"symbol": symbol, "eligible": False, "reason": "invalid RSI"}

        if rsi14 >= 30:
            return {"symbol": symbol, "eligible": False, "reason": f"RSI14={rsi14:.2f} >= 30"}

        if sma60 >= sma240:
            return {"symbol": symbol, "eligible": False, "reason": "SMA60 >= SMA240"}

        if not had_trades_last_24h(df):
            return {"symbol": symbol, "eligible": False, "reason": "no trades 24h"}

        return {"symbol": symbol, "eligible": True, "rsi14": rsi14, "sma60": sma60, "sma240": sma240, "last": last_close}
    except Exception as e:
        return {"symbol": symbol, "eligible": False, "reason": str(e)}

def main_loop():
    exchange = make_exchange()
    exchange.load_markets()
    while True:
        try:
            symbols = select_markets(exchange, QUOTE_ASSET)
            log.info(f"Checking {min(len(symbols), MAX_SYMBOLS_PER_RUN)} / {len(symbols)} markets")

            quote_bal = get_quote_balance(exchange, QUOTE_ASSET)
            log.info(f"Available {QUOTE_ASSET}: {quote_bal:.2f}")
            notional = quote_bal * 0.05

            to_check = symbols[:MAX_SYMBOLS_PER_RUN]
            for sym in to_check:
                market = exchange.market(sym)
                res = evaluate_symbol(exchange, market)
                if res["eligible"]:
                    log.info(f"✅ {sym} meets criteria: RSI={res['rsi14']:.2f}")
                    if notional >= MIN_USD_ORDER:
                        place_market_buy(exchange, market, notional)
                else:
                    log.debug(f"❌ {sym}: {res['reason']}")
        except Exception as e:
            log.error(f"Main loop error: {e}")

        log.info(f"Sleeping {INTERVAL_SEC}s…")
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main_loop()
