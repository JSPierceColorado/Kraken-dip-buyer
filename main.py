import os
import time
import math
import logging
import random
from datetime import datetime, timezone, timedelta
import traceback
from typing import Tuple, Dict, List, Optional

import ccxt
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from ccxt.base.errors import DDoSProtection, ExchangeNotAvailable, RateLimitExceeded

load_dotenv()

# -------- Config (with verbose logging knobs) --------
EXCHANGE_ID = os.getenv("EXCHANGE", "kraken")
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USD").upper()
DRY_RUN = os.getenv("DRY_RUN", "1") == "1"
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "900"))          # 15m default
MAX_SYMBOLS_PER_RUN = int(os.getenv("MAX_SYMBOLS_PER_RUN", "20"))
ENABLE_LISTING_PRICE = os.getenv("ENABLE_LISTING_PRICE", "0") == "1"
MIN_USD_ORDER = float(os.getenv("MIN_USD_ORDER", "5"))
DRY_RUN_NOTIONAL = float(os.getenv("DRY_RUN_NOTIONAL", "100"))
DRY_RUN_USE_BALANCE = os.getenv("DRY_RUN_USE_BALANCE", "1") == "1"  # whether DRY_RUN can call fetch_balance for weights

# NEW: per-asset portfolio weight cap
MAX_WEIGHT_PCT = float(os.getenv("MAX_WEIGHT_PCT", "0.10"))  # 10% cap per asset

# Logging controls
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
VERBOSE_SYMBOL_LOG = os.getenv("VERBOSE_SYMBOL_LOG", "1") == "1"

API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

# -------- Logging --------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
log = logging.getLogger("kraken-bot")
log.info("=== Bot boot ===")
log.info(
    f"Config: EXCHANGE={EXCHANGE_ID} QUOTE_ASSET={QUOTE_ASSET} "
    f"DRY_RUN={DRY_RUN} INTERVAL_SEC={INTERVAL_SEC} MAX_SYMBOLS_PER_RUN={MAX_SYMBOLS_PER_RUN} "
    f"ENABLE_LISTING_PRICE={ENABLE_LISTING_PRICE} MIN_USD_ORDER={MIN_USD_ORDER} "
    f"DRY_RUN_NOTIONAL={DRY_RUN_NOTIONAL} DRY_RUN_USE_BALANCE={DRY_RUN_USE_BALANCE} "
    f"MAX_WEIGHT_PCT={MAX_WEIGHT_PCT} LOG_LEVEL={LOG_LEVEL} VERBOSE_SYMBOL_LOG={VERBOSE_SYMBOL_LOG}"
)

# -------- Helpers --------
STABLE_LIKE = {
    "USD", "USDT", "USDC", "DAI", "EUR", "GBP", "TUSD", "FDUSD", "PYUSD", "BUSD", "GUSD", "USDP"
}
BTC_BASES = {"BTC", "XBT"}  # accommodate exchanges that label BTC as XBT (e.g., Kraken)

def is_stablecoin(symbol: str) -> bool:
    base = symbol.split("/")[0]
    return base.upper() in STABLE_LIKE

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Wilder's RSI using EMA with alpha=1/length.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.bfill()
    return rsi

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

def had_trades_last_24h(df_15m: pd.DataFrame) -> Tuple[bool, float]:
    # last 96 * 15m = 24h
    vol_24h = float(df_15m["volume"].tail(96).sum())
    return bool(vol_24h and vol_24h > 0), vol_24h

# ---- Market-wide BTC MA condition (15m SMA180 < SMA720) ----
def find_btc_symbol(exchange, quote_asset: str) -> Optional[str]:
    """
    Try to find a BTC/XBT market for the given quote (e.g., BTC/USD, XBT/USD).
    Returns unified symbol string or None.
    """
    markets = exchange.load_markets()
    candidates = []
    for m in markets.values():
        if not m.get("active", True):
            continue
        if not m.get("spot", False):
            continue
        if m.get("quote") != quote_asset:
            continue
        base = (m.get("base") or "").upper()
        if base in BTC_BASES:
            candidates.append(m["symbol"])
    for sym in candidates:
        if sym.upper().startswith("BTC/"):
            return sym
    return candidates[0] if candidates else None

def check_btc_ma_condition(exchange, quote_asset: str, timeframe: str = "15m") -> Tuple[bool, str, dict]:
    """
    Returns (ok, note, metrics) where ok is True iff SMA180 < SMA720.
    metrics includes last, sma180, sma720, symbol.
    If data insufficient/unavailable, returns ok=False with a note.
    """
    btc_sym = find_btc_symbol(exchange, quote_asset)
    if not btc_sym:
        return False, f"No BTC market found for quote={quote_asset}", {}

    try:
        ohlcv = fetch_ohlcv(exchange, btc_sym, timeframe=timeframe, limit=1000)
        if not ohlcv or len(ohlcv) < 720:
            return False, f"{btc_sym} insufficient candles for SMA720 (have {len(ohlcv) if ohlcv else 0})", {"symbol": btc_sym}

        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        close = df["close"]
        sma180 = close.rolling(window=180, min_periods=180).mean().iloc[-1]
        sma720 = close.rolling(window=720, min_periods=720).mean().iloc[-1]
        last = float(close.iloc[-1])

        if pd.isna(sma180) or pd.isna(sma720):
            return False, f"{btc_sym} SMA values NaN", {"symbol": btc_sym}

        ok = float(sma180) < float(sma720)
        note = (f"{btc_sym} 15m SMA180={float(sma180):.6f} < SMA720={float(sma720):.6f} ⇒ {ok} | last={last:.10g}")
        return ok, note, {"symbol": btc_sym, "sma180": float(sma180), "sma720": float(sma720), "last": last}
    except Exception as e:
        return False, f"{btc_sym} MA check error: {e}", {"symbol": btc_sym}

# -------- Exchange --------
def make_exchange() -> ccxt.Exchange:
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })
    ex.verbose = False
    return ex

def set_verbose_public(exchange: ccxt.Exchange, enabled: bool):
    """Enable ccxt.verbose only for PUBLIC endpoints (avoid leaking API-Key on private)."""
    try:
        exchange.verbose = bool(enabled)
    except Exception:
        pass

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def fetch_ohlcv(exchange, symbol, timeframe="15m", limit=300):
    set_verbose_public(exchange, LOG_LEVEL == "DEBUG")
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    finally:
        set_verbose_public(exchange, False)
        time.sleep(0.05 + random.random() * 0.05)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def fetch_ohlcv_since(exchange, symbol, timeframe="1d", since=None, limit=2000):
    set_verbose_public(exchange, LOG_LEVEL == "DEBUG")
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    finally:
        set_verbose_public(exchange, False)
        time.sleep(0.05 + random.random() * 0.05)

def safe_get_quote_balance(exchange, quote_asset: str) -> float:
    """
    Fetches quote balance using Kraken private API with polite handling.
    Returns 0.0 on failure so caller can skip orders.
    """
    set_verbose_public(exchange, False)
    try:
        bal = exchange.fetch_balance()
        free = bal.get("free", {}).get(quote_asset, 0.0)
        if not free:
            free = bal.get("total", {}).get(quote_asset, 0.0)
        return float(free or 0.0)
    except DDoSProtection as e:
        log.warning(f"Balance fetch lockout/rate-limit: {e}. Backing off 180s; skipping orders this run.")
        time.sleep(180)
        return 0.0
    except (RateLimitExceeded, ExchangeNotAvailable) as e:
        log.warning(f"Balance fetch throttled/unavailable: {e}. Backing off 60s; skipping orders this run.")
        time.sleep(60)
        return 0.0
    except Exception as e:
        log.warning(f"Balance fetch failed: {e}. Skipping orders this run.")
        return 0.0

def select_markets(exchange, quote_asset: str):
    markets = exchange.load_markets()
    total, inactive, non_spot, wrong_quote, stable_filtered = 0, 0, 0, 0, 0
    symbols = []
    for m in markets.values():
        total += 1
        if not m.get("active", True):
            inactive += 1
            continue
        if not m.get("spot", False):
            non_spot += 1
            continue
        if m.get("quote") != quote_asset:
            wrong_quote += 1
            continue
        sym = m["symbol"]
        if is_stablecoin(sym):
            stable_filtered += 1
            continue
        symbols.append(sym)
    symbols = sorted(set(symbols))
    log.info(
        f"Market scan: total={total}, spot&active&quote={len(symbols)} "
        f"(inactive={inactive}, non_spot={non_spot}, wrong_quote={wrong_quote}, stable_filtered={stable_filtered})"
    )
    return symbols

def place_market_buy(exchange, market, notional):
    set_verbose_public(exchange, False)
    ticker = exchange.fetch_ticker(market["symbol"])
    price = ticker.get("last")
    if not price or price <= 0:
        raise ValueError(f"No valid last price for {market['symbol']}")
    amount = notional / price
    amount = exchange.amount_to_precision(market["symbol"], amount)

    if DRY_RUN:
        log.info(
            f"[DRY_RUN] Would place MARKET BUY {market['symbol']} amount={amount} "
            f"price≈{price} notional≈{notional:.2f}"
        )
        return {"id": "dry-run", "symbol": market["symbol"], "amount": amount, "price": price, "notional": notional}

    order = exchange.create_order(symbol=market["symbol"], type="market", side="buy", amount=amount)
    log.info(f"Order placed: {order}")
    return order

def price_above_listing(exchange, symbol, current_price) -> Tuple[bool, str]:
    # Optional guard: current >= listing_open * 1.05
    try:
        since = int((datetime.now(timezone.utc) - timedelta(days=3650)).timestamp() * 1000)
        candles = fetch_ohlcv_since(exchange, symbol, timeframe="1d", since=since, limit=2000)
        if candles:
            earliest_open = candles[0][1]
            if earliest_open and current_price is not None:
                ok = current_price >= earliest_open * 1.05
                reason = f"current={current_price:.10g} listing_open={earliest_open:.10g} require>=5% ⇒ {ok}"
                return ok, reason
        ok = not ENABLE_LISTING_PRICE
        reason = "no daily candles; passing" if ok else "no daily candles; failing (enabled)"
        return ok, reason
    except Exception as e:
        ok = not ENABLE_LISTING_PRICE
        return ok, f"listing check error: {e}; {'passing' if ok else 'failing'}"

def evaluate_symbol(exchange, market) -> dict:
    symbol = market["symbol"]
    try:
        ohlcv = fetch_ohlcv(exchange, symbol, timeframe="15m", limit=300)
        if not ohlcv or len(ohlcv) < 240:
            return {"symbol": symbol, "eligible": False, "reason": "insufficient candles (<240)"}

        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        ind = compute_indicators(df)
        row = ind.iloc[-1]
        sma60 = row["sma60"]
        sma240 = row["sma240"]
        rsi14 = row["rsi14"]
        last_close = row["close"]

        has_24h, vol_24h = had_trades_last_24h(df)

        if VERBOSE_SYMBOL_LOG:
            log.debug(
                f"[{symbol}] last={float(last_close):.10g} rsi14={float(rsi14) if pd.notna(rsi14) else rsi14} "
                f"sma60={float(sma60) if pd.notna(sma60) else sma60} "
                f"sma240={float(sma240) if pd.notna(sma240) else sma240} "
                f"vol24h={vol_24h:.10g}"
            )

        if pd.isna(rsi14) or not (rsi14 > 0):
            return {"symbol": symbol, "eligible": False, "reason": "invalid RSI (0 or NaN or <=0)",
                    "rsi14": None if pd.isna(rsi14) else float(rsi14)}

        if float(rsi14) >= 30:
            return {"symbol": symbol, "eligible": False, "reason": f"RSI14={float(rsi14):.2f} >= 30",
                    "rsi14": float(rsi14)}

        if pd.isna(sma60) or pd.isna(sma240) or not (float(sma60) < float(sma240)):
            return {"symbol": symbol, "eligible": False, "reason": f"SMA60 !< SMA240",
                    "sma60": None if pd.isna(sma60) else float(sma60),
                    "sma240": None if pd.isna(sma240) else float(sma240)}

        if not has_24h:
            return {"symbol": symbol, "eligible": False, "reason": "no trades last 24h (zero volume)",
                    "vol24h": vol_24h}

        if ENABLE_LISTING_PRICE:
            ok, lp_reason = price_above_listing(exchange, symbol, float(last_close))
            if VERBOSE_SYMBOL_LOG:
                log.debug(f"[{symbol}] listing-price check: {lp_reason}")
            if not ok:
                return {"symbol": symbol, "eligible": False, "reason": "below listing price +5%",
                        "listing_note": lp_reason}

        return {
            "symbol": symbol,
            "eligible": True,
            "sma60": float(sma60),
            "sma240": float(sma240),
            "rsi14": float(rsi14),
            "last": float(last_close),
            "vol24h": vol_24h,
        }
    except Exception as e:
        log.exception(f"[{symbol}] evaluation error")
        return {"symbol": symbol, "eligible": False, "reason": f"error: {e}"}

def print_eligible_table(eligible_list):
    if not eligible_list:
        log.info("Eligible table: (none)")
        return
    headers = ["SYMBOL", "RSI14", "SMA60", "SMA240", "LAST", "VOL24H"]
    row_fmt = "{:<12} {:>8} {:>12} {:>12} {:>12} {:>12}"
    log.info("Eligible table:")
    log.info(row_fmt.format(*headers))
    for r in eligible_list:
        log.info(row_fmt.format(
            r["symbol"],
            f"{r['rsi14']:.2f}",
            f"{r['sma60']:.6f}",
            f"{r['sma240']:.6f}",
            f"{r['last']:.8f}",
            f"{r.get('vol24h', 0):.4f}",
        ))

# -------- Portfolio weights + cap logic --------
def get_portfolio_weights(exchange: ccxt.Exchange, quote_asset: str) -> Tuple[Dict[str, float], float]:
    """
    Returns (weights_by_symbol, total_equity_in_quote).
    weights_by_symbol keys are unified symbols like 'ETH/USD'.
    Only assets that have a BASE/QUOTE spot market are counted as positions (cash stays as cash).
    """
    set_verbose_public(exchange, False)  # private call
    bal = exchange.fetch_balance() or {}

    totals = (bal.get("total") or {}) | (bal.get("free") or {})
    markets = exchange.load_markets()

    def find_symbol_for_base(base: str) -> Optional[str]:
        for m in markets.values():
            if not m.get("active", True):
                continue
            if not m.get("spot", False):
                continue
            if (m.get("base") or "").upper() == base.upper() and (m.get("quote") or "").upper() == quote_asset.upper():
                return m["symbol"]
        return None

    quote_cash = float(totals.get(quote_asset, 0.0) or 0.0)
    total_equity = quote_cash
    positions: List[tuple[str, float]] = []

    for cur, amt in totals.items():
        if not amt or amt <= 0:
            continue
        if cur.upper() == quote_asset.upper():
            continue
        sym = find_symbol_for_base(cur)
        if not sym:
            continue
        try:
            ticker = exchange.fetch_ticker(sym)
            px = float(ticker.get("last") or 0.0)
            if px <= 0:
                continue
            value = float(amt) * px
            positions.append((sym, value))
            total_equity += value
        except Exception:
            continue

    weights: Dict[str, float] = {}
    if total_equity > 0:
        for sym, value in positions:
            weights[sym] = value / total_equity

    return weights, total_equity

def clamp_notional_for_cap(
    symbol: str,
    proposed_notional: float,
    weights: Dict[str, float],
    total_equity: float,
    cap: float,
) -> float:
    """
    Returns the max allowed notional for this buy without exceeding 'cap' weight.
    """
    current_w = float(weights.get(symbol, 0.0))
    room = max(0.0, cap - current_w)  # fraction of equity available
    max_add = room * total_equity
    return max(0.0, min(proposed_notional, max_add))

# -------- Main loop --------
def main_loop():
    exchange = make_exchange()
    log.info(f"Connected to exchange: {EXCHANGE_ID} (rateLimit={getattr(exchange, 'rateLimit', 'n/a')} ms)")
    exchange.load_markets()

    while True:
        try:
            # ---- BTC market regime gate ----
            btc_ok, btc_note, btc_metrics = check_btc_ma_condition(exchange, QUOTE_ASSET, timeframe="15m")
            log.info(f"BTC regime gate: {btc_note}")

            symbols = select_markets(exchange, QUOTE_ASSET)
            log.info(f"Evaluating up to {min(len(symbols), MAX_SYMBOLS_PER_RUN)} of {len(symbols)} symbols (quote={QUOTE_ASSET})")

            # ---------- PUBLIC: Evaluate first ----------
            to_check = symbols[:MAX_SYMBOLS_PER_RUN]
            eligible = []
            ineligible = []
            for sym in to_check:
                market = exchange.market(sym)
                res = evaluate_symbol(exchange, market)
                if res.get("eligible"):
                    eligible.append(res)
                    log.info(
                        f"ELIGIBLE: {sym} | RSI14={res['rsi14']:.2f} "
                        f"SMA60={res['sma60']:.6f} < SMA240={res['sma240']:.6f} "
                        f"LAST={res['last']:.8f} VOL24H={res.get('vol24h', 0):.4f}"
                    )
                else:
                    ineligible.append(res)
                    if VERBOSE_SYMBOL_LOG:
                        note_parts = []
                        for k in ("rsi14", "sma60", "sma240", "vol24h", "listing_note"):
                            if k in res:
                                note_parts.append(f"{k}={res[k]}")
                        notes = (" | " + ", ".join(str(x) for x in note_parts)) if note_parts else ""
                        log.debug(f"INELIGIBLE: {sym} — {res.get('reason')}{notes}")

            log.info(f"Run summary (pre-balance): {len(eligible)} eligible, {len(ineligible)} ineligible (checked {len(to_check)})")
            print_eligible_table(eligible)

            # ---------- PRIVATE: Only proceed if we might place orders ----------
            if not eligible:
                log.info("No eligible symbols. Skipping balance fetch and orders.")
            else:
                if not btc_ok:
                    log.info("BTC 15m SMA180 < SMA720 condition NOT met — skipping all buy orders this run.")
                else:
                    # ---- Determine sizing and portfolio weights ----
                    weights: Dict[str, float] = {}
                    total_equity: float = 0.0

                    if DRY_RUN and not DRY_RUN_USE_BALANCE:
                        # Simulate equity across the batch if we refuse to call private APIs in DRY_RUN
                        base_notional = DRY_RUN_NOTIONAL
                        total_equity = base_notional * max(1, len(eligible))
                        log.info(
                            "DRY_RUN=1 & DRY_RUN_USE_BALANCE=0 → "
                            "using simulated equity for weights; per-order base notional "
                            f"{base_notional:.2f}"
                        )
                    else:
                        # Real (or DRY_RUN but with balances) equity & weights
                        try:
                            weights, total_equity = get_portfolio_weights(exchange, QUOTE_ASSET)
                            log.info(
                                "Portfolio snapshot: "
                                + ", ".join(f"{k}={v:.2%}" for k, v in sorted(weights.items(), key=lambda x: -x[1])) if weights else "no positions"
                            )
                        except Exception as e:
                            log.warning(f"Failed to compute portfolio weights: {e}. Falling back to simulated equity.")
                            weights, total_equity = {}, 0.0

                        if DRY_RUN:
                            base_notional = DRY_RUN_NOTIONAL
                            log.info(f"DRY_RUN=1 → Using DRY_RUN_NOTIONAL={base_notional:.2f} per order.")
                            if total_equity <= 0:
                                total_equity = base_notional * max(1, len(eligible))
                        else:
                            quote_bal = safe_get_quote_balance(exchange, QUOTE_ASSET)
                            log.info(f"Quote balance {QUOTE_ASSET}: {quote_bal:.8f}")
                            base_notional = quote_bal * 0.05
                            if base_notional < MIN_USD_ORDER:
                                log.info(
                                    f"Notional {base_notional:.2f} < MIN_USD_ORDER {MIN_USD_ORDER:.2f} — will skip orders this run."
                                )

                    # ---- Place capped orders ----
                    if btc_ok and (DRY_RUN or base_notional >= MIN_USD_ORDER):
                        for res in eligible:
                            sym = res["symbol"]

                            # Skip if already at/above cap
                            current_w = float(weights.get(sym, 0.0))
                            if current_w >= MAX_WEIGHT_PCT:
                                log.info(
                                    f"Weight cap: skipping {sym} — current weight {current_w:.2%} "
                                    f"≥ cap {MAX_WEIGHT_PCT:.0%}"
                                )
                                continue

                            # Clamp notional to avoid exceeding cap
                            adj_notional = clamp_notional_for_cap(
                                sym, base_notional, weights, total_equity, MAX_WEIGHT_PCT
                            )

                            if adj_notional < MIN_USD_ORDER:
                                log.info(
                                    f"Weight cap: adjusted notional {adj_notional:.2f} for {sym} "
                                    f"< MIN_USD_ORDER {MIN_USD_ORDER:.2f} — skipping."
                                )
                                continue

                            try:
                                market = exchange.market(sym)
                                place_market_buy(exchange, market, notional=adj_notional)

                                # Optimistically update in-memory weights so subsequent orders see the new equity/weight
                                new_total_equity = total_equity + adj_notional
                                new_weight = (current_w * total_equity + adj_notional) / new_total_equity if new_total_equity > 0 else 0.0
                                weights[sym] = min(MAX_WEIGHT_PCT, new_weight)
                                total_equity = new_total_equity

                                log.info(
                                    f"Post-trade est. weight {sym}: {weights[sym]:.2%} "
                                    f"(cap {MAX_WEIGHT_PCT:.0%})"
                                )
                            except Exception:
                                log.exception(f"Order failed for {sym}")

        except Exception:
            log.exception("Top-level loop error")

        log.info(f"Sleeping {INTERVAL_SEC}s…")
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main_loop()
