#!/usr/bin/env python3
"""Kraken crypto scanner.

Scans Kraken's public market data (no API key, no trading) and ranks pairs
by a composite, math-based "setup score". It only surfaces candidates to look
at; it never places orders. See README.md for the math behind every signal.

Usage:
    python3 kraken_scanner.py                     # scan USD pairs once
    python3 kraken_scanner.py --top 15 --min-volume 2000000
    python3 kraken_scanner.py --loop 900          # rescan every 15 minutes
    python3 kraken_scanner.py --json results.json # also save results
"""

import argparse
import json
import math
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.kraken.com/0/public"
HOURS_PER_YEAR = 24 * 365


# --------------------------------------------------------------------------
# Kraken public API
# --------------------------------------------------------------------------

def kraken_get(endpoint, params=None, retries=3):
    url = f"{API}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kraken-scanner/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.load(resp)
            if data.get("error"):
                raise RuntimeError(f"Kraken error: {data['error']}")
            return data["result"]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def usd_pairs():
    """Return {pair_key: display_name} for online spot pairs quoted in USD."""
    pairs = kraken_get("AssetPairs")
    out = {}
    for key, p in pairs.items():
        if key.endswith(".d") or p.get("status", "online") != "online":
            continue
        if p.get("quote") not in ("ZUSD", "USD"):
            continue
        out[key] = p.get("wsname") or p.get("altname") or key
    return out


def tickers(pair_keys):
    result = {}
    for i in range(0, len(pair_keys), 100):
        chunk = pair_keys[i:i + 100]
        result.update(kraken_get("Ticker", {"pair": ",".join(chunk)}))
        time.sleep(1.1)
    return result


def ohlc(pair_key, interval=60):
    """Hourly candles (Kraken returns up to 720). Drops the still-open last candle."""
    res = kraken_get("OHLC", {"pair": pair_key, "interval": interval})
    rows = next(v for k, v in res.items() if k != "last")
    candles = [
        {"t": int(r[0]), "o": float(r[1]), "h": float(r[2]), "l": float(r[3]),
         "c": float(r[4]), "v": float(r[6])}
        for r in rows
    ]
    return candles[:-1]


# --------------------------------------------------------------------------
# Math
# --------------------------------------------------------------------------

def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ema(xs, n):
    k = 2 / (n + 1)
    out = [xs[0]]
    for x in xs[1:]:
        out.append(x * k + out[-1] * (1 - k))
    return out


def rsi(closes, n=14):
    """Wilder's RSI."""
    gains, losses = [], []
    for a, b in zip(closes, closes[1:]):
        d = b - a
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_g = mean(gains[:n])
    avg_l = mean(losses[:n])
    for g, l in zip(gains[n:], losses[n:]):
        avg_g = (avg_g * (n - 1) + g) / n
        avg_l = (avg_l * (n - 1) + l) / n
    if avg_l == 0:
        return 100.0
    return 100 - 100 / (1 + avg_g / avg_l)


def atr(candles, n=14):
    trs = []
    for prev, cur in zip(candles, candles[1:]):
        trs.append(max(cur["h"] - cur["l"], abs(cur["h"] - prev["c"]), abs(cur["l"] - prev["c"])))
    a = mean(trs[:n])
    for tr in trs[n:]:
        a = (a * (n - 1) + tr) / n
    return a


def linreg(ys):
    """OLS of ys on 0..n-1. Returns (slope, r_squared, t_stat_of_slope)."""
    n = len(ys)
    xs = range(n)
    mx, my = (n - 1) / 2, mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in resid)
    sst = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sse / sst if sst > 0 else 0.0
    se = math.sqrt(sse / (n - 2) / sxx) if n > 2 and sse > 0 else 0.0
    t = slope / se if se > 0 else 0.0
    return slope, r2, t


def percentile_rank(history, value):
    return sum(1 for h in history if h <= value) / len(history)


def clip(x, lo=-3.0, hi=3.0):
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------
# Signals and score
# --------------------------------------------------------------------------

class Result:
    """One scanned pair's signals (a plain class so it runs on Python 3.6)."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def analyze(pair, candles):
    if len(candles) < 300:
        return None
    closes = [c["c"] for c in candles]
    vols = [c["v"] * c["c"] for c in candles]  # quote (USD) volume per hour
    if min(closes) <= 0:
        return None
    logp = [math.log(c) for c in closes]
    rets = [b - a for a, b in zip(logp, logp[1:])]

    # Volatility: stdev of hourly log returns over the last 30 days.
    sigma = stdev(rets[-720:])
    if sigma == 0:
        return None
    ann_vol = sigma * math.sqrt(HOURS_PER_YEAR)

    # Momentum z-scores: return over n hours / (sigma * sqrt(n)).
    # Under a random walk this is ~N(0,1), so |z| > 2 is unusual.
    r24 = logp[-1] - logp[-25]
    r168 = logp[-1] - logp[-169]
    z24 = r24 / (sigma * math.sqrt(24))
    z168 = r168 / (sigma * math.sqrt(168))

    # Volume surge: z-score of log(last 24h USD volume) vs prior daily windows.
    daily = [sum(vols[i:i + 24]) for i in range(len(vols) - 24 * 30, len(vols), 24)]
    daily = [d for d in daily if d > 0]
    if len(daily) < 10:
        return None
    logd = [math.log(d) for d in daily]
    base = logd[:-1]
    sd = stdev(base)
    vol_z = (logd[-1] - mean(base)) / sd if sd > 0 else 0.0

    # Trend quality: regression of log price over 72h. R^2 = how clean the
    # trend is, t-stat = how significant the slope is.
    _, r2, t_stat = linreg(logp[-72:])

    # Bollinger squeeze: percentile of current 20h bandwidth vs the last 30 days.
    bw = []
    for i in range(20, len(closes) + 1):
        w = closes[i - 20:i]
        m = mean(w)
        bw.append(4 * stdev(w) / m)
    squeeze_pct = percentile_rank(bw[-720:], bw[-1])
    w = closes[-20:]
    upper = mean(w) + 2 * stdev(w)
    breakout = closes[-1] > upper

    r = rsi(closes)
    ema50, ema200 = ema(closes, 50)[-1], ema(closes, 200)[-1]
    uptrend = ema50 > ema200

    # ATR-based levels: stop 2 ATR below, target 3 ATR above (1.5 R:R baseline),
    # target extended to the 30-day high if that is further.
    a = atr(candles[-200:])
    price = closes[-1]
    stop = price - 2 * a
    hi30 = max(c["h"] for c in candles[-720:])
    target = max(price + 3 * a, hi30)
    rr = (target - price) / (price - stop) if price > stop else 0.0

    # Composite score: weighted sum of bounded, roughly unit-scale features.
    rsi_term = -abs(r - 60) / 15 + 1          # best near 60 (strong, not overbought)
    rsi_term -= 1.5 if r > 80 else 0          # extra penalty for blow-off tops
    raw = (
        0.25 * clip(z24)
        + 0.20 * clip(z168)
        + 0.20 * clip(vol_z)
        + 0.15 * clip(t_stat / 3) * r2
        + 0.10 * (1 if uptrend else -1)
        + 0.10 * clip(rsi_term)
        + (0.15 if breakout and squeeze_pct < 0.5 else 0.0)  # breakout from a squeeze
    )
    score = 100 / (1 + math.exp(-2 * raw))   # logistic map to 0..100

    notes = []
    if breakout and squeeze_pct < 0.5:
        notes.append("squeeze breakout")
    elif squeeze_pct < 0.15:
        notes.append("tight squeeze")
    if vol_z > 2:
        notes.append("volume surge")
    if r > 80:
        notes.append("overbought")
    if z24 > 2.5:
        notes.append("extended 24h")
    if uptrend:
        notes.append("EMA50>200")

    return Result(
        pair=pair, price=price, score=round(score, 1),
        ret_24h=round(math.expm1(r24) * 100, 2), ret_7d=round(math.expm1(r168) * 100, 2),
        mom_z_24h=round(z24, 2), mom_z_7d=round(z168, 2), vol_z=round(vol_z, 2),
        rsi14=round(r, 1), trend_r2=round(r2, 2), trend_t=round(t_stat, 1),
        squeeze_pct=round(squeeze_pct, 2), breakout=breakout,
        ann_vol=round(ann_vol * 100, 1), stop=stop, target=target,
        reward_risk=round(rr, 2), notes=", ".join(notes),
    )


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def fmt_price(p):
    return f"{p:,.2f}" if p >= 1 else f"{p:.6g}"


def print_table(results):
    hdr = f"{'#':>2} {'PAIR':<14}{'SCORE':>6}{'PRICE':>13}{'24h%':>8}{'7d%':>8}{'z24':>6}{'volZ':>6}{'RSI':>6}{'R2':>5}{'R:R':>6}  NOTES"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(results, 1):
        print(f"{i:>2} {r.pair:<14}{r.score:>6.1f}{fmt_price(r.price):>13}{r.ret_24h:>8.2f}{r.ret_7d:>8.2f}"
              f"{r.mom_z_24h:>6.2f}{r.vol_z:>6.2f}{r.rsi14:>6.1f}{r.trend_r2:>5.2f}{r.reward_risk:>6.2f}  {r.notes}")
    if results:
        print("\nLevels for top 5 (ATR-based, not advice):")
        for r in results[:5]:
            print(f"  {r.pair:<14} entry {fmt_price(r.price)}  stop {fmt_price(r.stop)}  target {fmt_price(r.target)}")


def scan(min_volume, top, max_pairs):
    pairs = usd_pairs()
    tick = tickers(list(pairs))
    liquid = []
    for key, t in tick.items():
        last = float(t["c"][0])
        vol_usd = float(t["v"][1]) * last
        if vol_usd >= min_volume:
            liquid.append((vol_usd, key))
    liquid.sort(reverse=True)
    liquid = liquid[:max_pairs]
    print(f"Scanning {len(liquid)} USD pairs with >= ${min_volume:,.0f} 24h volume...", file=sys.stderr)

    results = []
    for _, key in liquid:
        try:
            res = analyze(pairs.get(key, key), ohlc(key))
            if res:
                results.append(res)
        except Exception as e:
            print(f"  skip {key}: {e}", file=sys.stderr)
        time.sleep(1.1)  # stay under Kraken's public rate limit
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scan Kraken USD pairs for momentum/volume setups.")
    ap.add_argument("--min-volume", type=float, default=1_000_000, help="minimum 24h USD volume")
    ap.add_argument("--top", type=int, default=10, help="how many results to show")
    ap.add_argument("--max-pairs", type=int, default=80, help="cap on pairs analyzed (by volume)")
    ap.add_argument("--json", metavar="FILE", help="also write results to a JSON file")
    ap.add_argument("--loop", type=int, metavar="SECONDS", help="rescan every N seconds")
    args = ap.parse_args(argv)

    while True:
        results = scan(args.min_volume, args.top, args.max_pairs)
        print(time.strftime("\n%Y-%m-%d %H:%M:%S %Z"))
        print_table(results)
        if args.json:
            with open(args.json, "w") as f:
                json.dump([vars(r) for r in results], f, indent=2)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
