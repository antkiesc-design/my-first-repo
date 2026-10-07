# my-first-repo

## Kraken crypto scanner

`kraken_scanner.py` scans every liquid USD pair on Kraken using the **public**
API. It needs no API key and never places trades. It ranks pairs by a
composite, statistics-based "setup score" (0–100).

> This is a screening tool, not financial advice. A high score means "unusual
> strength worth a look." It does not mean the price will go up. Crypto is
> volatile; size positions so a stop-out doesn't hurt.

### Run it

Requires Python 3.9+ and nothing else (standard library only).

```bash
python3 kraken_scanner.py                       # scan once, top 10
python3 kraken_scanner.py --top 20 --min-volume 5000000
python3 kraken_scanner.py --loop 900            # rescan every 15 min
python3 kraken_scanner.py --json results.json   # save results
python3 -m unittest test_kraken_scanner         # run tests
```

A full scan takes 1–2 minutes. It pauses about 1 second between requests to
stay under Kraken's public rate limit.

### How it works

1. Gets all online spot pairs quoted in USD (`AssetPairs`).
2. Keeps pairs with 24h USD volume ≥ `--min-volume` (`Ticker`). Thin markets
   give noisy signals and are hard to exit.
3. Downloads 720 hourly candles (30 days) per pair (`OHLC`) and computes the
   signals below.

### The math

Let `p_t` be the hourly close and `r_t = ln(p_t / p_{t-1})` the log return.
`σ` is the standard deviation of hourly log returns over the last 30 days.

| Signal | Formula | Why it matters |
|---|---|---|
| **Momentum z-score (24h, 7d)** | `z_n = ln(p_now / p_{now-n}) / (σ·√n)` | Under a random walk, an n-hour return has stdev `σ√n`, so `z` is roughly N(0,1). `z > 2` means a move larger than ~97.5% of what noise alone would produce. Dividing by σ also makes coins with very different volatility comparable. |
| **Volume z-score** | `(ln V_24h − mean(ln V_daily)) / sd(ln V_daily)` over the prior 29 days | Price moves backed by unusual volume are more likely to continue. Logs are used because volume is roughly log-normal. |
| **Trend quality** | OLS regression of `ln p` on time over 72h → slope t-stat and R² | The t-stat measures whether the slope is significantly non-zero. R² measures how *clean* the trend is: a steady climb scores higher than a single spike. |
| **Bollinger squeeze** | Bandwidth `4·sd₂₀ / mean₂₀`, ranked against the last 30 days | Periods of low volatility tend to be followed by high volatility. A close above the upper band coming out of a low-bandwidth period is flagged as a *squeeze breakout*. |
| **RSI(14)** | Wilder's smoothing: `100 − 100/(1 + avgGain/avgLoss)` | Scores highest near 60 (strong but not stretched). Above 80 is penalized as likely overbought. |
| **Regime** | EMA50 vs EMA200 on hourly closes | Favors coins already in an uptrend. |
| **Volatility** | `σ · √8760` (annualized) | Shown for context and position sizing. |

**Composite score.** Each feature is clipped to [-3, 3] so one outlier can't
dominate. The clipped features are combined as:

```
raw = 0.25·z24 + 0.20·z7d + 0.20·volZ + 0.15·(t/3)·R²
    + 0.10·(EMA50>EMA200 ? 1 : −1) + 0.10·rsiTerm + 0.15·[squeeze breakout]
score = 100 / (1 + e^(−2·raw))
```

The logistic function maps `raw` to 0–100. A pure random walk lands near 50,
and strong setups approach 100.

**Risk levels.** ATR(14) is the average true range, a measure of typical hourly
movement. The scanner sets:

- `stop = price − 2·ATR`
- `target = max(price + 3·ATR, 30-day high)`
- `R:R = (target − price) / (price − stop)`

Prefer setups with R:R ≥ 1.5. One way to size a position is
`risk_per_trade / (price − stop)`, for example risking 1% of your account per
trade.

### Limitations

- These are heuristics. The weights were chosen by hand, not fit to data.
  Before trusting the score with real money, backtest it: log each scan's top
  picks with `--json` and check how they performed 24–72h later.
- Momentum signals catch moves already in progress. They will sometimes flag
  the top of a pump. That's why the `overbought` and `extended 24h` notes exist.
- The scanner only uses hourly candles. Events like news and token unlocks are
  not considered.
