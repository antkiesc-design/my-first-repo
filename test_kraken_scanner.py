import math
import random
import unittest

import kraken_scanner as ks


def make_candles(n=720, drift=0.0, sigma=0.01, seed=1, last_day_drift=None, last_day_vol_mult=1.0):
    rng = random.Random(seed)
    price, out = 100.0, []
    for i in range(n):
        in_last_day = i >= n - 24
        d = last_day_drift if (in_last_day and last_day_drift is not None) else drift
        o = price
        price *= math.exp(d + sigma * rng.gauss(0, 1))
        hi, lo = max(o, price) * 1.002, min(o, price) * 0.998
        v = 1000 * rng.uniform(0.8, 1.2) * (last_day_vol_mult if in_last_day else 1.0)
        out.append({"t": i * 3600, "o": o, "h": hi, "l": lo, "c": price, "v": v})
    return out


class MathTests(unittest.TestCase):
    def test_rsi_bounds(self):
        self.assertEqual(ks.rsi(list(range(1, 50))), 100.0)
        self.assertLess(ks.rsi(list(range(50, 1, -1))), 1.0)

    def test_linreg_perfect_line(self):
        slope, r2, _ = ks.linreg([2 * x + 1 for x in range(20)])
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(r2, 1.0)

    def test_ema_constant(self):
        self.assertAlmostEqual(ks.ema([5.0] * 30, 10)[-1], 5.0)


class AnalyzeTests(unittest.TestCase):
    def test_random_walk_scores_near_middle(self):
        r = ks.analyze("NOISE/USD", make_candles(seed=3))
        self.assertIsNotNone(r)
        self.assertTrue(20 < r.score < 80, r.score)

    def test_pump_with_volume_scores_higher_than_dump(self):
        pump = ks.analyze("PUMP/USD", make_candles(seed=4, last_day_drift=0.004, last_day_vol_mult=4))
        dump = ks.analyze("DUMP/USD", make_candles(seed=4, last_day_drift=-0.004))
        self.assertGreater(pump.score, dump.score)
        self.assertGreater(pump.vol_z, 2)
        self.assertGreater(pump.mom_z_24h, 0)
        self.assertLess(dump.mom_z_24h, 0)

    def test_levels_are_ordered(self):
        r = ks.analyze("X/USD", make_candles(seed=5))
        self.assertLess(r.stop, r.price)
        self.assertGreater(r.target, r.price)

    def test_too_little_history(self):
        self.assertIsNone(ks.analyze("NEW/USD", make_candles(n=100)))


if __name__ == "__main__":
    unittest.main()
