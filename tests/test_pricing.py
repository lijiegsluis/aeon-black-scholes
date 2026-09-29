import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main as bs  # noqa: E402
import quant_models as qm  # noqa: E402

# Reference: Hull, S=K=100, T=1, sigma=20%, r=5%
S, K, T, SIG, R = 100.0, 100.0, 1.0, 0.20, 0.05


class NormalDistribution(unittest.TestCase):
    def test_cdf_known_values(self):
        for cdf in (bs.norm_cdf, qm.norm_cdf):
            self.assertAlmostEqual(cdf(0.0), 0.5, places=7)
            self.assertAlmostEqual(cdf(1.96), 0.9750021, places=6)
            self.assertAlmostEqual(cdf(-1.96), 0.0249979, places=6)

    def test_cdf_symmetry(self):
        for x in (0.1, 0.5, 1.0, 2.5):
            self.assertAlmostEqual(bs.norm_cdf(x) + bs.norm_cdf(-x), 1.0, places=12)


class BlackScholes(unittest.TestCase):
    def test_reference_prices(self):
        self.assertAlmostEqual(bs.bs_call(S, K, T, SIG, R), 10.4506, places=4)
        self.assertAlmostEqual(bs.bs_put(S, K, T, SIG, R), 5.5735, places=4)
        self.assertAlmostEqual(qm.bs_price(S, K, T, R, SIG, True), 10.4506, places=4)

    def test_put_call_parity(self):
        for s, k, t, sig, r in [(110, 100, 0.5, 0.3, 0.03), (90, 100, 2, 0.15, 0.0)]:
            lhs = bs.bs_call(s, k, t, sig, r) - bs.bs_put(s, k, t, sig, r)
            self.assertAlmostEqual(lhs, s - k * math.exp(-r * t), places=10)

    def test_greeks(self):
        g = bs.bs_greeks(S, K, T, SIG, R, "call")
        self.assertAlmostEqual(g["delta"], 0.6368, places=4)
        self.assertAlmostEqual(g["gamma"], 0.0188, places=4)
        self.assertAlmostEqual(g["vega"], 0.3752, places=4)
        self.assertAlmostEqual(g["theta"], -0.0176, places=4)
        self.assertAlmostEqual(g["rho"], 0.5323, places=4)
        p = bs.bs_greeks(S, K, T, SIG, R, "put")
        self.assertAlmostEqual(g["delta"] - p["delta"], 1.0, places=10)

    def test_implied_vol_round_trip(self):
        for vol in (0.1, 0.2, 0.45):
            price = bs.bs_call(S, K, T, vol, R)
            self.assertAlmostEqual(bs.implied_vol_newton(price, S, K, T, R, "call"), vol, places=6)


class Bonds(unittest.TestCase):
    def test_par_bond(self):
        b = bs.bond_analytics(1000, 0.05, 2, 10, 0.05)
        self.assertAlmostEqual(b["clean"], 1000.0, places=6)
        self.assertAlmostEqual(b["modified"], 7.7946, places=4)

    def test_zero_coupon(self):
        b = bs.bond_analytics(1000, 0.0, 2, 10, 0.04)
        self.assertAlmostEqual(b["clean"], 1000 / 1.02 ** 20, places=6)
        self.assertAlmostEqual(b["macaulay"], 10.0, places=10)


class MonteCarlo(unittest.TestCase):
    def test_vanilla_mc_converges_to_black_scholes(self):
        random.seed(7)
        res = qm.price_exotic(S, K, T, R, SIG, 4000, "vanilla_call", barrier=0)
        self.assertLess(abs(res["price"] - 10.4506), 4 * res["std_error"] + 0.05)

    def test_bachelier_atm(self):
        # ATM Bachelier call = sigma * sqrt(T / 2pi)
        self.assertAlmostEqual(qm.bachelier_price(100, 100, 1, 20, True)["price"], 20 / math.sqrt(2 * math.pi), places=9)


if __name__ == "__main__":
    unittest.main()
