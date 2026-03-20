"""
Aeon Black-Scholes Suite
========================
Quantitative Models for Options Pricing, Fixed Income Analytics, and Stochastic Simulation

LiJie Guo · Aeon Nimbus Research · lijieguo.substack.com

Modules:
    1. Black-Scholes Option Pricer  — European calls & puts, full Greeks, implied vol
    2. Monte Carlo GBM Simulator    — Geometric Brownian Motion paths, terminal stats
    3. Bond Analytics Engine        — Clean/dirty price, duration, convexity, DV01
    4. Volatility Engine            — Uncertainty cone, P(hit target), 1-sigma range

Dependencies: Python standard library only (math, statistics, random)
"""

import math
import statistics
import random
import os

# ---------------------------------------------------------------------------
# SECTION 1: MATHEMATICAL PRIMITIVES
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """
    Cumulative normal distribution via Abramowitz & Stegun approximation.
    Maximum absolute error: 1.5e-7.

    Constants from AS 26.2.17:
        a1=0.254829592, a2=-0.284496736, a3=1.421413741,
        a4=-1.453152027, a5=1.061405429, p=0.3275911
    """
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    sign = 1.0 if x >= 0.0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    poly = (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5)))) * t
    approx = 1.0 - poly * math.exp(-0.5 * x * x)
    return 0.5 * (1.0 + sign * approx)


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def randn() -> float:
    """
    Standard normal random variate using the Box-Muller transform.
    Draws two uniform samples and returns one standard normal deviate.
    """
    while True:
        u1 = random.random()
        u2 = random.random()
        if u1 > 0.0:
            break
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z


# ---------------------------------------------------------------------------
# SECTION 2: BLACK-SCHOLES OPTION PRICER
# ---------------------------------------------------------------------------

def bs_d1_d2(S: float, K: float, T: float, sigma: float, r: float):
    """
    Compute d1 and d2 per the Black-Scholes-Merton formula.

        d1 = [ln(S/K) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
    """
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_call(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """Black-Scholes European call price: S*N(d1) - K*exp(-r*T)*N(d2)."""
    d1, d2 = bs_d1_d2(S, K, T, sigma, r)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_put(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """Black-Scholes European put price: K*exp(-r*T)*N(-d2) - S*N(-d1)."""
    d1, d2 = bs_d1_d2(S, K, T, sigma, r)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_greeks(S: float, K: float, T: float, sigma: float, r: float, option_type: str) -> dict:
    """
    Full Black-Scholes Greeks for a European option.

    Returns a dict with keys: delta, gamma, theta, vega, rho
    Theta is per calendar day. Vega and Rho are per 1% move.
    """
    d1, d2 = bs_d1_d2(S, K, T, sigma, r)
    sqrt_T = math.sqrt(T)
    nd1    = norm_pdf(d1)
    disc   = math.exp(-r * T)

    # Delta
    if option_type == "call":
        delta = norm_cdf(d1)
    else:
        delta = norm_cdf(d1) - 1.0

    # Gamma (same for call and put)
    gamma = nd1 / (S * sigma * sqrt_T)

    # Theta (per calendar day)
    if option_type == "call":
        theta = (
            -S * nd1 * sigma / (2.0 * sqrt_T)
            - r * K * disc * norm_cdf(d2)
        ) / 365.0
    else:
        theta = (
            -S * nd1 * sigma / (2.0 * sqrt_T)
            + r * K * disc * norm_cdf(-d2)
        ) / 365.0

    # Vega per 1% move
    vega = S * nd1 * sqrt_T / 100.0

    # Rho per 1% move
    if option_type == "call":
        rho = K * T * disc * norm_cdf(d2) / 100.0
    else:
        rho = -K * T * disc * norm_cdf(-d2) / 100.0

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega":  vega,
        "rho":   rho,
    }


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """
    Implied volatility via Newton-Raphson inversion of the BS price.

    Iterates: sigma_new = sigma - (BS(sigma) - market_price) / vega(sigma)
    Returns NaN if it fails to converge.
    """
    sigma = 0.25  # initial guess: 25%
    for _ in range(max_iter):
        if option_type == "call":
            price = bs_call(S, K, T, sigma, r)
        else:
            price = bs_put(S, K, T, sigma, r)

        d1, _ = bs_d1_d2(S, K, T, sigma, r)
        vega_raw = S * norm_pdf(d1) * math.sqrt(T)  # raw vega (not per 1%)

        if abs(vega_raw) < 1e-12:
            return float("nan")

        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        sigma -= diff / vega_raw

        if sigma <= 0.0:
            sigma = 1e-6  # clamp to positive

    return float("nan")


def run_black_scholes():
    """Interactive Black-Scholes option pricer."""
    print("\n" + "=" * 60)
    print("  BLACK-SCHOLES OPTION PRICER")
    print("  European Options · Full Greeks · Implied Volatility")
    print("=" * 60)
    print("  Press Enter to use default values shown in [brackets].\n")

    S     = _get_float("  Spot price S              [100.00]: ", 100.0)
    K     = _get_float("  Strike price K            [100.00]: ", 100.0)
    T     = _get_float("  Time to expiry T (years)  [  0.50]: ", 0.5)
    sigma = _get_float("  Volatility sigma (%)      [ 20.00]: ", 20.0) / 100.0
    r     = _get_float("  Risk-free rate r (%)      [  5.00]: ", 5.0) / 100.0

    otype_raw = input("  Option type (call/put)     [  call]: ").strip().lower()
    otype = otype_raw if otype_raw in ("call", "put") else "call"

    # --- Price ---
    if otype == "call":
        price = bs_call(S, K, T, sigma, r)
    else:
        price = bs_put(S, K, T, sigma, r)

    greeks = bs_greeks(S, K, T, sigma, r, otype)
    d1, d2 = bs_d1_d2(S, K, T, sigma, r)

    print("\n" + "-" * 60)
    print(f"  INPUTS")
    print(f"  {'Spot':.<30} ${S:>10.4f}")
    print(f"  {'Strike':.<30} ${K:>10.4f}")
    print(f"  {'Maturity':.<30} {T:>10.4f} yrs")
    print(f"  {'Implied Vol':.<30} {sigma*100:>9.2f}%")
    print(f"  {'Risk-Free Rate':.<30} {r*100:>9.2f}%")
    print(f"  {'Type':.<30} {'CALL' if otype=='call' else 'PUT':>10}")
    print("-" * 60)
    print(f"  INTERMEDIATES")
    print(f"  {'d1':.<30} {d1:>10.6f}")
    print(f"  {'d2':.<30} {d2:>10.6f}")
    print(f"  {'N(d1)':.<30} {norm_cdf(d1):>10.6f}")
    print(f"  {'N(d2)':.<30} {norm_cdf(d2):>10.6f}")
    print("-" * 60)
    print(f"  OPTION PRICE")
    print(f"  {'BSM Price':.<30} ${price:>10.4f}")
    print("-" * 60)
    print(f"  GREEKS")
    print(f"  {'Delta':.<30} {greeks['delta']:>10.6f}")
    print(f"  {'Gamma':.<30} {greeks['gamma']:>10.6f}")
    print(f"  {'Theta (per day)':.<30} {greeks['theta']:>10.6f}")
    print(f"  {'Vega (per 1% vol)':.<30} {greeks['vega']:>10.6f}")
    print(f"  {'Rho (per 1% rate)':.<30} {greeks['rho']:>10.6f}")
    print("-" * 60)

    # --- Implied Vol inversion ---
    iv_raw = input("\n  Enter a market price to back out implied vol\n"
                   "  (press Enter to skip): ").strip()
    if iv_raw:
        try:
            mkt = float(iv_raw)
            iv = implied_vol_newton(mkt, S, K, T, r, otype)
            if math.isnan(iv):
                print("  [!] Newton-Raphson did not converge for this price.")
            else:
                print(f"\n  {'Market Price':.<30} ${mkt:>10.4f}")
                print(f"  {'Implied Volatility':.<30} {iv*100:>9.4f}%")
        except ValueError:
            print("  [!] Invalid price entered.")

    print("=" * 60)


# ---------------------------------------------------------------------------
# SECTION 3: MONTE CARLO GBM SIMULATOR
# ---------------------------------------------------------------------------

def gbm_path(S0: float, mu: float, sigma: float, dt: float, steps: int) -> list:
    """
    Simulate one Geometric Brownian Motion price path.

        S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        where Z ~ N(0,1)
    """
    path = [S0]
    drift     = (mu - 0.5 * sigma ** 2) * dt
    vol_scale = sigma * math.sqrt(dt)
    S = S0
    for _ in range(steps):
        S = S * math.exp(drift + vol_scale * randn())
        path.append(S)
    return path


def run_monte_carlo():
    """Interactive Monte Carlo GBM simulator."""
    print("\n" + "=" * 60)
    print("  MONTE CARLO GBM SIMULATOR")
    print("  Geometric Brownian Motion · Terminal Distribution Statistics")
    print("=" * 60)
    print("  Press Enter to use default values shown in [brackets].\n")

    S0    = _get_float("  Initial price S0          [100.00]: ", 100.0)
    mu    = _get_float("  Expected return mu (%)    [  8.00]: ", 8.0) / 100.0
    sigma = _get_float("  Volatility sigma (%)      [ 20.00]: ", 20.0) / 100.0
    N     = int(_get_float("  Number of paths N         [  500]: ", 500.0))
    days  = int(_get_float("  Simulation horizon (days) [  252]: ", 252.0))

    dt = 1.0 / 252.0   # daily steps
    steps = days

    print(f"\n  Running {N:,} GBM paths over {days} days ...")

    terminals = []
    for _ in range(N):
        path = gbm_path(S0, mu, sigma, dt, steps)
        terminals.append(path[-1])

    terminals_sorted = sorted(terminals)

    mean_T   = statistics.mean(terminals)
    median_T = statistics.median(terminals)
    stdev_T  = statistics.stdev(terminals)
    min_T    = terminals_sorted[0]
    max_T    = terminals_sorted[-1]

    # Percentiles
    def percentile(data_sorted, pct):
        n = len(data_sorted)
        idx = pct / 100.0 * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return data_sorted[lo] + frac * (data_sorted[hi] - data_sorted[lo])

    p5   = percentile(terminals_sorted, 5)
    p25  = percentile(terminals_sorted, 25)
    p75  = percentile(terminals_sorted, 75)
    p95  = percentile(terminals_sorted, 95)

    # Analytical GBM terminal distribution moments
    T_yr       = days / 252.0
    analytical_mean   = S0 * math.exp(mu * T_yr)
    analytical_median = S0 * math.exp((mu - 0.5 * sigma ** 2) * T_yr)
    analytical_std    = analytical_mean * math.sqrt(math.exp(sigma ** 2 * T_yr) - 1.0)

    # Returns distribution
    log_returns = [math.log(t / S0) for t in terminals]
    mean_logret = statistics.mean(log_returns)
    std_logret  = statistics.stdev(log_returns)

    print("\n" + "-" * 60)
    print(f"  SIMULATION PARAMETERS")
    print(f"  {'Paths':.<30} {N:>10,}")
    print(f"  {'Horizon':.<30} {days:>9} days  ({T_yr:.2f} yrs)")
    print(f"  {'Expected Return':.<30} {mu*100:>9.2f}%")
    print(f"  {'Volatility':.<30} {sigma*100:>9.2f}%")
    print("-" * 60)
    print(f"  TERMINAL PRICE — SIMULATED")
    print(f"  {'Mean':.<30} ${mean_T:>10.4f}")
    print(f"  {'Median':.<30} ${median_T:>10.4f}")
    print(f"  {'Std Dev':.<30} ${stdev_T:>10.4f}")
    print(f"  {'Min':.<30} ${min_T:>10.4f}")
    print(f"  {'Max':.<30} ${max_T:>10.4f}")
    print("-" * 60)
    print(f"  TERMINAL PRICE — ANALYTICAL (LOGNORMAL)")
    print(f"  {'E[S(T)] = S0*exp(mu*T)':.<30} ${analytical_mean:>10.4f}")
    print(f"  {'Median = S0*exp((mu-0.5s²)T)':.<30} ${analytical_median:>10.4f}")
    print(f"  {'Std Dev':.<30} ${analytical_std:>10.4f}")
    print("-" * 60)
    print(f"  PERCENTILES")
    print(f"  {'5th':.<30} ${p5:>10.4f}")
    print(f"  {'25th':.<30} ${p25:>10.4f}")
    print(f"  {'75th':.<30} ${p75:>10.4f}")
    print(f"  {'95th':.<30} ${p95:>10.4f}")
    print("-" * 60)
    print(f"  LOG-RETURN DISTRIBUTION")
    print(f"  {'Mean log-return':.<30} {mean_logret:>10.4f}")
    print(f"  {'Std log-return':.<30} {std_logret:>10.4f}")
    print(f"  {'Annualised vol (from sim)':.<30} {std_logret/math.sqrt(T_yr)*100:>9.4f}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# SECTION 4: BOND ANALYTICS ENGINE
# ---------------------------------------------------------------------------

def bond_cashflows(face: float, coupon_rate: float, freq: int, maturity_yrs: float) -> list:
    """
    Generate bond cashflow schedule.

    Returns list of (period, time_years, cashflow) tuples.
    The final period includes the face value repayment.
    """
    n_periods  = int(round(maturity_yrs * freq))
    coupon_pmt = face * coupon_rate / freq
    cfs = []
    for i in range(1, n_periods + 1):
        t = i / freq
        cf = coupon_pmt if i < n_periods else coupon_pmt + face
        cfs.append((i, t, cf))
    return cfs


def bond_price_from_ytm(
    face: float,
    coupon_rate: float,
    freq: int,
    maturity_yrs: float,
    ytm: float,
) -> float:
    """
    Clean bond price = sum of PV(cashflows) discounted at YTM.
    Assumes settlement at a coupon date (no accrued interest for clean price).
    """
    cfs = bond_cashflows(face, coupon_rate, freq, maturity_yrs)
    ytm_per = ytm / freq
    price = 0.0
    for period, t, cf in cfs:
        price += cf / (1.0 + ytm_per) ** period
    return price


def bond_analytics(
    face: float,
    coupon_rate: float,
    freq: int,
    maturity_yrs: float,
    ytm: float,
    accrued_fraction: float = 0.0,
) -> dict:
    """
    Full bond analytics: clean price, dirty price, duration, convexity, DV01.

    accrued_fraction: fraction of coupon period elapsed since last coupon payment
                      (0.0 = settlement on coupon date, default).
    """
    cfs = bond_cashflows(face, coupon_rate, freq, maturity_yrs)
    ytm_per = ytm / freq
    coupon_pmt = face * coupon_rate / freq

    # Clean price (PV of all cashflows)
    clean = 0.0
    for period, t, cf in cfs:
        clean += cf / (1.0 + ytm_per) ** period

    # Accrued interest
    accrued = coupon_pmt * accrued_fraction

    # Dirty price
    dirty = clean + accrued

    # Macaulay Duration = sum(t * PV(cf)) / dirty price
    mac_num = 0.0
    for period, t, cf in cfs:
        pv = cf / (1.0 + ytm_per) ** period
        mac_num += t * pv
    macaulay = mac_num / dirty

    # Modified Duration = Macaulay / (1 + YTM/freq)
    modified = macaulay / (1.0 + ytm / freq)

    # Convexity = sum(t*(t+1/freq)*PV) / (dirty * (1+ytm_per)^2 * freq^2)
    # Standard bond convexity formula using period numbering:
    # Convexity = [1/(dirty*(1+ytm_per)^2)] * sum[period*(period+1)*PV / freq^2]
    conv_num = 0.0
    for period, t, cf in cfs:
        pv = cf / (1.0 + ytm_per) ** period
        conv_num += period * (period + 1) * pv
    convexity = conv_num / (dirty * (1.0 + ytm_per) ** 2 * freq ** 2)

    # DV01 = Modified Duration * dirty price * 0.0001
    dv01 = modified * dirty * 0.0001

    return {
        "clean":     clean,
        "dirty":     dirty,
        "accrued":   accrued,
        "macaulay":  macaulay,
        "modified":  modified,
        "convexity": convexity,
        "dv01":      dv01,
        "cashflows": cfs,
        "coupon_pmt": coupon_pmt,
    }


def run_bond_analytics():
    """Interactive bond analytics engine."""
    print("\n" + "=" * 60)
    print("  BOND ANALYTICS ENGINE")
    print("  Clean/Dirty Price · Duration · Convexity · DV01")
    print("=" * 60)
    print("  Press Enter to use default values shown in [brackets].\n")

    face        = _get_float("  Face value ($)            [1000.00]: ", 1000.0)
    coupon_rate = _get_float("  Annual coupon rate (%)    [   5.00]: ", 5.0) / 100.0
    freq_raw    = input("  Coupon frequency (1/2/4)   [      2]: ").strip()
    freq        = int(freq_raw) if freq_raw.isdigit() else 2
    maturity    = _get_float("  Maturity (years)          [  10.00]: ", 10.0)
    ytm         = _get_float("  YTM (%)                   [   4.50]: ", 4.5) / 100.0

    result = bond_analytics(face, coupon_rate, freq, maturity, ytm)

    freq_label = {1: "Annual", 2: "Semi-annual", 4: "Quarterly"}.get(freq, f"{freq}x/yr")

    print("\n" + "-" * 60)
    print(f"  BOND SPECIFICATIONS")
    print(f"  {'Face Value':.<30} ${face:>10,.2f}")
    print(f"  {'Coupon Rate':.<30} {coupon_rate*100:>9.3f}%")
    print(f"  {'Coupon Payment':.<30} ${result['coupon_pmt']:>10.4f}")
    print(f"  {'Frequency':.<30} {freq_label:>10}")
    print(f"  {'Maturity':.<30} {maturity:>9.1f} yrs")
    print(f"  {'YTM':.<30} {ytm*100:>9.3f}%")
    print("-" * 60)
    print(f"  PRICING")
    print(f"  {'Clean Price':.<30} ${result['clean']:>10.4f}")
    print(f"  {'Accrued Interest':.<30} ${result['accrued']:>10.4f}")
    print(f"  {'Dirty Price':.<30} ${result['dirty']:>10.4f}")
    print(f"  {'Price / Par':.<30} {result['clean']/face*100:>9.4f}%")
    print("-" * 60)
    print(f"  RISK METRICS")
    print(f"  {'Macaulay Duration':.<30} {result['macaulay']:>9.4f} yrs")
    print(f"  {'Modified Duration':.<30} {result['modified']:>9.4f} yrs")
    print(f"  {'Convexity':.<30} {result['convexity']:>10.4f}")
    print(f"  {'DV01 ($/bp)':.<30} ${result['dv01']:>10.4f}")
    print("-" * 60)

    show_cf = input("\n  Show full cashflow schedule? (y/n) [n]: ").strip().lower()
    if show_cf == "y":
        print("\n" + "-" * 60)
        print(f"  {'Period':<8} {'Time (yr)':<12} {'Cashflow':<14} {'PV':<14}")
        print("  " + "-" * 52)
        ytm_per = ytm / freq
        for period, t, cf in result["cashflows"]:
            pv = cf / (1.0 + ytm_per) ** period
            print(f"  {period:<8} {t:<12.4f} ${cf:<13.4f} ${pv:<13.4f}")
        print("  " + "-" * 52)
        print(f"  {'':8} {'':12} {'Total PV:':14} ${result['clean']:.4f}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# SECTION 5: VOLATILITY ENGINE
# ---------------------------------------------------------------------------

def run_volatility_engine():
    """
    Volatility analysis: expected/median terminal price, 1-sigma cone,
    P(hit target), and terminal distribution under GBM.
    """
    print("\n" + "=" * 60)
    print("  VOLATILITY ENGINE")
    print("  Uncertainty Cones · Terminal Distribution · P(Hit Target)")
    print("=" * 60)
    print("  Press Enter to use default values shown in [brackets].\n")

    S      = _get_float("  Current price S            [100.00]: ", 100.0)
    sigma  = _get_float("  Annual volatility (%)      [ 20.00]: ", 20.0) / 100.0
    mu     = _get_float("  Expected annual return (%) [  8.00]: ", 8.0) / 100.0
    days   = int(_get_float("  Horizon (days)             [  252]: ", 252.0))
    target = _get_float("  Target price               [120.00]: ", 120.0)

    T = days / 252.0  # time in years

    # --- Analytical GBM terminal distribution ---
    # ln(S(T)/S0) ~ N((mu - 0.5*sigma^2)*T, sigma^2*T)
    mu_logret  = (mu - 0.5 * sigma ** 2) * T
    std_logret = sigma * math.sqrt(T)

    expected_terminal = S * math.exp(mu * T)
    median_terminal   = S * math.exp(mu_logret)

    # 1-sigma range (covers ~68.3% of outcomes)
    lower_1sigma = S * math.exp(mu_logret - std_logret)
    upper_1sigma = S * math.exp(mu_logret + std_logret)

    # 2-sigma range (covers ~95.4% of outcomes)
    lower_2sigma = S * math.exp(mu_logret - 2.0 * std_logret)
    upper_2sigma = S * math.exp(mu_logret + 2.0 * std_logret)

    # P(S(T) >= target) under risk-neutral measure (mu -> r, set mu as given)
    # Using lognormal: z = (ln(target/S) - mu_logret) / std_logret
    if target > 0:
        z_target = (math.log(target / S) - mu_logret) / std_logret
        prob_above = 1.0 - norm_cdf(z_target)
        prob_below = norm_cdf(z_target)
    else:
        prob_above = 1.0
        prob_below = 0.0

    # Annualised vol percentage bands at horizon
    # Vol cone: at time t, the 1-sigma band = S * exp(±sigma*sqrt(t))
    # (ignoring drift for the pure vol cone)
    cone_lo = S * math.exp(-sigma * math.sqrt(T))
    cone_hi = S * math.exp( sigma * math.sqrt(T))

    # Expected maximum drawdown approximation (reflection principle for GBM)
    # E[max drawdown] ≈ sigma * sqrt(T) * sqrt(2/pi) is a simple bound
    expected_vol_move = sigma * math.sqrt(T) * 100.0  # in %

    print("\n" + "-" * 60)
    print(f"  INPUTS")
    print(f"  {'Current Price':.<30} ${S:>10.4f}")
    print(f"  {'Annual Volatility':.<30} {sigma*100:>9.2f}%")
    print(f"  {'Expected Return':.<30} {mu*100:>9.2f}%")
    print(f"  {'Horizon':.<30} {days:>9} days ({T:.4f} yrs)")
    print(f"  {'Target Price':.<30} ${target:>10.4f}")
    print("-" * 60)
    print(f"  TERMINAL PRICE DISTRIBUTION  (S(T) ~ Lognormal)")
    print(f"  {'Expected E[S(T)]':.<30} ${expected_terminal:>10.4f}")
    print(f"  {'Median S(T)':.<30} ${median_terminal:>10.4f}")
    print(f"  {'Mean log-return':.<30} {mu_logret:>10.6f}")
    print(f"  {'Std of log-return':.<30} {std_logret:>10.6f}")
    print("-" * 60)
    print(f"  UNCERTAINTY CONE  (at T = {T:.4f} yrs)")
    print(f"  {'1-Sigma Lower (~16th pct)':.<30} ${lower_1sigma:>10.4f}")
    print(f"  {'1-Sigma Upper (~84th pct)':.<30} ${upper_1sigma:>10.4f}")
    print(f"  {'2-Sigma Lower (~2.3rd pct)':.<30} ${lower_2sigma:>10.4f}")
    print(f"  {'2-Sigma Upper (~97.7th pct)':.<30} ${upper_2sigma:>10.4f}")
    print(f"  {'Pure Vol Cone Lo':.<30} ${cone_lo:>10.4f}")
    print(f"  {'Pure Vol Cone Hi':.<30} ${cone_hi:>10.4f}")
    print(f"  {'1-Sigma % Move':.<30} {expected_vol_move:>9.2f}%")
    print("-" * 60)
    print(f"  TARGET ANALYSIS  (target = ${target:.2f})")
    print(f"  {'P(S(T) >= target)':.<30} {prob_above*100:>9.4f}%")
    print(f"  {'P(S(T) <  target)':.<30} {prob_below*100:>9.4f}%")
    print(f"  {'Z-score to target':.<30} {z_target:>10.4f}")
    print("-" * 60)

    # Breakeven implied vol (what sigma makes target the median)
    # target = S * exp((sigma_be - 0.5*sigma_be^2)*T + sigma_be*0) -> no closed form
    # Simpler: what sigma_annualized makes target at 1-sigma upper bound (zero drift)
    # S * exp(sigma*sqrt(T)) = target => sigma = ln(target/S) / sqrt(T)
    if target > S:
        sigma_be = math.log(target / S) / math.sqrt(T)
        print(f"  {'Vol for target = 1-sigma upper':.<30} {sigma_be*100:>9.4f}%")

    print("=" * 60)


# ---------------------------------------------------------------------------
# SECTION 6: UTILITY / INPUT HELPERS
# ---------------------------------------------------------------------------

def _get_float(prompt: str, default: float) -> float:
    """Prompt for a float, returning default on empty input."""
    raw = input(prompt).strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  [!] Invalid input, using default {default}.")
        return default


def _clear():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def _print_banner():
    print("\n" + "=" * 60)
    print("  AEON BLACK-SCHOLES SUITE")
    print("  Quantitative Finance · Aeon Nimbus Research")
    print("  LiJie Guo · lijieguo.substack.com")
    print("=" * 60)
    print("  Select a module:\n")
    print("  [1] Black-Scholes Option Pricer")
    print("      European calls & puts · Full Greeks · Implied Vol")
    print()
    print("  [2] Monte Carlo GBM Simulator")
    print("      Geometric Brownian Motion paths · Terminal stats")
    print()
    print("  [3] Bond Analytics Engine")
    print("      Clean/Dirty price · Duration · Convexity · DV01")
    print()
    print("  [4] Volatility Engine")
    print("      Uncertainty cone · P(hit target) · 1-sigma range")
    print()
    print("  [0] Exit")
    print("=" * 60)


# ---------------------------------------------------------------------------
# SECTION 7: MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    """Main interactive CLI loop."""
    while True:
        _print_banner()
        choice = input("  Enter choice [0-4]: ").strip()

        if choice == "0":
            print("\n  Goodbye.\n")
            break
        elif choice == "1":
            run_black_scholes()
        elif choice == "2":
            run_monte_carlo()
        elif choice == "3":
            run_bond_analytics()
        elif choice == "4":
            run_volatility_engine()
        else:
            print("  [!] Invalid choice. Please enter 0-4.")

        input("\n  Press Enter to return to main menu...")


if __name__ == "__main__":
    main()
