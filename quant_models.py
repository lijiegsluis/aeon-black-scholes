"""
Aeon Quantitative Finance Models — Extended Suite
===================================================
Advanced implementations beyond the core Black-Scholes suite:

  1. Exotic Options Pricer (Monte Carlo)
       Vanilla, Binary, Asian, Barrier (Up-and-Out, Up-and-In)
  2. GBM Path Simulator
       Geometric Brownian Motion — visualise price path uncertainty
  3. Heston Stochastic Volatility
       Mean-reverting CIR variance correlated with price — vol clustering
  4. Bachelier (ABM) Model
       Arithmetic Brownian Motion — for negative-price environments (rates, spreads)

All models implemented from scratch in the Python standard library.
No external dependencies (no numpy, no scipy, no pandas).

LiJie Guo · Aeon Nimbus Research · lijieguo.substack.com
"""

import math
import random
import os
import sys
import statistics


# ---------------------------------------------------------------------------
# SECTION 1: MATHEMATICAL PRIMITIVES
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """
    Normal CDF via Abramowitz & Stegun (1964) rational approximation.
    Coefficients: a1..a5, p  |  Max absolute error: 1.5 × 10^-7
    """
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    sign = 1 if x >= 0 else -1
    ax = abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-ax * ax)
    return 0.5 * (1.0 + sign * y)


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def randn() -> float:
    """
    Box-Muller transform: generate Z ~ N(0,1) from two uniform samples.
    u1, u2 ∈ (0,1] — avoids log(0) by rejection of exact 0.
    """
    u1 = random.random()
    u2 = random.random()
    while u1 == 0:
        u1 = random.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Standard Black-Scholes price for European call or put."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if is_call else max(K - S, 0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if is_call:
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


# ---------------------------------------------------------------------------
# SECTION 2: DISPLAY HELPERS
# ---------------------------------------------------------------------------

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def separator(char: str = "─", width: int = 64):
    print("  " + char * width)


def header(title: str):
    print()
    separator("═")
    print(f"  {title}")
    separator("═")
    print()


def pf(label: str, value: str, note: str = ""):
    """Print a labelled result line."""
    line = f"  {label:<32s} : {value}"
    if note:
        line += f"   ({note})"
    print(line)


def prompt(label: str, default) -> str:
    raw = input(f"  {label} [{default}]: ").strip()
    return raw if raw else str(default)


def prompt_float(label: str, default: float) -> float:
    while True:
        raw = input(f"  {label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  [!] Please enter a number.")


def prompt_int(label: str, default: int) -> int:
    while True:
        raw = input(f"  {label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  [!] Please enter an integer.")


def ascii_bar(value: float, max_val: float, width: int = 40) -> str:
    if max_val == 0:
        return "░" * width
    filled = round(min(value / max_val, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# SECTION 3: EXOTIC OPTIONS — MONTE CARLO
# ---------------------------------------------------------------------------

OPTION_TYPES = {
    "1": ("vanilla_call",     "Vanilla Call"),
    "2": ("vanilla_put",      "Vanilla Put"),
    "3": ("binary_call",      "Binary Call  ($1 payout if S_T > K)"),
    "4": ("binary_put",       "Binary Put   ($1 payout if S_T < K)"),
    "5": ("asian_call",       "Asian Call   (payoff on average price)"),
    "6": ("asian_put",        "Asian Put    (payoff on average price)"),
    "7": ("barrier_uo_call",  "Barrier — Up-and-Out Call"),
    "8": ("barrier_ui_call",  "Barrier — Up-and-In  Call"),
}


def price_exotic(
    S: float, K: float, T: float, r: float, sigma: float,
    N: int, option_type: str, barrier: float
) -> dict:
    """
    Price an exotic option via Monte Carlo simulation.

    Discretises T into steps ~ 252 per year.
    Each path tracks:
      - terminal price S_T
      - running average (for Asian options)
      - running maximum (for barrier options)

    Returns dict with: price, std_error, ci95, bs_reference
    """
    steps = max(int(T * 252), 1)
    dt = T / steps
    discount = math.exp(-r * T)

    payoffs = []
    for _ in range(N):
        s = S
        sum_s = s
        max_s = s
        knocked_out = False
        knocked_in  = False

        for _ in range(steps):
            z = randn()
            s *= math.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z)
            sum_s += s
            if s > max_s:
                max_s = s
            if option_type == "barrier_uo_call" and s >= barrier:
                knocked_out = True
            if option_type == "barrier_ui_call" and s >= barrier:
                knocked_in = True

        avg = sum_s / (steps + 1)

        if option_type == "vanilla_call":
            pv = max(s - K, 0) * discount
        elif option_type == "vanilla_put":
            pv = max(K - s, 0) * discount
        elif option_type == "binary_call":
            pv = (1.0 if s > K else 0.0) * discount
        elif option_type == "binary_put":
            pv = (1.0 if s < K else 0.0) * discount
        elif option_type == "asian_call":
            pv = max(avg - K, 0) * discount
        elif option_type == "asian_put":
            pv = max(K - avg, 0) * discount
        elif option_type == "barrier_uo_call":
            pv = (0.0 if knocked_out else max(s - K, 0)) * discount
        elif option_type == "barrier_ui_call":
            pv = (max(s - K, 0) if knocked_in else 0.0) * discount
        else:
            pv = 0.0

        payoffs.append(pv)

    mean_p = sum(payoffs) / N
    variance = sum((p - mean_p) ** 2 for p in payoffs) / N
    std_err = math.sqrt(variance / N)
    is_call = "call" in option_type
    bs_ref = bs_price(S, K, T, r, sigma, is_call)

    return {
        "price":        mean_p,
        "std_error":    std_err,
        "ci95":         1.96 * std_err,
        "bs_reference": bs_ref,
        "n_paths":      N,
    }


def run_exotic_options():
    clear()
    header("EXOTIC OPTIONS PRICER — MONTE CARLO")

    print("  Option types:")
    for k, (_, label) in OPTION_TYPES.items():
        print(f"    {k}. {label}")
    print()

    type_key = input("  Select option type [1]: ").strip() or "1"
    option_type, type_label = OPTION_TYPES.get(type_key, OPTION_TYPES["1"])

    separator()
    print("  Parameters:")
    S       = prompt_float("Spot price S",            100.0)
    K       = prompt_float("Strike K",                100.0)
    T       = prompt_float("Time to expiry T (years)",   0.5)
    r_pct   = prompt_float("Risk-free rate r (%)",       5.0)
    sig_pct = prompt_float("Volatility σ (%)",          20.0)
    N       = prompt_int  ("Number of paths N",        5000 )
    barrier = S  # default
    if "barrier" in option_type:
        barrier = prompt_float("Barrier level (default 120% of spot)", S * 1.2)

    r     = r_pct / 100.0
    sigma = sig_pct / 100.0

    print()
    print(f"  Running {N:,} Monte Carlo paths...")
    result = price_exotic(S, K, T, r, sigma, N, option_type, barrier)

    header(f"RESULTS — {type_label}")
    pf("MC Price",        f"${result['price']:.4f}")
    pf("95% CI ±",        f"${result['ci95']:.4f}")
    pf("BS Reference",    f"${result['bs_reference']:.4f}", "vanilla benchmark (call/put)")
    pf("Paths simulated", f"{result['n_paths']:,}")
    print()
    if "barrier" in option_type:
        print(f"  Barrier level:  ${barrier:.2f}")
        print(f"  Note: Barrier options are path-dependent — every step of each path")
        print(f"        is simulated individually. Use N ≥ 10,000 for stable estimates.")
    if "binary" in option_type:
        print(f"  Binary payout:  $1.00 if triggered, $0 otherwise")
    if "asian" in option_type:
        print(f"  Asian payoff:   based on arithmetic average price across all path steps")
    print()
    print(f"  Convergence note: Standard error = ${result['std_error']:.5f}")
    print(f"  Increase N for narrower confidence intervals (SE ∝ 1/√N).")


# ---------------------------------------------------------------------------
# SECTION 4: GBM PATH SIMULATOR
# ---------------------------------------------------------------------------

def simulate_gbm_paths(
    S0: float, mu: float, sigma: float, T: float, n_paths: int
) -> list:
    """
    Simulate n_paths of Geometric Brownian Motion.
    Returns list of paths, each a list of (steps+1) prices.
    S(t+dt) = S(t) * exp[(μ - σ²/2)·dt + σ·√dt·Z]
    """
    steps = max(int(T * 252), 1)
    dt = T / steps
    drift = (mu - 0.5 * sigma ** 2) * dt
    vol_dt = sigma * math.sqrt(dt)

    paths = []
    for _ in range(n_paths):
        path = [S0]
        s = S0
        for _ in range(steps):
            s *= math.exp(drift + vol_dt * randn())
            path.append(s)
        paths.append(path)
    return paths


def run_gbm_simulator():
    clear()
    header("GBM PATH SIMULATOR — Geometric Brownian Motion")

    print("  dS = μ·S·dt + σ·S·dW")
    print("  S(t) = S₀ · exp[(μ − σ²/2)t + σ√t·Z]  where Z ~ N(0,1)")
    print()

    S0      = prompt_float("Initial price S₀",          100.0)
    mu_pct  = prompt_float("Annual drift μ (%)",            8.0)
    sig_pct = prompt_float("Annual volatility σ (%)",      20.0)
    T       = prompt_float("Time horizon T (years)",         1.0)
    n_paths = prompt_int  ("Number of paths",               20  )

    mu    = mu_pct  / 100.0
    sigma = sig_pct / 100.0
    n_paths = min(n_paths, 100)

    paths = simulate_gbm_paths(S0, mu, sigma, T, n_paths)
    finals = [p[-1] for p in paths]

    # Theoretical vs simulated
    E_ST   = S0 * math.exp(mu * T)
    med_ST = S0 * math.exp((mu - 0.5 * sigma ** 2) * T)

    sim_mean = sum(finals) / n_paths
    sim_std  = math.sqrt(sum((f - sim_mean) ** 2 for f in finals) / n_paths)
    p_profit = sum(1 for f in finals if f > S0) / n_paths

    header("GBM RESULTS")
    pf("Theoretical E[S_T]",   f"${E_ST:.2f}",  "S₀·exp(μT)")
    pf("Theoretical Median",   f"${med_ST:.2f}", "S₀·exp((μ−σ²/2)T)")
    pf("Simulated Mean",       f"${sim_mean:.2f}")
    pf("Simulated Std Dev",    f"${sim_std:.2f}")
    pf("P(S_T > S₀)",          f"{p_profit*100:.1f}%")
    print()

    # ASCII fan chart
    steps_display = max(int(T * 252), 1)
    n_cols = 50
    lo1 = []
    hi1 = []
    med_path = []
    for t_idx in range(0, steps_display + 1, max(1, steps_display // n_cols)):
        col = [p[min(t_idx, len(p)-1)] for p in paths]
        col.sort()
        q05 = col[max(0, int(len(col)*0.05))]
        q50 = col[len(col)//2]
        q95 = col[min(len(col)-1, int(len(col)*0.95))]
        lo1.append(q05)
        hi1.append(q95)
        med_path.append(q50)

    all_vals = lo1 + hi1 + [S0]
    min_v = min(all_vals) * 0.97
    max_v = max(all_vals) * 1.03
    span = max_v - min_v

    chart_h = 12
    chart_w = len(med_path)

    print("  ── Simulation Fan Chart (ASCII) ─────────────────────────────")
    # Build grid
    grid = [[" "] * chart_w for _ in range(chart_h)]
    for col_i in range(chart_w):
        lo_row = int((1 - (lo1[col_i] - min_v) / span) * (chart_h - 1))
        hi_row = int((1 - (hi1[col_i] - min_v) / span) * (chart_h - 1))
        med_row = int((1 - (med_path[col_i] - min_v) / span) * (chart_h - 1))
        lo_row  = max(0, min(chart_h-1, lo_row))
        hi_row  = max(0, min(chart_h-1, hi_row))
        med_row = max(0, min(chart_h-1, med_row))
        for r in range(hi_row, lo_row + 1):
            grid[r][col_i] = "·"
        if 0 <= med_row < chart_h:
            grid[med_row][col_i] = "█"

    for row_i, row in enumerate(grid):
        price = max_v - row_i * (span / (chart_h - 1))
        print(f"  ${price:6.1f} |{''.join(row)}|")
    print(f"  {'':>8}  T=0" + " " * (chart_w - 4) + f"T={T:.1f}yr")
    print()
    print("  Legend: █ = median path   · = 5th–95th percentile band")


# ---------------------------------------------------------------------------
# SECTION 5: HESTON STOCHASTIC VOLATILITY
# ---------------------------------------------------------------------------

def simulate_heston(
    S0: float, mu: float, v0: float, theta: float,
    kappa: float, xi: float, rho: float, T: float, n_paths: int
) -> dict:
    """
    Simulate n_paths under the Heston (1993) stochastic volatility model.

    Price dynamics:
        dS  = μS dt + √v · S dW₁
        dv  = κ(θ − v) dt + ξ√v dW₂
        corr(dW₁, dW₂) = ρ

    Discretised via Euler-Maruyama with variance floor at 1e-8.
    Feller condition for strictly positive variance: 2κθ > ξ²
    """
    steps = max(int(T * 252), 1)
    dt = T / steps

    heston_finals = []
    gbm_finals    = []  # GBM comparison (constant vol = sqrt(v0))
    heston_paths  = []
    gbm_paths     = []

    for _ in range(n_paths):
        S  = S0
        v  = v0
        Sg = S0
        hp = [S]
        gp = [Sg]

        for _ in range(steps):
            z1 = randn()
            z2 = randn()
            # Cholesky decomposition for correlated Brownians
            w1 = z1
            w2 = rho * z1 + math.sqrt(max(1.0 - rho * rho, 0.0)) * z2

            v_floor = max(v, 0.0)

            # Heston price update (Euler-Maruyama)
            S *= math.exp(
                (mu - 0.5 * v_floor) * dt
                + math.sqrt(v_floor * dt) * w1
            )
            # Variance update (with reflection floor)
            v += kappa * (theta - v) * dt + xi * math.sqrt(v_floor * dt) * w2
            v  = max(v, 1e-8)

            # GBM comparison path (constant vol = sqrt(v0))
            Sg *= math.exp(
                (mu - 0.5 * v0) * dt
                + math.sqrt(v0 * dt) * randn()
            )

            hp.append(S)
            gp.append(Sg)

        heston_finals.append(S)
        gbm_finals.append(Sg)
        heston_paths.append(hp)
        gbm_paths.append(gp)

    def stats(vals):
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - m)**2 for x in vals) / len(vals))
        sv = sorted(vals)
        p5  = sv[max(0, int(len(sv)*0.05))]
        p95 = sv[min(len(sv)-1, int(len(sv)*0.95))]
        return {"mean": m, "std": sd, "p5": p5, "p95": p95}

    feller_ok = 2 * kappa * theta > xi ** 2

    return {
        "heston_stats":  stats(heston_finals),
        "gbm_stats":     stats(gbm_finals),
        "heston_paths":  heston_paths,
        "gbm_paths":     gbm_paths,
        "feller_ok":     feller_ok,
        "n_paths":       n_paths,
    }


def run_heston():
    clear()
    header("HESTON STOCHASTIC VOLATILITY MODEL")

    print("  dS = μS dt + √v · S dW₁")
    print("  dv = κ(θ−v) dt + ξ√v dW₂   corr(dW₁, dW₂) = ρ")
    print()
    print("  Black-Scholes assumes constant volatility. Heston relaxes this:")
    print("  variance v is itself stochastic and mean-reverting (CIR process).")
    print("  The correlation ρ < 0 produces the empirically observed 'leverage effect'")
    print("  — volatility rises when the stock falls.")
    print()

    S0      = prompt_float("Initial price S₀",             100.0)
    mu_pct  = prompt_float("Drift μ (%)",                     8.0)
    v0_pct2 = prompt_float("Initial variance v₀ (%² annualised, e.g. 4 → σ≈20%)", 4.0)
    theta_v = prompt_float("Long-run variance θ (%² annualised)",                  4.0)
    kappa   = prompt_float("Mean-reversion speed κ",                              1.5)
    xi      = prompt_float("Vol of vol ξ",                                        0.5)
    rho     = prompt_float("Correlation ρ (typically negative)",                 -0.7)
    T       = prompt_float("Time horizon T (years)",                               1.0)
    n_paths = prompt_int  ("Number of paths",                                       8 )

    mu    = mu_pct  / 100.0
    v0    = v0_pct2 / 10000.0   # convert %^2 to decimal^2
    theta = theta_v / 10000.0

    print()
    print("  Simulating...")
    result = simulate_heston(S0, mu, v0, theta, kappa, xi, rho, T, n_paths)

    header("HESTON vs GBM COMPARISON")
    feller = result["feller_ok"]
    print(f"  Feller condition (2κθ > ξ²): "
          f"2×{kappa}×{theta_v/10000:.4f} = {2*kappa*theta:.4f}  vs  ξ² = {xi**2:.4f}  "
          f"→ {'✓ satisfied' if feller else '✗ violated — variance may reach 0'}")
    print()

    hs = result["heston_stats"]
    gs = result["gbm_stats"]

    print(f"  {'Metric':<28}  {'Heston':>12}  {'GBM (const σ)':>14}")
    separator()
    print(f"  {'Mean S_T':<28}  ${hs['mean']:>10.2f}  ${gs['mean']:>12.2f}")
    print(f"  {'Std Dev':<28}  ${hs['std']:>10.2f}  ${gs['std']:>12.2f}")
    print(f"  {'5th percentile':<28}  ${hs['p5']:>10.2f}  ${gs['p5']:>12.2f}")
    print(f"  {'95th percentile':<28}  ${hs['p95']:>10.2f}  ${gs['p95']:>12.2f}")
    print()

    print("  Heston typically shows fatter tails and a wider distribution than GBM.")
    print("  This reflects the empirically observed leptokurtosis in equity returns.")
    print("  The negative ρ skews the distribution — more weight in the left tail,")
    print("  consistent with the equity volatility skew (put skew in implied vol).")


# ---------------------------------------------------------------------------
# SECTION 6: BACHELIER (ABM) MODEL
# ---------------------------------------------------------------------------

def bachelier_price(F: float, K: float, T: float, sigma_B: float, is_call: bool) -> dict:
    """
    Bachelier (1900) / ABM option pricing.

    Assumes price follows arithmetic Brownian motion: dS = μ dt + σ_B dW
    (not geometric — admits negative prices).

    Call:  C = (F−K)·N(d) + σ_B·√T·n(d)
    Put:   P = (K−F)·N(−d) + σ_B·√T·n(d)
    Delta: N(d)  [call],  N(d)−1  [put]
    where d = (F−K) / (σ_B·√T)
    """
    if sigma_B <= 0 or T <= 0:
        intrinsic = (F - K) if is_call else (K - F)
        return {"price": max(intrinsic, 0.0), "delta": 1.0 if F > K else 0.0, "d": 0.0}

    sqrt_T = math.sqrt(T)
    d = (F - K) / (sigma_B * sqrt_T)
    nd  = norm_cdf(d)
    nd_neg = norm_cdf(-d)
    phi_d = norm_pdf(d)

    if is_call:
        price = (F - K) * nd + sigma_B * sqrt_T * phi_d
        delta = nd
    else:
        price = (K - F) * nd_neg + sigma_B * sqrt_T * phi_d
        delta = nd - 1.0

    return {"price": max(price, 0.0), "delta": delta, "d": d}


def run_bachelier():
    clear()
    header("BACHELIER (ABM) MODEL")

    print("  dS = μ dt + σ_B dW          (arithmetic — not geometric)")
    print()
    print("  Key difference from Black-Scholes:")
    print("  • Black-Scholes: log-normal process, S_T > 0 always")
    print("  • Bachelier: normal process, S_T can be negative")
    print("  Applications: interest rate options (SABR model calibration),")
    print("  commodity spread options, swaptions where rates can go negative.")
    print()

    F       = prompt_float("Forward / Spot F",                   100.0)
    K       = prompt_float("Strike K",                           100.0)
    T       = prompt_float("Time to expiry T (years)",             0.5)
    sigma_B = prompt_float("Bachelier vol σ_B (absolute, in $)",   5.0)
    sigma_bs= prompt_float("BS vol σ_BS (%) for comparison",       5.0)

    sigma_bs_dec = sigma_bs / 100.0

    call = bachelier_price(F, K, T, sigma_B, is_call=True)
    put  = bachelier_price(F, K, T, sigma_B, is_call=False)
    bs_call = bs_price(F, K, T, 0.0, sigma_bs_dec, is_call=True)
    bs_put  = bs_price(F, K, T, 0.0, sigma_bs_dec, is_call=False)

    # ATM approximation: σ_B_atm ≈ σ_BS × F  (for ATM options)
    atm_approx = sigma_bs_dec * F

    header("BACHELIER vs BLACK-SCHOLES")
    print(f"  d = (F−K)/(σ_B·√T) = ({F}−{K})/({sigma_B}·{math.sqrt(T):.4f}) = {call['d']:.4f}")
    print()
    print(f"  {'Metric':<28}  {'Bachelier':>12}  {'Black-Scholes':>14}")
    separator()
    print(f"  {'Call Price':<28}  ${call['price']:>10.4f}  ${bs_call:>12.4f}")
    print(f"  {'Put Price':<28}  ${put['price']:>10.4f}  ${bs_put:>12.4f}")
    print(f"  {'Delta (Call)':<28}  {call['delta']:>12.4f}  {norm_cdf((math.log(F/K)+(0.5*sigma_bs_dec**2)*T)/(sigma_bs_dec*math.sqrt(T))) if F>0 and K>0 else 0:>14.4f}")
    print()
    print(f"  ATM approximation: σ_Bachelier ≈ σ_BS × F = {sigma_bs:.1f}% × {F} = {atm_approx:.2f}")
    print()
    print("  Note: Put-call parity in Bachelier is C − P = (F − K) · discount.")
    print("  For zero-rate, undiscounted: C − P = F − K (linear, not via PV(K)).")
    print()

    # Sensitivity: show prices across a range of strikes
    print("  Strike sensitivity (Bachelier call prices):")
    separator()
    print(f"  {'Strike K':>10}   {'Bachelier Call':>15}   {'BS Call':>12}   {'Intrinsic':>10}")
    separator()
    for k_mult in [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]:
        k_val = F * k_mult
        bc = bachelier_price(F, k_val, T, sigma_B, True)["price"]
        bsc = bs_price(F, k_val, T, 0.0, sigma_bs_dec, True)
        intr = max(F - k_val, 0.0)
        print(f"  ${k_val:>8.2f}   ${bc:>14.4f}   ${bsc:>11.4f}   ${intr:>9.4f}")


# ---------------------------------------------------------------------------
# SECTION 7: MAIN MENU
# ---------------------------------------------------------------------------

MENU_ITEMS = {
    "1": ("Exotic Options — Monte Carlo (Vanilla/Binary/Asian/Barrier)", run_exotic_options),
    "2": ("GBM Path Simulator — visualise stochastic price paths",       run_gbm_simulator),
    "3": ("Heston Stochastic Volatility — correlated vol-of-vol model",  run_heston),
    "4": ("Bachelier (ABM) Model — arithmetic BM for rates/spreads",     run_bachelier),
}


def main():
    while True:
        clear()
        header("AEON QUANTITATIVE MODELS — Extended Suite")
        print("  LiJie Guo · Aeon Nimbus Research · lijieguo.substack.com")
        print()
        print("  All computations in-browser (Python standard library, no dependencies)")
        print()
        for key, (label, _) in MENU_ITEMS.items():
            print(f"  {key}.  {label}")
        print()
        print("  0.  Exit")
        print()

        choice = input("  Select module [0–4]: ").strip()

        if choice == "0":
            print("\n  Goodbye.\n")
            sys.exit(0)

        if choice in MENU_ITEMS:
            _, fn = MENU_ITEMS[choice]
            fn()
            print()
            input("  Press Enter to return to main menu...")
        else:
            print("  [!] Invalid choice. Enter 0–4.")
            input("  Press Enter...")


if __name__ == "__main__":
    main()
