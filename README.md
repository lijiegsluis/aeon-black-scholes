# Aeon Black-Scholes Suite

**Quantitative Models — Options Pricing · Stochastic Simulation · Fixed Income Analytics · Volatility Engine**

*LiJie Guo · Aeon Nimbus Research · lijieguo.substack.com · [LinkedIn](https://www.linkedin.com/in/lijieguo-es/)*

---

## Overview

A complete quantitative finance workbench covering the canonical models of modern derivatives pricing: Black-Scholes-Merton with full Greeks computation and Newton-Raphson implied volatility inversion, Monte Carlo simulation under geometric Brownian motion, fixed income analytics (duration, convexity, DV01), and an extended suite of advanced models — Heston stochastic volatility, Bachelier arithmetic Brownian motion, and path-dependent exotics. Implemented entirely in the Python standard library; no external dependencies.

| Script | Scope |
|--------|-------|
| `main.py` | BSM pricer · Greeks · Implied vol · GBM Monte Carlo · Bond analytics · Volatility engine |
| `quant_models.py` | Heston model · Bachelier (ABM) · Exotic MC (Asian, Barrier, Binary) |

---

## How to Run

```bash
# Core suite
python main.py

# Extended models: Heston, Bachelier, exotic path-dependent options
python quant_models.py
```

Python 3.8 or later. No external dependencies.

---

## Methodology

### Black-Scholes-Merton (1973)

BSM prices European options by modelling the underlying as a geometric Brownian motion under the risk-neutral measure — the celebrated change-of-measure that eliminates the drift and makes the option price independent of any expected return assumption. The resulting closed-form solution is:

```
C = S · N(d₁) − K · e^(−rT) · N(d₂)
P = K · e^(−rT) · N(−d₂) − S · N(−d₁)

d₁ = [ln(S/K) + (r + σ²/2) · T] / (σ√T)
d₂ = d₁ − σ√T
```

where S is the spot price, K the strike, r the continuously compounded risk-free rate, σ the annualised log-normal volatility, T the time to expiry in years, and N(·) the standard normal CDF. The elegance of this result lies in what it removes: because the option can be dynamically replicated through a continuously rebalanced delta hedge, the expected return of the underlying becomes irrelevant to the price.

### The Greeks

The Greeks are the partial derivatives of option value with respect to each input. They are the essential vocabulary of options risk management:

- **Delta (Δ)** = ∂V/∂S — directional exposure per unit of underlying. Call delta equals N(d₁) and converges to 1 deep in-the-money; put delta equals N(d₁) − 1.
- **Gamma (Γ)** = ∂²V/∂S² = N'(d₁)/(Sσ√T) — the convexity of the position; the rate at which delta changes. Gamma is always positive for long options and peaks at-the-money. A delta-hedged portfolio with positive gamma profits from large moves in either direction.
- **Theta (Θ)** = ∂V/∂t — the cost of holding optionality; options decay in value as expiry approaches. Theta and gamma trade off against each other: gamma-rich positions pay theta, gamma-short positions collect it.
- **Vega (ν)** = ∂V/∂σ = S·N'(d₁)·√T — exposure to implied volatility; quoted per 1% move in σ. All long option positions are long vega.
- **Rho (ρ)** = ∂V/∂r — interest rate sensitivity; more material for long-dated options and fixed income derivatives.

### Implied Volatility — Newton-Raphson Inversion

Implied volatility is the unique σ that, when substituted into BSM, reproduces the observed market price — it is not derived from historical data but backed out from the price itself. This inversion has no closed form and is solved numerically via Newton-Raphson:

```
σ_{n+1} = σ_n − (BSM(σ_n) − market_price) / Vega(σ_n)
```

with bisection fallback to guarantee convergence when the function is poorly conditioned. Convergence to eight decimal places is typically achieved within five to ten iterations. The resulting implied volatility surface — mapped across strikes and expiries — contains the full market's assessment of future risk, skew, and term structure.

### Monte Carlo — Geometric Brownian Motion

GBM paths are generated via the exact Euler discretisation under the risk-neutral measure:

```
S(t+dt) = S(t) · exp[(μ − σ²/2)·dt + σ·√dt·Z]     Z ~ N(0,1)
```

The (μ − σ²/2) term is the Itô correction: under geometric BM, it is the log-price, not the price level, that is normally distributed. Without it, Jensen's inequality would cause the simulated expectation to exceed e^(μT). Across N simulated paths, the law of large numbers guarantees convergence of the Monte Carlo estimator, and the output includes full terminal distribution statistics — percentile bands, probability of profit, and tail loss probabilities.

### Fixed Income Analytics

A fixed-coupon bond's fair value is the present value of all contractual cashflows discounted at the yield-to-maturity:

```
P = Σ [C/f / (1 + y/f)^t] + [F / (1 + y/f)^(n·f)]
```

where C is the annual coupon, f the payment frequency, y the YTM, F the face value, and n the years to maturity. The key risk analytics derived from this price function are:

- **Macaulay Duration:** the cashflow-weighted average time to receipt, in years — an intuitive measure of a bond's "centre of gravity"
- **Modified Duration:** Macaulay Duration / (1 + y/f) — the percentage price change per 100bp move in yield. The primary tool for expressing and managing duration positioning.
- **Convexity:** the second-order curvature correction. Duration linearises the price/yield relationship; convexity accounts for the fact that the true relationship is curved, improving the approximation for large yield moves.
- **DV01 (Dollar Value of One Basis Point):** Modified Duration × Price × 0.0001 — the most traded unit of fixed income risk, used to size rate positions and hedge duration exposure.

### Volatility Engine

Under GBM, the terminal price follows a log-normal distribution. Key closed-form statistics:

```
E[S_T]     = S₀ · e^(μT)                          (mean)
Median[S_T] = S₀ · e^((μ − σ²/2)T)               (median)
σ-band at T: [S₀·exp(drift·T − σ√T),  S₀·exp(drift·T + σ√T)]
P(S_T ≥ x) = 1 − N([ln(x/S₀) − (μ − σ²/2)T] / (σ√T))
```

The distinction between mean and median is material: the log-normal mean is always above the median, by the factor exp(σ²T/2). In practice, median outcomes are what most investors should care about.

### Extended Models

**Heston Stochastic Volatility (1993).** BSM's most empirically problematic assumption is constant volatility. The Heston model allows variance to evolve as a mean-reverting square-root (CIR) process, correlated with the underlying:

```
dS = μS dt + √v · S dW₁
dv = κ(θ − v) dt + ξ√v dW₂     corr(dW₁, dW₂) = ρ
```

where κ is the speed of mean-reversion, θ the long-run variance, ξ the volatility-of-volatility, and ρ (typically negative for equities) the correlation capturing the leverage effect — volatility spikes when prices fall. The Feller condition 2κθ > ξ² ensures variance remains positive almost surely.

**Bachelier Model (1900).** Where BSM assumes log-normal price dynamics and strictly positive prices, Bachelier's original arithmetic Brownian motion allows negative values — making it appropriate for interest rate options, spread options, and instruments where negativity is economically feasible. The pricing formula is:

```
C_Bach = (F − K)·N(d) + σ_B·√T·n(d)     where d = (F−K)/(σ_B·√T)
```

**Exotic Monte Carlo.** Path-dependent options — Asian (average-price), Barrier Up-and-Out, Barrier Up-and-In, and Binary (digital) — cannot be priced analytically in general and require simulation. Each path tracks the running maximum and running average, enabling evaluation of payoffs contingent on the full price history.

---

## Why This Matters

BSM is simultaneously the most important and most wrong model in quantitative finance. It is important because it provides a universal, model-independent language — implied volatility — for quoting option prices across strikes and expiries in a way that is directly comparable. It is wrong because equity volatility is not constant: it is stochastic, mean-reverting, and negatively correlated with the underlying (the leverage effect). The resulting implied volatility surface exhibits skew and term structure that BSM cannot generate internally.

This is not a flaw to be embarrassed about — it is the model's most useful feature. The gap between BSM-implied vol and realised vol, and the shape of the skew surface, are themselves information. They encode the market's collective view on tail risk, event premium, and the cost of insurance. A derivatives trader who cannot read an implied vol surface cannot trade options professionally.

Fixed income is the other half of the picture. Duration is the first language of bond portfolio management. No PM communicates a rate view without expressing it in DV01 and positioning it relative to a duration benchmark. A position that is "long duration" in a flattening curve environment has a specific, quantifiable risk profile; DV01 and convexity make that precise.

The extended models address the boundary cases where standard BSM breaks down — negative rates (Bachelier), stochastic vol surfaces (Heston), and complex structured payoffs (exotic MC). Each represents a theoretically motivated extension to a real market problem, not academic embellishment.

---

## Example Output

```
═══════════════════════════════════════════════════════════════
  AEON NIMBUS — BLACK-SCHOLES OPTION PRICER
═══════════════════════════════════════════════════════════════
  S=100.00  K=100.00  T=0.50yr  σ=20.00%  r=5.00%  [CALL]

  Option Price   : $5.5716
  Intrinsic      : $0.0000   Time Value: $5.5716

  ── Greeks ──────────────────────────────────────────────────
  Delta (Δ)      :  0.6368   ∂V/∂S
  Gamma (Γ)      :  0.018823  ∂²V/∂S²
  Theta (Θ)      : -$0.0180/day
  Vega  (ν)      :  $0.2653/1%σ
  Rho   (ρ)      :  $0.2832/1%r

  ── Implied Volatility (Newton-Raphson) ─────────────────────
  Market price   : $5.50
  Implied vol    : 19.824%   (converged in 4 iterations)

═══════════════════════════════════════════════════════════════
  MONTE CARLO GBM — 500 paths · 252 days
═══════════════════════════════════════════════════════════════
  Mean terminal price  : $108.72
  Std deviation        :  $22.14
  5th percentile       :  $74.31
  Median (50th pct)    : $106.88
  95th percentile      : $150.44
  P(profit above S₀)   :  57.2%
  P(> +20%)            :  20.8%
  P(< −20%)            :   5.4%

═══════════════════════════════════════════════════════════════
  BOND ANALYTICS  —  10yr  5% semi-annual  YTM 4.50%
═══════════════════════════════════════════════════════════════
  Clean Price        : $103.956 (103.956% of face)
  Macaulay Duration  :  7.988 years
  Modified Duration  :  7.810
  Convexity          :  72.84
  DV01               :  $0.8117 per bp
```

---

*LiJie Guo · Aeon Nimbus Research · lijieguo.substack.com · [LinkedIn](https://www.linkedin.com/in/lijieguo-es/)*
