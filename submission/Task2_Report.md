# MAC-F244 — Stochastic Calculus & Application to Finance
## Task 2 — Final Submission Report

We study three NYSE/NASDAQ-listed firms with structurally different mechanisms of exposure to the Iran-US conflict:

| | Ticker | Company | Channel of war exposure |
|--|---|---|---|
| **Oil-direct**     | CVX  | Chevron Corp.        | absorbs overnight Brent moves through open-gap (jump channel) |
| **Risk-asset**     | NVDA | NVIDIA Corp.         | absorbs every oil tick continuously via broad risk sentiment  |
| **Insulated**      | PG   | Procter & Gamble     | defensive consumer staple — no detectable channel             |

**In-sample window**: 16 March – 15 April 2026 (hourly bars, n = 153 returns over 22 sessions).
**Out-of-sample window**: 16 April – 24 April 2026 (49 hourly bars, 7 sessions).
**Auxiliary geopolitical proxies**: Brent crude `BZ=F`, WTI crude `CL=F`, Lockheed Martin `LMT`.
**Data source**: Yahoo Finance via `yfinance`, cached to CSV.

> Run: `python task2_pipeline.py`. All tables and figures referenced
> below come from that script.

---

## 1. Statistical Summary, Distribution & Trend

For each ticker we work with hourly simple returns $r_t = (S_t-S_{t-1})/S_{t-1}$ and log-returns $\ell_t = \ln S_t - \ln S_{t-1}$. Sample moments (n = 153):

| Metric | CVX | NVDA | PG |
|---|---:|---:|---:|
| Mean $\hat\mu$ (% / hr) | −0.0389 | **+0.0535** | −0.0346 |
| Median (% / hr) | −0.0244 | +0.0174 | −0.0010 |
| Std-dev $\hat\sigma$ (% / hr) | **0.8878** | 0.7676 | **0.3991** |
| Min / Max | **−7.43 % / +2.15 %** | −2.15 % / +3.59 % | −1.84 % / +1.62 % |
| Skewness $\hat\gamma_1$ | **−2.541** | +1.065 | −0.148 |
| Excess kurtosis $\hat\gamma_2$ | **+14.310** | +5.033 | +5.419 |
| Annualised vol = $\hat\sigma\sqrt{6.5\!\cdot\!252}$ | **35.93 %** | 31.07 % | **16.15 %** |
| Total in-sample return | −6.35 % | **+8.04 %** | −5.27 % |
| OLS trend slope $\hat\beta$ (USD/h) | −0.107 | +0.081 | −0.032 |
| OLS $R^2$ / p | 0.369 / $7\!\!\times\!\!10^{-17}$ | 0.216 / $1\!\!\times\!\!10^{-9}$ | 0.294 / $4\!\!\times\!\!10^{-13}$ |

**Inferences across companies.**

- **Vol pattern matches the structural prior.** PG (consumer staple) has the lowest annualised vol (16 %) — about half of CVX (36 %) and NVDA (31 %): textbook defensive-vs-cyclical signature.
- **CVX has the heaviest left tail** ($\hat\gamma_1 = -2.54$, $\hat\gamma_2 = +14.3$). The −7.43 %/hr bar (4-Apr-08 13:30) is in the open bar — i.e. it is the absorption of an overnight oil/news shock. A Gaussian model would assign that bar probability $<10^{-15}$. **The leptokurtosis *is* the war-risk premium** that GBM does not capture and Merton-jump-diffusion would.
- **NVDA is right-skewed** ($\hat\gamma_1 = +1.07$). Its biggest hourly move was a *positive* +3.59 % (the 31-Mar-2026 peace-plan rally). NVDA's distribution rewards optimism — a different geopolitical signature from CVX.
- **All three trends are statistically significant**, but in different directions: CVX and PG drifted *down*, NVDA drifted *up*. CVX's slope is largest in absolute USD terms; PG's $R^2$ is highest because PG's noise is so much smaller — a clean low-vol downtrend.

### 1.1 Is each distribution Normal?

Three independent tests on $z_t = (r_t-\hat\mu)/\hat\sigma$:

| Test | CVX | NVDA | PG |
|---|---|---|---|
| **Shapiro–Wilk** $W$, p | $W=0.79$, $p\!\approx\!0$ | $W=0.88$, $p\!\approx\!0$ | $W=0.95$, $p\!\approx\!0$ |
| **Jarque–Bera** $JB=\tfrac{n}{6}(\hat\gamma_1^2+\tfrac14\hat\gamma_2^2)$, p | $\!\approx\!0$ | $\!\approx\!0$ | $\!\approx\!0$ |
| **Kolmogorov–Smirnov** $D$, p | $D=0.17$, $p=10^{-4}$ | $D=0.17$, $p=3\!\!\times\!\!10^{-4}$ | $D=0.10$, **$p=0.149$** |

Reading: SW and JB **reject normality for all three** at any reasonable level. PG is the only one whose KS p-value (0.149) survives — meaning its empirical CDF is *globally* close to a Gaussian even though the tails are still leptokurtic. This is the staple-versus-cyclical signature: PG's noise is roughly Gaussian *until* you look at the 5-sigma tail.

### 1.2 Stationarity (ADF on price + returns)

| Series | CVX | NVDA | PG |
|---|---|---|---|
| Price $S_t$ | p = **0.875** (unit root) | p = **0.976** (unit root) | p = **0.124** (unit root) |
| Return $r_t$ | p = 0.120 | p ≈ 0 | p ≈ 0 |

NVDA and PG returns are I(0) stationary. **CVX returns fail to reject the unit-root null at 5 %** (p = 0.12) — the heavy left tail and short sample inflate the variance estimator. CVX therefore sits at the edge of the standard ARIMA framework.

---

## 2. Forecasting hourly prices for 16 – 25 April 2026

> **Interpretation.** "Predicts the mean price fluctuation in the period"
> is interpreted as forecasting the **hourly price trajectory** for the
> test window. We report (a) per-hour accuracy (RMSE / MAE / MAPE /
> directional accuracy) and (b) the period-mean of the predicted prices
> versus the realised period-mean — the natural single-number "average
> fluctuation" summary.

### 2.1 The eight forecasting models

Each model produces a 49-hour price trajectory $\hat P_{T+1},\ldots,\hat P_{T+49}$ from the train end. All use only in-sample data.

| Family | Model | Form |
|---|---|---|
| Constant | **Naive** | $\hat P_{T+k} = P_T$ — random-walk baseline |
| Linear extrap. | **Drift** | $\hat P_{T+k} = P_T + \delta\cdot k$, $\delta = (P_T - P_0)/(N-1)$ |
| Box-Jenkins | **ARIMA(p,d,q)** | grid-search on $(p,d,q)\!\in\![0,2]\!\times\![0,1]\!\times\![0,2]$, AIC$=2k-2\ln L$, multi-step `.forecast` |
| Stochastic-calc. | **GBM Monte-Carlo** | $d\ln S = (\mu-\tfrac12\sigma^2)dt + \sigma dW_t$, simulate **4 000 paths**, mean trajectory + 95 % band |
| ML (linear) | **Ridge** | $\min\|y-Xw\|^2 + \lambda\|w\|^2$, $\lambda=1$, on 17 lag/rolling/price/time features (recursive) |
| ML (tree) | **Random Forest** | 400 trees, max-depth 4, min-leaf 3 (recursive) |
| ML (tree) | **Gradient Boosting** | 250 stumps, lr 0.05, max-depth 2 (recursive) |
| Bayes-blend | **Ensemble** | top-3 models by *validation* RMSE, weighted $w_i \propto 1/RMSE_i^2$ |

**Causality discipline.** ML features are 5 lagged returns, $\{$rolling mean, rolling std$\}$ at windows 3 & 5 over **shifted** returns (so the rolling window at $t$ uses $\{r_{t-1},\ldots,r_{t-w}\}$ — never $r_t$ itself), an interaction $r_{t-1}\!\cdot\!r_{t-2}$, two lagged prices, the lagged price-difference, plus four time-of-day indicators (hour, dow, $\sin/\cos$ of hour). At test time the ML stack runs **recursively**: each predicted return $\hat r_{T+k}$ is converted to a price $\hat P_{T+k} = \hat P_{T+k-1}\!\cdot\!(1+\hat r_{T+k})$ which then feeds back as `price_lag1` for step $k+1$. No look-ahead.

**Validation-based ensemble.** I split training 80/20, refit each model on the first 80 %, score validation RMSE, pick top-3, and weight $w_i\propto 1/RMSE_i^2$ (precision-weighting under iid Gaussian errors).

### 2.2 Forecast accuracy — per-hour metrics

#### Per-hour RMSE on price (USD)

| Model | CVX | NVDA | PG |
|---|---:|---:|---:|
| **Naive** | **2.097** | 3.933 | 2.771 |
| **ARIMA** | **2.097** $^{(0,1,0)}$ | 3.933 $^{(0,1,0)}$ | **2.760** $^{(2,1,2)}$ |
| **GBM** | 3.484 | **2.298** | 3.891 |
| **Drift** | 3.546 | 2.334 | 3.926 |
| **Ensemble (top-3)** | 3.124 | 2.641 | 3.236 |
| Random Forest | 6.462 | 3.092 | 3.243 |
| Ridge | 8.564 | 5.429 | 3.218 |
| Gradient Boosting | 10.510 | 4.309 | 3.275 |

#### Directional accuracy (sign of hour-to-hour change)

| Model | CVX | NVDA | PG |
|---|---:|---:|---:|
| Naive / ARIMA | 0.00 | 0.00 | 0.00–0.54 |
| Drift / GBM   | 0.42 | 0.56 | 0.54 |
| Random Forest | 0.35 | 0.54 | **0.56** |
| **Gradient Boosting** | **0.56** | **0.65** | **0.56** |
| **Ridge** | **0.58** | 0.50 | 0.50 |

#### Predicted period-mean price vs realised (USD)

| Model | CVX | NVDA | PG |
|---|---:|---:|---:|
| **Actual** | **185.31** | **201.33** | **144.99** |
| Naive | 184.83 | 198.87 | 143.35 |
| ARIMA | 184.83 | 198.87 | 143.36 |
| Drift | 182.78 | 201.29 | 142.04 |
| **GBM** | 182.82 | **201.36** | 142.07 |
| Ridge | 192.66 | 197.48 | 142.74 |
| Random Forest | 190.82 | 203.12 | 142.68 |
| Gradient Boosting | 194.78 | 198.49 | 142.84 |
| Ensemble | 183.23 | 200.50 | 142.76 |

### 2.3 Inferences — best model varies by company *and* by metric

The best forecaster is **different for each company**, and depends on which metric you prioritise:

**CVX (price was nearly flat in test, +0.15 %).** Naive and ARIMA(0,1,0) both predict the constant $P_T$ and tie at RMSE = 2.10. They win because *the test was a random-walk regime*. Their directional accuracy is 0 (a flat line never predicts a sign change). Drift/GBM extrapolate the in-sample downtrend (β = −0.107 USD/h), predicting CVX would continue falling — wrong in test. ML overshoots upward because end-of-train showed a recent rebound that recursive multi-step compounded; **Ridge has the highest directional accuracy (0.58)**.

**NVDA (price went up in test, +4.69 %).** **GBM wins** at RMSE = 2.30. Under GBM the expected log-return is $(\mu-\tfrac12\sigma^2)$; the in-sample drift was strongly positive, so the path mean projects up — and the test confirmed it. Drift is essentially tied. Random Forest hits the period-mean almost exactly (203.12 vs actual 201.33). **GBR has the highest directional accuracy (0.65)** — best at hour-to-hour signs.

**PG (price went up in test, +3.32 % with structure).** **ARIMA wins** at RMSE = 2.76 *with a non-trivial order $(2,1,2)$* — unlike CVX and NVDA where ARIMA collapsed to (0,1,0) = random walk, PG's grid-search picked two AR and two MA terms, meaning PG has detectable serial structure beyond pure random walk.

**Key conclusion: when models all predict similarly (PG) you have a noise-dominated regime; when they diverge (CVX) you have a regime-mismatch where models that extrapolate trends are punished if the trend reverses.** GBM's value is twofold: (i) competitive on all three names in MAPE and (ii) the 4 000-path Monte-Carlo gives a 95 % band (`{TICKER}_panel.png`) that is the only honest uncertainty quantifier in the toolkit.

### 2.4 How to improve

1. **Volatility model.** With $\hat\gamma_2 \in [5,14]$ a constant-σ assumption is wrong; the GBM band is too narrow for CVX. Fit GARCH(1,1) and feed $\sigma_t$ into a heteroskedastic-noise GBM, or directly to a Merton jump-diffusion: $dS/S = \mu dt + \sigma dW + (J-1)dN_t$ with $\ln J \sim \mathcal N$ and $N_t$ Poisson — that is the structural fix CVX needs.
2. **Exogenous regressors.** ARIMAX/VAR with Brent (especially overnight, see §3), USD index, 10-yr yield, sector ETFs (XLE/SOXX/XLP). Should beat the univariate AR(1) bound especially for CVX.
3. **Longer training window.** 22 sessions is small. A 6-month rolling window with proper time-series cross-validation would let RF/GBR learn real lag structure.
4. **Volatility-scaled targets.** Predict $r_t/\hat\sigma_t$ instead of $r_t$, equalising variance across regimes.

---

## 3. Iran-US war ⇄ stock-return correlation

### 3.1 Why a single Pearson r is the wrong question

A first attempt computed Pearson r between hourly stock returns and hourly Brent returns (in-sample, n = 153). Result:

| Pearson r (hourly intraday) | CVX | NVDA | PG |
|---|---:|---:|---:|
| Brent | +0.086 | **−0.199** ★ | −0.031 |
| WTI | **+0.156** ☆ | **−0.199** ★ | −0.002 |

★ = sig at 5 %, ☆ = sig at 5.4 %. By this metric **NVDA looks more war-coupled than CVX** — surprising for an oil major and an apparent contradiction of the structural prior.

But Pearson r is one of *many* ways to measure exposure, and it captures only one specific thing: continuous bar-by-bar co-movement during NYSE open hours. **CVX trades 9:30 AM – 4 PM ET only, while Brent trades almost 24 h.** Over a 24 h cycle the bulk of Brent's price action happens *while NYSE is closed*. CVX absorbs that overnight oil move in **one shot** — its open-gap bar — and intraday trading is comparatively quiet. So intraday Pearson dilutes the CVX-Brent signal across many quiet bars and makes oil-direct exposure invisible.

### 3.2 Three structurally different exposure channels

| Channel | What it measures | Best metric to detect | Names that should score high |
|---|---|---|---|
| **A.** Continuous intraday | NYSE-hours bar-by-bar co-movement with oil | hourly Pearson r, Spearman ρ, OLS β | high-β tech / risk-asset names |
| **B.** Overnight oil-direct | absorption of overnight Brent moves through the open-gap bar | open-bar Pearson r and β vs Brent overnight | oil majors with US listings |
| **C.** Jump / event   | discrete moves on news days | excess kurtosis $\hat\gamma_2$, $\|r_{pb}\|$, $\|$max event return$\|$ | commodity-direct, headline-sensitive names |

A stock can be highly affected through one channel and undetectable through another. So for a fair "war exposure" verdict we report all three.

### 3.3 Channel A — continuous intraday co-movement (hourly, n = 153)

Pearson r and OLS β at hourly intraday resolution, in-sample 16 Mar – 15 Apr:

| Proxy | CVX | NVDA | PG |
|---|---:|---:|---:|
| **Pearson r vs Brent** | +0.086 (p=0.29) | **−0.199 (p=0.014)** | −0.031 (p=0.71) |
| **Pearson r vs WTI** | +0.156 (p=0.054) | **−0.199 (p=0.014)** | −0.002 (p=0.98) |
| **OLS β vs WTI** | +0.130 | −0.143 | −0.001 |
| **Spearman ρ vs WTI** | +0.092 | **−0.259 (p=0.001)** | −0.011 |

**Reading.** NVDA is the cleanest continuous mover with both oil benchmarks (|r| ≈ 0.20, p < 0.02). The economic interpretation is straightforward: NVDA is a high-beta risk-asset name; every Brent tick reads through to broad risk sentiment, which prices into NVDA continuously. β = −0.143 vs WTI says a 1 % WTI rally maps to ≈ −14 bps in NVDA. CVX's *absolute* β is essentially the same (+0.130 vs WTI) — equal economic sensitivity, opposite signs — but its higher own-volatility (35.9 % vs 31.1 % annualised) dilutes the *correlation coefficient* via $r = \beta\cdot\sigma_{\text{proxy}}/\sigma_{\text{stock}}$.

### 3.4 Channel B — overnight / open-gap correlation (n = 21)

For each NYSE day, compute (i) the stock's first-bar return $r^{(\text{ovn})}_d = S^{(\text{open})}_d / S^{(\text{prev close})}_{d-1} - 1$ and (ii) Brent's same-period overnight return. **This isolates the channel where US-listed names absorb 24-h Brent action in one shot.**

| Proxy | CVX | NVDA | PG |
|---|---:|---:|---:|
| **Pearson r vs Brent overnight** | **+0.716 (p = 0.0003)** ★★★ | −0.656 (p = 0.0012) ★★ | −0.089 (p = 0.70) |
| **OLS β vs Brent overnight** | **+0.347** | −0.265 | −0.017 |
| **R² vs Brent overnight** | **51 %** | 43 % | 1 % |
| **Pearson r vs WTI overnight** | **+0.696 (p = 0.0005)** | −0.655 (p = 0.0013) | −0.131 |

**This is the cleanest oil-exposure signal in the entire study.** CVX's open-gap bar correlates +0.72 with Brent's overnight return. R² = **51 %** — half of the variance of CVX's open-gap return *is* the overnight oil move. β = +0.35 says a 1 % overnight Brent rally maps to ≈ +35 bps of CVX overnight gap. Translation: when Brent jumps 5 % overnight on a Hormuz headline, CVX opens roughly +1.7 % higher — exactly the textbook oil-major behaviour. **The Pearson coefficient is also four to five times higher than the intraday number** (0.72 vs 0.156 vs WTI), confirming that the war signal at CVX lives almost entirely in the open gap, which intraday Pearson cannot see.

NVDA's overnight |r| is 0.66 — slightly *lower* than CVX in magnitude but in the *opposite* direction, because NVDA reads overnight oil as risk-off (oil up → NVDA down) just like its intraday behaviour. PG is a clean null at every horizon.

### 3.5 Channel C — jump / event exposure

Five dated escalation/de-escalation news points and the realised next-day stock returns:

| Date | Sign | Description |
|---|:--:|---|
| 2026-03-17 | +1 | Larijani killed; IRGC consolidates |
| 2026-03-18 | +1 | Israel strikes South Pars; Iran hits Qatar LNG; Hormuz disrupted |
| 2026-03-31 | −1 | Pakistan-China 5-point peace proposal |
| 2026-04-06 | +1 | Trump Hormuz-reopen ultimatum deadline |
| 2026-04-07 | −1 | Two-week ceasefire announced; Hormuz to reopen |

Point-biserial: $r_{pb} = \dfrac{\bar r_{esc} - \bar r_{de}}{s_r}\sqrt{\dfrac{n_1 n_0}{n^2}}$.

| Company | $\bar r$ on escalation | $\bar r$ on de-escalation | $r_{pb}$ | max $\|$event return$\|$ | Excess kurtosis $\hat\gamma_2$ |
|---|---:|---:|---:|---:|---:|
| **CVX**  | **+0.30 %** | −0.27 % | +0.27 | 1.83 % | **+14.31** |
| **NVDA** | −0.46 %    | +2.95 % | **−0.69** | **+5.65 %** | +5.03 |
| **PG**   | −1.28 %    | −0.61 % | −0.30 | 3.16 % | +5.42 |

Two of three jump indicators give NVDA the larger response (driven by its +5.65 % rally on the 31-Mar peace plan). Excess kurtosis, the broadest distributional measure of jump-frequency, gives **CVX a 3× lead over NVDA** (14.3 vs 5.0). **The signs are exactly as predicted by the macro story**: positive for the oil major (escalation good for CVX), large negative for the tech name (escalation bad for NVDA), small negative for the staple. Statistical significance is limited by n = 5 events.

### 3.6 Composite war-exposure score

To compare like-for-like across channels I compute the **z-score** of each company on each channel input, average within a channel, then average across channels. With three companies the z-score lives in $[-1.22, +1.22]$.

| | Channel A — continuous intraday | Channel B — overnight oil-direct | Channel C — jump / event | **Composite (z-score average)** | Rank |
|---|---:|---:|---:|---:|:--:|
| **CVX**  | +0.205 | **+0.706** ★ | −0.123 | **+0.263** | **2** |
| **NVDA** | **+0.843** ★ | +0.435 | **+0.541** ★ | **+0.606** | **1** |
| **PG**   | −1.048 | −1.141 | −0.419 | **−0.869** | 3 |

★ = winner of the channel.

**Reading the table — directly answering "is CVX really less affected than NVDA?".**

- **By Channel A (continuous co-movement) — NVDA wins.** Higher Pearson, higher Spearman, higher β. NVDA's risk-asset transmission is real and broad: every Brent tick during NYSE hours reads continuously into broad risk sentiment that prices into NVDA.
- **By Channel B (overnight oil-direct) — CVX wins, decisively.** Open-gap r = +0.72 and R² = 51 % is the highest single correlation coefficient anywhere in this study. **This is the channel that most cleanly measures oil-direct exposure**, and CVX dominates it. NVDA's overnight |r| is only slightly lower (0.66) but inverted in sign — confirming NVDA's oil exposure is *general risk-asset*, not oil-direct.
- **By Channel C (jump / event) — split.** NVDA wins on |r_pb| and on max-event-return, driven by the single +5.65 % peace-plan rally. CVX wins decisively on excess kurtosis (14.3 vs 5.0) — a more comprehensive measure of jump-frequency that doesn't depend on which 5 events you pick. With our specific event list NVDA has the larger Channel C score.
- **Composite — NVDA edges CVX (0.61 vs 0.26).** The margin is small (0.35 standard deviations of cross-sectional variation). NVDA wins because the channel A and C gaps in NVDA's favour are slightly larger than the channel B gap in CVX's favour. Both stocks are *highly affected* — they simply absorb war risk through different transmission channels.

### 3.7 So what's the verdict?

The structural prior — *"CVX is more war-affected than NVDA because it's an oil major"* — is empirically supported on the channel that most directly measures oil exposure (B: overnight, R² = 51 %), but is *slightly contradicted* on the broader composite because NVDA's exposure as a risk-asset name is itself substantial, and that exposure manifests in the metrics (continuous intraday Pearson, event-day magnitude) more than CVX's does. Specifically, in the 21-day window 16 Mar – 15 Apr 2026:

- **CVX is the most oil-direct name** (Channel B winner; r = +0.72 with overnight Brent, R² = 51 %, β = +0.35; excess kurtosis +14.3 — highest in the panel).
- **NVDA is the most continuously-priced name** (Channel A winner; r = −0.20 with intraday Brent, p = 0.014; the strongest event-day reaction in the panel at +5.65 % on the peace plan).
- **PG is genuinely insulated** (z-score is the lowest across every channel; overnight Pearson with Brent is statistically zero).

So *yes*, CVX scores marginally lower than NVDA on the **composite** war-exposure ranking — but it scores **higher** on the cleanest single oil-direct measure. The honest one-line answer to "is CVX really less affected than NVDA?" is: *both are highly affected; NVDA's exposure shows up as a continuous price-in mechanism while CVX's lives in the overnight gap; on a 3-channel composite NVDA edges CVX, on the Channel-B oil-direct measure CVX wins decisively.* See `comparison_war_exposure_score.csv` and `comparison_war_exposure_score.png` for the full table and plot.

To strengthen the result: (i) longer sample (a year+, $n \gtrsim 100$ days) for tighter confidence intervals on every channel; (ii) the **Geopolitical Risk Index** (Caldara-Iacoviello) as a continuous regressor instead of a 5-event dummy; (iii) GARCH-filtered residual correlations to remove volatility bleed-through; (iv) Newey-West HAC standard errors on β.

---

## 4. Cross-company summary

| | **CVX** | **NVDA** | **PG** |
|---|---:|---:|---:|
| Annualised vol | **35.9 %** | 31.1 % | **16.2 %** |
| Skew / Excess-kurt | **−2.54 / +14.31** | +1.07 / +5.03 | −0.15 / +5.42 |
| Total return (window) | −6.4 % | **+8.0 %** | −5.3 % |
| Best forecaster (RMSE) | Naive ≈ ARIMA (flat) | **GBM** (drift continued) | **ARIMA(2,1,2)** (real structure) |
| Best directional model | **Ridge (0.58)** | **GBR (0.65)** | RF / GBR (0.56) |
| ARIMA chosen order | (0,1,0) | (0,1,0) | **(2,1,2)** |
| Pearson r vs WTI (intraday hourly) | +0.156 | **−0.199** ★ | −0.002 |
| **Pearson r vs Brent (overnight gap)** | **+0.716** ★★★ | −0.656 ★★ | −0.089 |
| OLS β vs Brent (overnight gap) | **+0.347** | −0.265 | −0.017 |
| Event-study $r_{pb}$ | +0.27 | **−0.69** | −0.30 |
| **Composite war-exposure z-score** | +0.26 (rank 2) | **+0.61 (rank 1)** | −0.87 (rank 3) |

**Overall narrative.** The three companies sit cleanly along the war-exposure axis but **through different transmission channels**, each requiring a different metric to detect. CVX's exposure lives in the **overnight gap** and the **excess kurtosis**: Brent moves while NYSE sleeps, then CVX absorbs the move in its first hour at a β of +0.35, with R² = 51 %. NVDA's exposure lives in **continuous intraday co-movement** (β = −0.14 against WTI hourly, p = 0.014) and in **dramatic single-event responses** (+5.65 % on peace, the largest event-day move in the panel). PG is detectably uncorrelated with every war proxy.

**The single most-important methodological lesson is this:** a single Pearson r at one resolution is *the wrong question* for a stock with structural exposure. We needed three distinct channels (intraday, overnight, jump) to recover the truth that CVX *is* the most oil-direct name (Channel B, R² = 51 %) even though NVDA *is* the more continuously-priced name (Channel A). On forecasting, the best model varies by company: Naive/ARIMA(0,1,0) wins when prices random-walk (CVX); GBM/Drift wins when prices trend (NVDA); ARIMA with non-trivial order wins when serial structure is present (PG).

---

## Appendix — Files in this submission

| File | Purpose |
|---|---|
| `task2_pipeline.py` | The single-file analysis pipeline (source of truth) |
| `Task2_Report.md` | This report |
| `CVX.csv`, `NVDA.csv`, `PG.csv` | Hourly OHLC for the three names (cached from Yahoo) |
| `BZ_F.csv`, `CL_F.csv`, `LMT.csv` | Geopolitical proxy hourly series |
| `{TICKER}_hourly_forecasts.csv` | All 8 model price predictions on the test window + GBM 95 % band |
| `{TICKER}_forecast_metrics.csv` | RMSE / MAE / MAPE / DirAcc / period-mean per model |
| `{TICKER}_war_correlation.csv` | Pearson / Spearman / OLS β at hourly *and* daily resolution |
| `{TICKER}_overnight_correlation.csv` | **NEW** — open-gap correlation with Brent and WTI overnight returns |
| `{TICKER}_brent_leadlag.csv` | Hourly lead/lag table with Brent |
| `{TICKER}_event_study.csv` | Event-day returns and point-biserial inputs |
| `{TICKER}_distribution.png` | 1×3: histogram with N(μ,σ) overlay + Q-Q plot + box plot — **all three measure normality** |
| `{TICKER}_forecast.png` | One continuous price plot: in-sample close + OLS trend + out-of-sample actual + 8 forecasts + GBM 95 % band |
| `{TICKER}_war_scatter.png` | 1×2: Channel-A intraday hourly scatter + Channel-B overnight open-gap scatter, both vs Brent |
| `comparison_stats.csv`, `comparison_stats.png` | Cross-company distributional moments (skew / excess kurt / annualised vol) |
| `comparison_forecast_*.csv` | Cross-company model × company tables (RMSE, MAE, MAPE, DirAcc, Pred_Mean, Mean_Error) |
| `comparison_forecast.png` | Side-by-side price-forecast plots |
| `comparison_war_*.csv`, `comparison_war.png` | Cross-company war-correlation tables; figure groups Channel-A and Channel-B Pearson r side-by-side, with Channel-C event-study on a separate panel |
| **`comparison_war_exposure_score.csv`, `comparison_war_exposure_score.png`** | **The three-channel composite war-exposure score** — the headline answer to "who is most affected" |
