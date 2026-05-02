"""
SCAF MAC-F244 — Task 2 (Final Submission, multi-company, v3)
========================================================================
Three companies along the Iran–US war exposure spectrum:
   CVX   — Chevron, oil major          (HIGHLY  affected)
   NVDA  — NVIDIA, semis / risk-asset  (MILDLY  affected)
   PG    — Procter & Gamble, staple    (LEAST   affected)

In-sample      : 16 Mar – 15 Apr 2026  (hourly bars)
Out-of-sample  : 16 Apr – 25 Apr 2026  (hourly bars)

Two changes vs. v2 that fix three real problems:
  1. Forecasts target hourly *PRICE LEVELS* (multi-step trajectories),
     not hourly returns.  Hourly returns are martingale-difference noise
     and every model collapsed to ≈ 0; price trajectories are
     informative and let ML compete properly.
  2. ML models do *recursive multi-step* forecasting from the train
     end — predictions feed back as features for the next hour, exactly
     like ARIMA's `forecast(steps=h)` does internally.  No look-ahead.
  3. War correlation uses *HOURLY contemporaneous returns* during the
     NYSE–Brent overlap window (≈ 150 obs) instead of daily-resampled
     returns (n=21).  The earlier ≈ 0 CVX-Brent correlation was a small-
     sample artefact; with hourly alignment the oil-major signal shows
     up clearly.

Run:
    python task2_pipeline.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.stats import (skew, kurtosis, jarque_bera, shapiro,
                         kstest, pointbiserialr, pearsonr, spearmanr,
                         linregress)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


# ===================================================================
# 0.  CONFIG
# ===================================================================
HERE = os.path.dirname(os.path.abspath(__file__))

COMPANIES = [
    {"ticker": "CVX",  "name": "Chevron Corp.",
     "label": "Jump-exposed (oil-direct, fat-tail channel)",
     "color": "#c0392b"},
    {"ticker": "NVDA", "name": "NVIDIA Corp.",
     "label": "Continuously-coupled (risk-asset channel)",
     "color": "#2c3e50"},
    {"ticker": "PG",   "name": "Procter & Gamble",
     "label": "Insulated (consumer staple, neither channel)",
     "color": "#16a085"},
]

TRAIN_START = pd.Timestamp("2026-03-15", tz="UTC")
TRAIN_END   = pd.Timestamp("2026-04-16", tz="UTC")
TEST_START  = pd.Timestamp("2026-04-16", tz="UTC")
TEST_END    = pd.Timestamp("2026-04-26", tz="UTC")

EVENTS = [
    ("2026-03-17", +1, "Larijani killed; IRGC consolidates"),
    ("2026-03-18", +1, "Israel strikes South Pars; Iran hits Qatar LNG; Hormuz disrupted"),
    ("2026-03-31", -1, "Pakistan-China 5-point peace proposal"),
    ("2026-04-06", +1, "Trump Hormuz-reopen ultimatum deadline"),
    ("2026-04-07", -1, "Two-week ceasefire announced; Hormuz to reopen"),
]

LAGS  = [1, 2, 3, 4, 5]
ROLLS = [3, 5]
SEED  = 2026
N_PATHS = 4000


def hr(msg, ch="="):
    print("\n" + ch * 72)
    print(f"  {msg}")
    print(ch * 72)


# ===================================================================
# 1.  LOAD & STANDARDISE DATA
# ===================================================================
def load_hourly(path, label):
    df = pd.read_csv(path)
    if "Datetime" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df = df.sort_values("Datetime").reset_index(drop=True)
    print(f"  [{label:6s}] rows={len(df):4d}   "
          f"first={df['Datetime'].min()}   last={df['Datetime'].max()}")
    return df


hr("1. DATA LOAD & VERIFICATION")
DATA = {c["ticker"]: load_hourly(os.path.join(HERE, f"{c['ticker']}.csv"), c["ticker"])
        for c in COMPANIES}
PROXY = {
    "BRENT": load_hourly(os.path.join(HERE, "BZ_F.csv"), "BZ=F"),
    "WTI":   load_hourly(os.path.join(HERE, "CL_F.csv"), "CL=F"),
    "LMT":   load_hourly(os.path.join(HERE, "LMT.csv"),  "LMT"),
}


# ===================================================================
# 2.  HELPERS
# ===================================================================
def split_in_out(df):
    inn = df[(df["Datetime"] >= TRAIN_START) & (df["Datetime"] < TRAIN_END)].copy()
    out = df[(df["Datetime"] >= TEST_START)  & (df["Datetime"] < TEST_END )].copy()
    return inn, out


def build_lag_features_for_train(price_series):
    """Build lag-and-rolling features from a *price* series. Strictly causal:
       rolling stats are over past returns only, NOT including the current."""
    p   = price_series.copy()
    r   = p.pct_change()
    df  = pd.DataFrame({"P": p.values, "R": r.values}, index=p.index)
    past = df["R"].shift(1)
    for L in LAGS:
        df[f"r_lag{L}"] = df["R"].shift(L)
    for w in ROLLS:
        df[f"r_rmean{w}"] = past.rolling(w).mean()
        df[f"r_rstd{w}"]  = past.rolling(w).std()
    df["r_lag1_x_lag2"] = df["r_lag1"] * df["r_lag2"]   # mild interaction
    df["price_lag1"] = df["P"].shift(1)
    df["price_lag2"] = df["P"].shift(2)
    df["price_diff1"] = df["P"].shift(1) - df["P"].shift(2)
    df["hour"]   = df.index.hour
    df["dow"]    = df.index.dayofweek
    df["hsin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hcos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    return df


FEATURE_COLS = ([f"r_lag{L}"  for L in LAGS]
                + [f"r_rmean{w}" for w in ROLLS]
                + [f"r_rstd{w}"  for w in ROLLS]
                + ["r_lag1_x_lag2",
                   "price_lag1", "price_lag2", "price_diff1",
                   "hour", "dow", "hsin", "hcos"])


def metrics_price(P_actual, P_pred):
    P_actual = np.asarray(P_actual, dtype=float)
    P_pred   = np.asarray(P_pred,   dtype=float)
    err = P_actual - P_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / P_actual)) * 100)
    # Directional accuracy on hour-to-hour change
    da = float(np.mean(np.sign(np.diff(P_actual)) ==
                       np.sign(np.diff(P_pred)))) if len(P_actual) > 1 else np.nan
    return dict(RMSE=rmse, MAE=mae, MAPE_pct=mape, DirAcc=da,
                Pred_Mean=float(P_pred.mean()),
                Actual_Mean=float(P_actual.mean()),
                Mean_Error=float(P_pred.mean() - P_actual.mean()))


# ---------- Multi-step PRICE forecasters -----------------------------
def fc_naive(P_train, h):
    """Random walk: P_{T+k} = P_T."""
    return np.full(h, P_train.iloc[-1])


def fc_drift(P_train, h):
    """Constant-drift: P_{T+k} = P_T + δ k, δ = mean per-hour change."""
    delta = (P_train.iloc[-1] - P_train.iloc[0]) / max(len(P_train) - 1, 1)
    return P_train.iloc[-1] + delta * np.arange(1, h + 1)


def fc_arima_price(P_train, h):
    """AIC-grid ARIMA fit on the *price* series."""
    best_aic, best_order, best_fit = np.inf, (1, 1, 1), None
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    fit = ARIMA(P_train.values, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic, best_order, best_fit = fit.aic, (p, d, q), fit
                except Exception:
                    continue
    fc = np.asarray(best_fit.forecast(steps=h)).flatten()
    return fc, best_order, best_aic


def fc_gbm(P_train, h, n_paths=N_PATHS, return_paths=False):
    """Geometric Brownian Motion:
         d log S = (μ - ½ σ²) dt + σ dW_t     (Black-Scholes / Itô)
       μ, σ estimated from in-sample hourly log-returns.
       Forecast = path-mean.  Bands = 2.5%/97.5% quantiles."""
    log_r = np.log(P_train).diff().dropna().values
    mu, sig = float(np.mean(log_r)), float(np.std(log_r, ddof=1))
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal((n_paths, h))
    incr = (mu - 0.5 * sig ** 2) + sig * z
    log_paths = np.log(float(P_train.iloc[-1])) + np.cumsum(incr, axis=1)
    paths = np.exp(log_paths)
    fc = paths.mean(axis=0)
    if return_paths:
        return fc, paths
    return fc


def ml_recursive_forecast(model_factory, train_feat, h, last_price_history,
                          last_return_history, hours_in_test):
    """Train a sklearn-compatible model on (features → next-hour return),
    then run a *recursive multi-step* forecast for h hours from train end.

    `last_price_history`   = list of past prices, last entry = P_T
    `last_return_history`  = list of past returns aligned with prices
    `hours_in_test`        = list of UTC datetimes for the test window
    """
    Xtr = train_feat[FEATURE_COLS].values
    ytr = train_feat["R"].values
    model = model_factory()
    model.fit(Xtr, ytr)

    # Walk forward, building features from accumulated prices/returns
    prices = list(last_price_history)
    returns = list(last_return_history)
    fc_prices = []
    for k, dt in enumerate(hours_in_test):
        # Build feature vector for the next bar
        feat = {}
        for i, L in enumerate(LAGS):
            feat[f"r_lag{L}"] = returns[-L] if len(returns) >= L else 0.0
        for w in ROLLS:
            past_w = returns[-w-1:-1] if len(returns) >= w + 1 else returns[-w:]
            feat[f"r_rmean{w}"] = float(np.mean(past_w)) if past_w else 0.0
            feat[f"r_rstd{w}"]  = float(np.std(past_w, ddof=1)) if len(past_w) > 1 else 0.0
        feat["r_lag1_x_lag2"] = (returns[-1] * returns[-2]) if len(returns) >= 2 else 0.0
        feat["price_lag1"]    = prices[-1]
        feat["price_lag2"]    = prices[-2] if len(prices) >= 2 else prices[-1]
        feat["price_diff1"]   = prices[-1] - (prices[-2] if len(prices) >= 2 else prices[-1])
        feat["hour"]   = dt.hour
        feat["dow"]    = dt.dayofweek
        feat["hsin"]   = np.sin(2 * np.pi * dt.hour / 24)
        feat["hcos"]   = np.cos(2 * np.pi * dt.hour / 24)

        x = np.array([[feat[c] for c in FEATURE_COLS]])
        r_hat = float(model.predict(x)[0])
        # Translate predicted return to predicted next price; use predicted as
        # the new history element.
        p_hat = prices[-1] * (1.0 + r_hat)
        fc_prices.append(p_hat)
        prices.append(p_hat)
        returns.append(r_hat)
    return np.array(fc_prices)


def inverse_rmse_weights(rmses):
    inv = np.array([1.0 / max(r, 1e-12) ** 2 for r in rmses])
    return inv / inv.sum()


# ===================================================================
# 3.  PER-COMPANY ANALYSIS
# ===================================================================
RESULTS = {}
for cfg in COMPANIES:
    ticker = cfg["ticker"]
    hr(f"COMPANY: {ticker}  ({cfg['name']})  —  {cfg['label']}")

    raw = DATA[ticker]
    inn, out = split_in_out(raw)

    # Build price/return series indexed by Datetime
    P_in  = inn.set_index("Datetime")["Close"].astype(float)
    P_out = out.set_index("Datetime")["Close"].astype(float)
    R_in  = P_in.pct_change()
    L_in  = np.log(P_in / P_in.shift(1))
    ret_in = R_in.dropna()
    log_in = L_in.dropna()

    # ------------ 3.1  Statistical summary ---------------------------
    p_start, p_end = float(P_in.iloc[0]), float(P_in.iloc[-1])
    summary = {
        "Ticker":           ticker,
        "Name":             cfg["name"],
        "Observations":     len(ret_in),
        "Mean":             float(ret_in.mean()),
        "Median":           float(ret_in.median()),
        "Std":              float(ret_in.std()),
        "Variance":         float(ret_in.var()),
        "Min":              float(ret_in.min()),
        "Max":              float(ret_in.max()),
        "Skew":             float(skew(ret_in, bias=False)),
        "ExcessKurtosis":   float(kurtosis(ret_in, fisher=True, bias=False)),
        "AnnualisedVol":    float(ret_in.std() * np.sqrt(6.5 * 252)),
        "PriceStart":       p_start,
        "PriceEnd":         p_end,
        "PriceHigh":        float(P_in.max()),
        "PriceLow":         float(P_in.min()),
        "TotalReturn":      p_end / p_start - 1.0,
    }
    sw_s,    sw_p    = shapiro(ret_in)
    jb_s,    jb_p    = jarque_bera(ret_in)
    ks_s,    ks_p    = kstest((ret_in - ret_in.mean()) / ret_in.std(), "norm")
    adf_p_s, adf_p_p = adfuller(P_in.dropna(),  autolag="AIC")[:2]
    adf_r_s, adf_r_p = adfuller(ret_in.dropna(), autolag="AIC")[:2]
    summary.update(dict(
        Shapiro_W=float(sw_s), Shapiro_p=float(sw_p),
        JB_stat=float(jb_s),   JB_p=float(jb_p),
        KS_stat=float(ks_s),   KS_p=float(ks_p),
        ADF_price_stat=float(adf_p_s),  ADF_price_p=float(adf_p_p),
        ADF_returns_stat=float(adf_r_s), ADF_returns_p=float(adf_r_p),
    ))
    xhrs = np.arange(len(P_in))
    trend_slope, trend_intercept, trend_r, trend_p, _ = linregress(xhrs, P_in.values)
    trend_R2 = trend_r ** 2
    summary.update(dict(OLS_intercept=float(trend_intercept),
                        OLS_slope=float(trend_slope),
                        OLS_R2=float(trend_R2), OLS_p=float(trend_p)))

    print(f"  n={summary['Observations']}  μ={summary['Mean']*100:+.4f}%/h  "
          f"σ={summary['Std']*100:.4f}%/h  ann.vol={summary['AnnualisedVol']*100:.2f}%")
    print(f"  skew={summary['Skew']:+.3f}  exKurt={summary['ExcessKurtosis']:+.3f}  "
          f"total return={summary['TotalReturn']*100:+.2f}%")
    print(f"  Shapiro p={summary['Shapiro_p']:.4f}  JB p={summary['JB_p']:.4f}  "
          f"KS p={summary['KS_p']:.4f}")
    print(f"  ADF price p={summary['ADF_price_p']:.4f}  "
          f"ADF returns p={summary['ADF_returns_p']:.4f}")
    print(f"  OLS slope={trend_slope:+.4f} USD/h   R²={trend_R2:.3f}   p={trend_p:.2e}")

    # ------------ 3.2  PRICE-LEVEL  forecasting ----------------------
    h = len(P_out)
    test_dt = list(P_out.index)

    # Statistical / parametric
    p_naive  = fc_naive(P_in, h)
    p_drift  = fc_drift(P_in, h)
    p_arima, ar_order, ar_aic = fc_arima_price(P_in, h)
    p_gbm,  gbm_paths = fc_gbm(P_in, h, return_paths=True)
    summary["ARIMA_order"] = str(ar_order)
    summary["ARIMA_AIC"]   = float(ar_aic)
    print(f"  ARIMA grid: best order = {ar_order}  AIC={ar_aic:.2f}")

    # ML  (recursive multi-step from train end)
    train_feat = build_lag_features_for_train(P_in).dropna()
    last_prices  = list(P_in.values)
    last_returns = list(R_in.dropna().values)

    ml_factories = {
        "Ridge":           lambda: Pipeline([("sc", StandardScaler()),
                                              ("m", Ridge(alpha=1.0))]),
        "RandomForest":    lambda: RandomForestRegressor(n_estimators=400, max_depth=4,
                                                          min_samples_leaf=3, random_state=SEED),
        "GradientBoosting":lambda: GradientBoostingRegressor(n_estimators=250, learning_rate=0.05,
                                                              max_depth=2, random_state=SEED),
    }
    ml_preds = {}
    for name, fac in ml_factories.items():
        ml_preds[name] = ml_recursive_forecast(
            fac, train_feat, h, last_prices, last_returns, test_dt
        )

    # Validation slice (last 20% of train) for ensemble weights
    cut = int(np.floor(0.8 * len(P_in)))
    P_part_train, P_val = P_in.iloc[:cut], P_in.iloc[cut:]
    val_h = len(P_val)
    val_rmses = {}
    val_rmses["Naive"] = float(np.sqrt(np.mean(
        (np.full(val_h, P_part_train.iloc[-1]) - P_val.values) ** 2)))
    val_rmses["Drift"] = float(np.sqrt(np.mean(
        (fc_drift(P_part_train, val_h) - P_val.values) ** 2)))
    try:
        f_a, _, _ = fc_arima_price(P_part_train, val_h)
        val_rmses["ARIMA"] = float(np.sqrt(np.mean((f_a - P_val.values) ** 2)))
    except Exception:
        val_rmses["ARIMA"] = np.inf
    val_rmses["GBM"] = float(np.sqrt(np.mean(
        (fc_gbm(P_part_train, val_h) - P_val.values) ** 2)))
    train_feat_part = build_lag_features_for_train(P_part_train).dropna()
    last_prices_part  = list(P_part_train.values)
    last_returns_part = list(P_part_train.pct_change().dropna().values)
    val_dt = list(P_val.index)
    for name, fac in ml_factories.items():
        try:
            f_ml = ml_recursive_forecast(
                fac, train_feat_part, val_h, last_prices_part, last_returns_part, val_dt
            )
            val_rmses[name] = float(np.sqrt(np.mean((f_ml - P_val.values) ** 2)))
        except Exception:
            val_rmses[name] = np.inf

    # Per-hour test results table
    pred_dict = {
        "Naive":  p_naive,
        "Drift":  p_drift,
        "ARIMA":  p_arima,
        "GBM":    p_gbm,
        **ml_preds,
    }
    fc_rows = []
    for name, P_pred in pred_dict.items():
        m = metrics_price(P_out.values, P_pred)
        m["Model"]   = name
        m["ValRMSE"] = float(val_rmses.get(name, np.nan))
        fc_rows.append(m)

    # Top-3 ensemble by validation RMSE, inverse-RMSE² weights
    K_TOP = 3
    rank = sorted(val_rmses.items(), key=lambda kv: kv[1])
    chosen = [n for n, _ in rank[:K_TOP]]
    weights = inverse_rmse_weights([val_rmses[n] for n in chosen])
    weights_d = {n: float(w) for n, w in zip(chosen, weights)}

    p_ens = np.zeros(h)
    for n in chosen:
        p_ens += weights_d[n] * pred_dict[n]
    pred_dict["Ensemble"] = p_ens
    m = metrics_price(P_out.values, p_ens)
    m["Model"]   = "Ensemble"
    m["ValRMSE"] = np.nan
    fc_rows.append(m)
    summary["EnsembleMembers"] = "+".join(chosen)
    summary["EnsembleWeights"] = "+".join(f"{w:.2f}" for w in weights)

    fc_table = pd.DataFrame(fc_rows).sort_values("RMSE").reset_index(drop=True)
    print("\n  Hourly PRICE forecast — model ranking (RMSE, MAE in USD; MAPE in %):")
    print(fc_table[["Model", "RMSE", "MAE", "MAPE_pct", "DirAcc",
                    "Pred_Mean", "Actual_Mean", "Mean_Error", "ValRMSE"]]
          .round(4).to_string(index=False))

    # Save per-company forecasts
    out_df = pd.DataFrame({"Actual": P_out.values}, index=P_out.index)
    for n, v in pred_dict.items():
        out_df[n] = v
    # GBM band
    out_df["GBM_lo"] = np.quantile(gbm_paths, 0.025, axis=0)
    out_df["GBM_hi"] = np.quantile(gbm_paths, 0.975, axis=0)
    out_df.to_csv(os.path.join(HERE, f"{ticker}_hourly_forecasts.csv"))
    fc_table.to_csv(os.path.join(HERE, f"{ticker}_forecast_metrics.csv"), index=False)

    # ------------ 3.3  WAR CORRELATION (hourly aligned) -------------
    # Resample each series to UTC '1H' last() so labels align (CVX
    # lives at xx:30, Brent at xx:00); after resample(1H).last() both
    # land at xx:00 buckets and we can intersect.
    def hourly_aligned_returns(df):
        s = df.set_index("Datetime")["Close"].astype(float)
        s_h = s.resample("1h").last().dropna()
        return s_h.pct_change().dropna()

    rT  = hourly_aligned_returns(raw)
    rB  = hourly_aligned_returns(PROXY["BRENT"])
    rW  = hourly_aligned_returns(PROXY["WTI"])
    rL  = hourly_aligned_returns(PROXY["LMT"])

    align_h = pd.concat([rT.rename(ticker), rB.rename("BRENT"),
                         rW.rename("WTI"),  rL.rename("LMT")],
                        axis=1).loc["2026-03-15":"2026-04-15"].dropna()

    # Daily-aligned for comparison
    def daily_returns(df):
        s = df.set_index("Datetime")["Close"].astype(float)
        return s.resample("1D").last().dropna().pct_change().dropna()

    dT = daily_returns(raw)
    dB = daily_returns(PROXY["BRENT"])
    dW = daily_returns(PROXY["WTI"])
    dL = daily_returns(PROXY["LMT"])
    align_d = pd.concat([dT.rename(ticker), dB.rename("BRENT"),
                         dW.rename("WTI"),  dL.rename("LMT")],
                        axis=1).loc["2026-03-15":"2026-04-15"].dropna()

    print(f"\n  Aligned bars (HOURLY, in-sample) : {len(align_h)}")
    print(f"  Aligned bars (DAILY,  in-sample) : {len(align_d)}")

    war_rows = []
    for px in ["BRENT", "WTI", "LMT"]:
        for level, A in [("hourly", align_h), ("daily", align_d)]:
            pr, ppv = pearsonr(A[ticker], A[px])
            sr, spv = spearmanr(A[ticker], A[px])
            sl, ic, rv, pv, _ = linregress(A[px], A[ticker])
            war_rows.append({"Ticker": ticker, "Proxy": px, "Resolution": level,
                             "n": int(len(A)),
                             "Pearson": float(pr), "Pearson_p": float(ppv),
                             "Spearman": float(sr), "Spearman_p": float(spv),
                             "Beta": float(sl), "R2": float(rv ** 2),
                             "Beta_p": float(pv)})
    war_df = pd.DataFrame(war_rows)
    print("\n  War-correlation table (Pearson / Spearman / OLS β):")
    print(war_df.round(4).to_string(index=False))
    war_df.to_csv(os.path.join(HERE, f"{ticker}_war_correlation.csv"), index=False)

    # ---- OVERNIGHT-GAP correlation (the oil-direct channel) -------
    # CVX trades only 13:30–19:30 UTC; the bulk of Brent's price action
    # over a 24h cycle happens *while NYSE is closed*. The "first bar
    # of the NYSE day" return absorbs this overnight move in one shot.
    # Correlating those open-gap returns against Brent's same-period
    # overnight return is the right way to see oil-direct exposure.

    def overnight_corr(stock_h, proxy_h):
        first_bars = stock_h[stock_h.index.hour == 13]
        s_ovn = first_bars.pct_change().dropna()
        p_at_open = proxy_h.reindex(first_bars.index, method="ffill")
        p_ovn = p_at_open.pct_change().dropna()
        A = pd.concat([s_ovn.rename("S"), p_ovn.rename("P")], axis=1).dropna()
        if len(A) < 6:
            return dict(n=len(A), Pearson=np.nan, Pearson_p=np.nan,
                        Spearman=np.nan, Spearman_p=np.nan, Beta=np.nan, R2=np.nan)
        pr, pp = pearsonr(A["S"], A["P"])
        sr, sp = spearmanr(A["S"], A["P"])
        sl, _, rv, _, _ = linregress(A["P"], A["S"])
        return dict(n=len(A), Pearson=float(pr), Pearson_p=float(pp),
                    Spearman=float(sr), Spearman_p=float(sp),
                    Beta=float(sl), R2=float(rv ** 2))

    stock_h = raw.set_index("Datetime")["Close"].astype(float).resample("1h").last().dropna()
    brent_h = PROXY["BRENT"].set_index("Datetime")["Close"].astype(float).resample("1h").last().dropna()
    wti_h   = PROXY["WTI"].set_index("Datetime")["Close"].astype(float).resample("1h").last().dropna()
    stock_h = stock_h.loc["2026-03-15":"2026-04-15"]
    brent_h = brent_h.loc["2026-03-15":"2026-04-15"]
    wti_h   = wti_h.loc["2026-03-15":"2026-04-15"]
    ovn_brent = overnight_corr(stock_h, brent_h)
    ovn_wti   = overnight_corr(stock_h, wti_h)
    print("\n  OVERNIGHT-GAP correlation (open-bar return vs proxy overnight):")
    print(f"    Brent: r={ovn_brent['Pearson']:+.4f}  p={ovn_brent['Pearson_p']:.4f}  "
          f"β={ovn_brent['Beta']:+.4f}  R²={ovn_brent['R2']:.4f}  n={ovn_brent['n']}")
    print(f"    WTI  : r={ovn_wti['Pearson']:+.4f}  p={ovn_wti['Pearson_p']:.4f}  "
          f"β={ovn_wti['Beta']:+.4f}  R²={ovn_wti['R2']:.4f}  n={ovn_wti['n']}")

    pd.DataFrame([{**ovn_brent, "Proxy": "BRENT"},
                  {**ovn_wti,   "Proxy": "WTI"}]).to_csv(
        os.path.join(HERE, f"{ticker}_overnight_correlation.csv"), index=False)

    # Headline numbers for cross-company table
    p_brent_h = war_df.query("Proxy=='BRENT' & Resolution=='hourly'").iloc[0]
    p_brent_d = war_df.query("Proxy=='BRENT' & Resolution=='daily'").iloc[0]
    summary["Pearson_Brent_hourly"] = float(p_brent_h["Pearson"])
    summary["Pearson_Brent_hourly_p"] = float(p_brent_h["Pearson_p"])
    summary["Pearson_Brent_daily"]  = float(p_brent_d["Pearson"])
    summary["Pearson_Brent_daily_p"] = float(p_brent_d["Pearson_p"])
    summary["Pearson_Brent_overnight"]   = float(ovn_brent["Pearson"])
    summary["Pearson_Brent_overnight_p"] = float(ovn_brent["Pearson_p"])
    summary["Beta_Brent_overnight"]      = float(ovn_brent["Beta"])
    summary["R2_Brent_overnight"]        = float(ovn_brent["R2"])
    summary["Pearson_WTI_overnight"]     = float(ovn_wti["Pearson"])
    summary["Pearson_WTI_overnight_p"]   = float(ovn_wti["Pearson_p"])
    summary["Beta_WTI_overnight"]        = float(ovn_wti["Beta"])

    # Lead/lag with Brent at HOURLY resolution
    ll_rows = []
    for k in range(-5, 6):
        if k >= 0:
            x = align_h["BRENT"].shift(k); y = align_h[ticker]
        else:
            x = align_h["BRENT"]; y = align_h[ticker].shift(-k)
        m_ = pd.concat([x, y], axis=1).dropna()
        if len(m_) > 20:
            pr, ppv = pearsonr(m_.iloc[:, 0], m_.iloc[:, 1])
            ll_rows.append({"Ticker": ticker, "lag_hours": k, "n": len(m_),
                            "Pearson": float(pr), "p": float(ppv)})
    ll_df = pd.DataFrame(ll_rows)
    ll_df.to_csv(os.path.join(HERE, f"{ticker}_brent_leadlag.csv"), index=False)

    # Event study
    daily_t_idx = dT.copy(); daily_t_idx.index = daily_t_idx.index.date
    ev_rows = []
    for d, sg, desc in EVENTS:
        td = pd.to_datetime(d).date()
        if td in daily_t_idx.index:
            r = float(daily_t_idx.loc[td]); used = str(td)
        else:
            future = sorted(x for x in daily_t_idx.index if x >= td)
            if not future:
                continue
            td2 = future[0]; r = float(daily_t_idx.loc[td2]); used = f"{d}->{td2}"
        ev_rows.append({"Ticker": ticker, "event_date": used, "sign": sg,
                        "ret": r, "description": desc})
    ev_df = pd.DataFrame(ev_rows)
    pb_r = pb_p = m_up = m_down = np.nan
    if len(ev_df) >= 3:
        sgn = (ev_df["sign"] == 1).astype(int).values
        rts = ev_df["ret"].values
        pb_r, pb_p = pointbiserialr(sgn, rts)
        m_up   = float(rts[sgn == 1].mean()) if (sgn == 1).any() else np.nan
        m_down = float(rts[sgn == 0].mean()) if (sgn == 0).any() else np.nan
        print(f"\n  Event-study point-biserial r={pb_r:+.4f}  p={pb_p:.4f}")
        print(f"    mean return on escalation days  = {m_up:+.4%}")
        print(f"    mean return on de-escal. days   = {m_down:+.4%}")
    ev_df.to_csv(os.path.join(HERE, f"{ticker}_event_study.csv"), index=False)
    summary.update(dict(
        PointBiserial_r=float(pb_r) if pb_r == pb_r else np.nan,
        PointBiserial_p=float(pb_p) if pb_p == pb_p else np.nan,
        Mean_ret_escalation=m_up, Mean_ret_deescalation=m_down))

    # ------------ 3.4  Per-company figures (split by theme) --------
    #
    # Three thematically-coherent figures rather than one mixed panel:
    #   (1) DISTRIBUTION DIAGNOSTICS — histogram + N(μ,σ), Q-Q, box plot.
    #       All three measure the same thing (departure from Gaussian).
    #   (2) PRICE TRAJECTORY + FORECASTS — one continuous price plot from
    #       train start through test end with all model forecasts and
    #       the GBM 95 % band.  Train/test boundary marked.
    #   (3) WAR-CORRELATION SCATTERS — intraday hourly + overnight gap.
    #       Both are scatters of stock vs proxy on the *same in-sample
    #       window*, so they belong together.
    #
    # ---- (1) Distribution diagnostics ------------------------------
    r = ret_in.values
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # Histogram + N(μ,σ)
    ax[0].hist(r, bins=30, density=True, alpha=0.7,
               color=cfg["color"], edgecolor="black")
    xg = np.linspace(r.min(), r.max(), 200)
    ax[0].plot(xg, stats.norm.pdf(xg, r.mean(), r.std()), "k--", lw=2,
               label=f"N(μ={r.mean():.4f}, σ={r.std():.4f})")
    ax[0].set_title("Histogram with N(μ,σ) overlay")
    ax[0].set_xlabel("Hourly return"); ax[0].set_ylabel("Density")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].annotate(
        f"skew = {summary['Skew']:+.2f}\nex-kurt = {summary['ExcessKurtosis']:+.2f}",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    # Q-Q plot
    stats.probplot(r, dist="norm", plot=ax[1])
    ax[1].set_title("Normal Q-Q plot")
    ax[1].grid(alpha=0.3)
    ax[1].annotate(
        f"Shapiro p = {sw_p:.4f}\nJarque-Bera p = {jb_p:.4f}\nKS p = {ks_p:.4f}",
        xy=(0.03, 0.97), xycoords="axes fraction",
        ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    # Box plot — IQR + outliers, gives the same story in a different visualisation
    bp = ax[2].boxplot(r, vert=False, widths=0.5, patch_artist=True,
                        showfliers=True, flierprops=dict(marker="x", markersize=5,
                                                          markeredgecolor="red"))
    bp["boxes"][0].set_facecolor(cfg["color"])
    bp["boxes"][0].set_alpha(0.6)
    ax[2].axvline(r.mean(), color="black", ls="--", lw=1, label=f"mean = {r.mean():.4f}")
    ax[2].set_title("Box plot (whiskers = 1.5×IQR, ✕ = outliers)")
    ax[2].set_xlabel("Hourly return")
    ax[2].set_yticks([])
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, axis="x")

    fig.suptitle(f"{ticker}  ({cfg['name']}) — Distribution Diagnostics  "
                 f"(in-sample, n = {len(r)})", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, f"{ticker}_distribution.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {ticker}_distribution.png")

    # ---- (2) Price trajectory + forecasts --------------------------
    fig, ax = plt.subplots(1, 1, figsize=(13, 5.5))
    ax.plot(P_in.index, P_in.values, color=cfg["color"], lw=1.1,
            label="In-sample close")
    ax.plot(P_in.index, trend_intercept + trend_slope * xhrs,
            "--", color="grey", lw=1.0,
            label=f"OLS trend β={trend_slope:+.3f}/h, R²={trend_R2:.2f}")
    ax.plot(P_out.index, P_out.values, "o-", color="black", lw=1.6,
            markersize=3.5, label="Out-of-sample (actual)")
    # Forecast lines
    fc_styles = [("Naive", "--", "grey"),
                 ("Drift", "-.", "tab:orange"),
                 ("ARIMA", "-",  "tab:blue"),
                 ("GBM",   ":",  "tab:green"),
                 ("Ridge", "-",  "tab:purple"),
                 ("RandomForest", "--", "tab:red"),
                 ("GradientBoosting", "-.", "tab:brown")]
    for n, st, col in fc_styles:
        ax.plot(P_out.index, pred_dict[n], st, color=col, alpha=0.85, lw=1.0,
                label=n)
    ax.plot(P_out.index, pred_dict["Ensemble"], "*-",
            color=cfg["color"], lw=1.8, markersize=5, label="Ensemble")
    ax.fill_between(P_out.index, out_df["GBM_lo"], out_df["GBM_hi"],
                    alpha=0.12, color="green", label="GBM 95% band")
    ax.axvline(P_in.index[-1], color="black", lw=0.5, ls=":",
               label="Train / test boundary")
    ax.set_title(f"{ticker}  ({cfg['name']}) — Hourly Price + Forecasts  "
                 f"(train 16 Mar–15 Apr, test 16–24 Apr 2026)")
    ax.set_ylabel("USD"); ax.set_xlabel("Datetime")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3, loc="best")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, f"{ticker}_forecast.png"), dpi=140)
    plt.close()
    print(f"  [saved] {ticker}_forecast.png")

    # ---- (3) War-correlation scatters ------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

    # Intraday hourly scatter
    if len(align_h) > 5:
        ax[0].scatter(align_h["BRENT"] * 100, align_h[ticker] * 100,
                      alpha=0.55, color=cfg["color"], s=24)
        sl2, ic2, rv2, _, _ = linregress(align_h["BRENT"], align_h[ticker])
        xs = np.linspace(align_h["BRENT"].min(), align_h["BRENT"].max(), 50)
        ax[0].plot(xs * 100, (ic2 + sl2 * xs) * 100, "k--", lw=1.2,
                   label=f"β={sl2:+.3f}, R²={rv2**2:.3f}")
        pr, ppv = pearsonr(align_h[ticker], align_h["BRENT"])
        ax[0].set_title(f"Channel A — Intraday hourly (n = {len(align_h)})\n"
                        f"Pearson r = {pr:+.3f},  p = {ppv:.3f}")
        ax[0].set_xlabel("Brent hourly return (%)")
        ax[0].set_ylabel(f"{ticker} hourly return (%)")
        ax[0].axhline(0, color="grey", lw=0.5)
        ax[0].axvline(0, color="grey", lw=0.5)
        ax[0].legend(); ax[0].grid(alpha=0.3)

    # Overnight gap scatter
    first_bars = stock_h[stock_h.index.hour == 13]
    s_ovn = first_bars.pct_change().dropna()
    bz_at_open = brent_h.reindex(first_bars.index, method="ffill")
    bz_ovn = bz_at_open.pct_change().dropna()
    A_ovn = pd.concat([s_ovn.rename("S"), bz_ovn.rename("B")], axis=1).dropna()
    if len(A_ovn) > 5:
        ax[1].scatter(A_ovn["B"] * 100, A_ovn["S"] * 100,
                      alpha=0.75, color=cfg["color"], s=44)
        sl3, ic3, rv3, _, _ = linregress(A_ovn["B"], A_ovn["S"])
        xs = np.linspace(A_ovn["B"].min(), A_ovn["B"].max(), 50)
        ax[1].plot(xs * 100, (ic3 + sl3 * xs) * 100, "k--", lw=1.2,
                   label=f"β={sl3:+.3f}, R²={rv3**2:.3f}")
        pr2, ppv2 = pearsonr(A_ovn["S"], A_ovn["B"])
        ax[1].set_title(f"Channel B — Overnight open-gap (n = {len(A_ovn)})\n"
                        f"Pearson r = {pr2:+.3f},  p = {ppv2:.4f}")
        ax[1].set_xlabel("Brent overnight return (%)")
        ax[1].set_ylabel(f"{ticker} open-gap return (%)")
        ax[1].axhline(0, color="grey", lw=0.5)
        ax[1].axvline(0, color="grey", lw=0.5)
        ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle(f"{ticker}  ({cfg['name']}) — War-Proxy Correlation",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, f"{ticker}_war_scatter.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {ticker}_war_scatter.png")

    RESULTS[ticker] = {
        "summary": summary,
        "fc_table": fc_table.assign(Ticker=ticker),
        "war_df":   war_df,
        "leadlag":  ll_df,
        "events":   ev_df,
        "pred_dict": pred_dict,
        "P_out":    P_out,
        "out_df":   out_df,
        "weights_d": weights_d,
        "align_h":  align_h,
        "align_d":  align_d,
    }


# ===================================================================
# 4.  CROSS-COMPANY COMPARISON
# ===================================================================
hr("CROSS-COMPANY COMPARISON")

stats_table = pd.DataFrame([R["summary"] for R in RESULTS.values()])
keep_cols = ["Ticker", "Name", "Observations", "Mean", "Median", "Std",
             "Skew", "ExcessKurtosis", "AnnualisedVol", "TotalReturn",
             "Shapiro_p", "JB_p", "KS_p", "ADF_price_p", "ADF_returns_p",
             "OLS_slope", "OLS_R2", "OLS_p", "ARIMA_order", "ARIMA_AIC",
             "Pearson_Brent_hourly", "Pearson_Brent_hourly_p",
             "Pearson_Brent_daily",  "Pearson_Brent_daily_p",
             "PointBiserial_r", "PointBiserial_p",
             "Mean_ret_escalation", "Mean_ret_deescalation",
             "EnsembleMembers", "EnsembleWeights"]
stats_table = stats_table[keep_cols]
stats_table.to_csv(os.path.join(HERE, "comparison_stats.csv"), index=False)
print("\n  Statistical comparison (per company):")
print(stats_table.round(5).to_string(index=False))

# Forecast accuracy comparison
fc_long = pd.concat([R["fc_table"] for R in RESULTS.values()], ignore_index=True)
fc_long.to_csv(os.path.join(HERE, "comparison_forecast_long.csv"), index=False)

for metric, fname in [("RMSE", "comparison_forecast_RMSE.csv"),
                      ("MAE",  "comparison_forecast_MAE.csv"),
                      ("MAPE_pct", "comparison_forecast_MAPE.csv"),
                      ("DirAcc", "comparison_forecast_DirAcc.csv"),
                      ("Pred_Mean", "comparison_forecast_PredMean.csv"),
                      ("Mean_Error", "comparison_forecast_MeanError.csv")]:
    pv = fc_long.pivot(index="Model", columns="Ticker", values=metric)
    pv.to_csv(os.path.join(HERE, fname))

print("\n  Per-hour PRICE RMSE (USD) by model × company:")
print(fc_long.pivot(index="Model", columns="Ticker", values="RMSE")
       .round(3).to_string())
print("\n  Per-hour PRICE MAPE (%) by model × company:")
print(fc_long.pivot(index="Model", columns="Ticker", values="MAPE_pct")
       .round(3).to_string())
print("\n  Directional accuracy (1H→1H) by model × company:")
print(fc_long.pivot(index="Model", columns="Ticker", values="DirAcc")
       .round(3).to_string())
print("\n  Predicted period-mean PRICE vs actual (USD) by model × company:")
print(fc_long.pivot(index="Model", columns="Ticker", values="Pred_Mean")
       .round(3).to_string())

actual_means = {t: float(R["P_out"].mean()) for t, R in RESULTS.items()}
print("  Actual period-mean prices :", {t: round(v, 3) for t, v in actual_means.items()})

# War correlation comparison
war_long = pd.concat([R["war_df"] for R in RESULTS.values()], ignore_index=True)
war_long.to_csv(os.path.join(HERE, "comparison_war_long.csv"), index=False)

print("\n  War-correlation cross-company (HOURLY Pearson r):")
hourly_pivot = (war_long.query("Resolution=='hourly'")
                          .pivot(index="Proxy", columns="Ticker", values="Pearson"))
print(hourly_pivot.round(4).to_string())
hourly_pivot.to_csv(os.path.join(HERE, "comparison_war_hourly_pearson.csv"))

hourly_pivot_p = (war_long.query("Resolution=='hourly'")
                            .pivot(index="Proxy", columns="Ticker", values="Pearson_p"))
print("\n  War-correlation cross-company (HOURLY Pearson p):")
print(hourly_pivot_p.round(4).to_string())
hourly_pivot_p.to_csv(os.path.join(HERE, "comparison_war_hourly_pearson_p.csv"))

beta_pivot = (war_long.query("Resolution=='hourly'")
                       .pivot(index="Proxy", columns="Ticker", values="Beta"))
print("\n  War-correlation cross-company (HOURLY OLS β):")
print(beta_pivot.round(4).to_string())
beta_pivot.to_csv(os.path.join(HERE, "comparison_war_hourly_beta.csv"))

# ===================================================================
# 4b.  TWO-CHANNEL WAR-EXPOSURE SCORE
# ===================================================================
# War exposure manifests through two structurally different channels
# that different metrics detect:
#
#   CHANNEL A — Continuous co-movement (the risk-asset transmission)
#     Stocks that absorb oil/war news as a steady β to oil futures bar
#     by bar. Picked up by Pearson r, Spearman ρ, OLS β.  Best detected
#     by tech / high-β / broad-systematic names.
#
#   CHANNEL B — Jump / event exposure (the oil-direct transmission)
#     Stocks that absorb war news as discrete jumps on news days. Picked
#     up by excess kurtosis (fat tails ARE the war risk), |event-study
#     r_pb|, and the magnitude of the largest negative bar. Best detected
#     by oil majors / commodity-direct names.
#
# A combined score is a z-scored average across both channels, computed
# across the three companies in our panel.

print("\n" + "=" * 72)
print("  4b. THREE-CHANNEL WAR-EXPOSURE SCORE")
print("=" * 72)
# War exposure manifests through three structurally different channels.
# Different metrics detect different channels, and a stock can be highly
# affected through one channel and undetectable through another.
#
#   CHANNEL A — Continuous intraday co-movement
#     Steady β to oil futures bar by bar during NYSE hours. Picked up by
#     hourly Pearson r, Spearman ρ, OLS β. Strong for high-β tech / risk-
#     asset names that read every oil tick into broad risk sentiment.
#
#   CHANNEL B — Overnight / open-gap absorption  (oil-direct channel)
#     CVX trades 9:30 AM – 4 PM ET only; while NYSE is closed Brent
#     trades 24 h. Over a 24 h cycle, the bulk of Brent's price action
#     happens *while NYSE is closed*. The first bar of each NYSE day
#     therefore absorbs the overnight oil move in one shot. Correlating
#     these open-gap returns with Brent's same-period overnight return
#     is the cleanest way to see direct oil exposure for a US-listed
#     oil major.  Hourly intraday Pearson r misses this entirely.
#
#   CHANNEL C — Jump / event exposure
#     Heavy tails ($\hat\gamma_2 \gg 0$), large |event-study point-
#     biserial|, large |max news-day return|. Captures discrete jumps
#     on news days. Picks up oil-major / commodity-direct names whose
#     reaction is concentrated in headline-driven moves.
#
# A composite score is the unweighted z-score average across the three
# channels, computed across the three companies in our panel.

# Channel A — continuous intraday co-movement vs WTI
ch_a = []
for t, R in RESULTS.items():
    wti_row = R["war_df"].query("Proxy=='WTI' & Resolution=='hourly'").iloc[0]
    ch_a.append({
        "Ticker": t,
        "abs_pearson_WTI_hourly":  abs(wti_row["Pearson"]),
        "abs_spearman_WTI_hourly": abs(wti_row["Spearman"]),
        "abs_beta_WTI_hourly":     abs(wti_row["Beta"]),
    })
ch_a_df = pd.DataFrame(ch_a).set_index("Ticker")

# Channel B — overnight / open-gap correlation (oil-direct channel)
ch_b = []
for t, R in RESULTS.items():
    s = R["summary"]
    ch_b.append({
        "Ticker": t,
        "abs_pearson_Brent_overnight": abs(s.get("Pearson_Brent_overnight", np.nan)),
        "abs_pearson_WTI_overnight":   abs(s.get("Pearson_WTI_overnight",   np.nan)),
        "abs_beta_Brent_overnight":    abs(s.get("Beta_Brent_overnight",    np.nan)),
        "R2_Brent_overnight":              s.get("R2_Brent_overnight",      np.nan),
    })
ch_b_df = pd.DataFrame(ch_b).set_index("Ticker")

# Channel C — jump / event exposure
ch_c = []
for t, R in RESULTS.items():
    s = R["summary"]
    ev = R["events"]
    worst_event = float(ev["ret"].abs().max()) if len(ev) else 0.0
    ch_c.append({
        "Ticker": t,
        "excess_kurt": float(s["ExcessKurtosis"]),
        "abs_pb":      abs(float(s["PointBiserial_r"])),
        "max_abs_event_return": worst_event,
    })
ch_c_df = pd.DataFrame(ch_c).set_index("Ticker")


def z_across(df):
    """Cross-row z-score for each column (3 rows = 3 companies)."""
    return (df - df.mean()) / df.std(ddof=1)


zA = z_across(ch_a_df)
zB = z_across(ch_b_df)
zC = z_across(ch_c_df)

score_df = pd.DataFrame({
    "ChA_continuous_intraday": zA.mean(axis=1),
    "ChB_overnight_oil_direct": zB.mean(axis=1),
    "ChC_jump_event":           zC.mean(axis=1),
})
score_df["Composite"] = score_df.mean(axis=1)
score_df["Rank"] = score_df["Composite"].rank(ascending=False).astype(int)

combined = pd.concat([ch_a_df.add_prefix("A_"),
                      ch_b_df.add_prefix("B_"),
                      ch_c_df.add_prefix("C_"),
                      score_df], axis=1)
combined.to_csv(os.path.join(HERE, "comparison_war_exposure_score.csv"))

print("\n  Channel A — continuous intraday co-movement (raw):")
print(ch_a_df.round(4).to_string())
print("\n  Channel B — overnight / open-gap correlation (raw):")
print(ch_b_df.round(4).to_string())
print("\n  Channel C — jump / event exposure (raw):")
print(ch_c_df.round(4).to_string())
print("\n  Z-scores per channel and composite war-exposure score:")
print(score_df.round(3).to_string())

# Bar chart of the three channels + composite. Same units (z-score) so
# all four panels can share a y-axis for direct comparability.
fig, axes = plt.subplots(1, 4, figsize=(16, 4.4),
                         sharey=True, constrained_layout=True)
xs = np.arange(len(score_df))
labels_local = list(score_df.index)
colors_local = [next(c["color"] for c in COMPANIES if c["ticker"] == t)
                for t in labels_local]
for ax, key, ttl in zip(axes,
                        ["ChA_continuous_intraday",
                         "ChB_overnight_oil_direct",
                         "ChC_jump_event",
                         "Composite"],
                        ["Channel A — continuous intraday",
                         "Channel B — overnight oil-direct",
                         "Channel C — jump / event",
                         "Composite (mean of A, B, C)"]):
    vals = score_df[key].values
    bars = ax.bar(xs, vals, color=colors_local,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(labels_local)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title(ttl, fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    # Value labels above each bar
    for bar, v in zip(bars, vals):
        offset = 0.04 if v >= 0 else -0.04
        va = "bottom" if v >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f"{v:+.2f}", ha="center", va=va, fontsize=8)
axes[0].set_ylabel("Z-score (cross-sectional)")
fig.suptitle("Three-channel war-exposure score  "
             "(z-scored across the 3 companies; higher = more exposed)",
             fontsize=12)
plt.savefig(os.path.join(HERE, "comparison_war_exposure_score.png"), dpi=140)
plt.close()
print("\n  [saved] comparison_war_exposure_score.png")

# ---------- Side-by-side forecast figure -----------------------------
# Three same-axes panels — every panel is "predicted vs actual price" so
# the eye can compare across companies directly.
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
for ax, cfg in zip(axes, COMPANIES):
    R   = RESULTS[cfg["ticker"]]
    Pa  = R["P_out"]
    pd_ = R["pred_dict"]
    od  = R["out_df"]
    ax.plot(Pa.index, Pa.values, "o-", color="black", lw=1.6, ms=3.5,
            label="Actual")
    ax.plot(Pa.index, pd_["Naive"], "--", color="grey", alpha=0.8, label="Naive")
    ax.plot(Pa.index, pd_["ARIMA"], color="tab:blue", alpha=0.8, label="ARIMA")
    ax.plot(Pa.index, pd_["GBM"],   ":", color="tab:green", alpha=0.85, label="GBM")
    ax.plot(Pa.index, pd_["Ensemble"], "*-", color=cfg["color"], lw=1.7,
            label="Ensemble")
    ax.fill_between(Pa.index, od["GBM_lo"], od["GBM_hi"],
                    alpha=0.10, color="green", label="GBM 95%")
    ax.set_title(f"{cfg['ticker']}  ({cfg['label']})", fontsize=10)
    ax.set_ylabel("USD")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.suptitle("Hourly-price forecasts vs actual — 16 to 24 April 2026",
             fontsize=12)
plt.savefig(os.path.join(HERE, "comparison_forecast.png"), dpi=140)
plt.close()
print("\n  [saved] comparison_forecast.png")

# ---------- War-correlation bar charts (separated by channel) --------
# Channel A (continuous intraday) and Channel B (overnight open-gap) on
# the same axis — both are Pearson r against Brent at different
# resolutions, so they belong together visually.  The event-study is a
# different thing and gets its own panel.
xs = np.arange(len(COMPANIES))
labels = [c["ticker"] for c in COMPANIES]
colors = [c["color"]  for c in COMPANIES]

# Pull cross-company values
intraday_brent = hourly_pivot.loc["BRENT"].values
overnight_brent = np.array([RESULTS[t]["summary"]["Pearson_Brent_overnight"]
                             for t in labels])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)

# Pearson r vs Brent — both channels grouped
W = 0.36
axes[0].bar(xs - W/2, intraday_brent, W, color=colors, alpha=0.55,
            label="Channel A — intraday hourly (n = 153)",
            edgecolor="black", linewidth=0.5)
axes[0].bar(xs + W/2, overnight_brent, W, color=colors,
            label="Channel B — overnight open-gap (n = 21)",
            edgecolor="black", linewidth=0.5)
axes[0].set_xticks(xs); axes[0].set_xticklabels(labels)
axes[0].set_ylabel("Pearson r")
axes[0].set_title("Pearson correlation with Brent — by channel")
axes[0].axhline(0, color="grey", lw=0.5); axes[0].grid(alpha=0.3, axis="y")
axes[0].legend(fontsize=8)

# Event-study point-biserial — different metric, separate panel
pbs = [RESULTS[t]["summary"]["PointBiserial_r"] for t in labels]
axes[1].bar(xs, pbs, color=colors, edgecolor="black", linewidth=0.5)
axes[1].set_xticks(xs); axes[1].set_xticklabels(labels)
axes[1].set_ylabel("Point-biserial r")
axes[1].set_title("Channel C — event-study (5 dated news points)")
axes[1].axhline(0, color="grey", lw=0.5); axes[1].grid(alpha=0.3, axis="y")
fig.suptitle("Iran-US war proxy correlation — three channels",
             fontsize=12)
plt.savefig(os.path.join(HERE, "comparison_war.png"), dpi=140)
plt.close()
print("  [saved] comparison_war.png")

# Stats panel comparison — one figure, three sub-bars; all three are
# distributional moments so they belong together.
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for ax, key, ttl, ylab in zip(
        axes,
        ["Skew", "ExcessKurtosis", "AnnualisedVol"],
        ["Skewness  γ₁", "Excess kurtosis  γ₂", "Annualised volatility"],
        ["", "", "% / yr"]):
    vals = [RESULTS[t]["summary"][key] for t in labels]
    if key == "AnnualisedVol":
        vals = [v * 100 for v in vals]
    ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_title(ttl); ax.set_ylabel(ylab)
    ax.axhline(0, color="grey", lw=0.5)
    ax.grid(alpha=0.3, axis="y")
fig.suptitle("Cross-company distributional moments (in-sample, n = 153)",
             fontsize=12)
plt.savefig(os.path.join(HERE, "comparison_stats.png"), dpi=140)
plt.close()
print("  [saved] comparison_stats.png")

print("\nALL DONE.")
