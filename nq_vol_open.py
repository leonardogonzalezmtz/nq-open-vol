#!/usr/bin/env python3
"""
nq_vol_open.py - Daily volatility forecast for NQ futures over the 09:30-11:00 New York window.

Run it every morning between 09:15 and 09:29 NY:   python nq_vol_open.py

MODEL (two layers)
------------------
Layer 1 - daily, ~10 years of history.
    Log HAR (Corsi 2009) on the daily Garman-Klass variance of NQ=F, extended
    with VXN implied volatility (HAR-IV, Busch, Christensen & Nielsen 2011):
        log GK[t] = b0 + bd*log GK[t-1] + bw*log mean5(GK) + bm*log mean22(GK)
                    + bv*log VXN_daily[t-1]^2 + e
    It forecasts how much the whole session should move. It is fitted on an
    expanding window, so each day's forecast only uses data from before that day.

Layer 2 - the 09:30-11:00 window, 5-minute bars.
        log Y[t] = a + b*logGKhat[t] + c*log RV_overnight[t] + d*log RV_premarket[t] + u
    with Y = realized variance of the window (sum of squared 5-min returns) and,
    in a separate fit, Y = high-low range of the window / price.
    - RV_overnight: Globex 18:00 -> 08:30 (Asia + Europe).
    - RV_premarket: 08:30 -> cutoff time (picks up the 08:30 macro releases).
    Intraday variance follows a U shape (Andersen & Bollerslev 1997). This
    layer learns how much of the daily variance lands in the window and how it
    is conditioned by what already happened overnight.

Why two layers: Yahoo only serves 60 days of 5-minute bars. A HAR fitted
directly on the window needs 22 days of lags and eats most of that sample. The
long-memory part comes from the daily layer (thousands of days) and the window
layer only estimates 3-4 coefficients with ridge shrinkage. The local cache
grows every day the script runs, so layer 2 improves over time on its own.

Means in levels use Duan (1983) smearing, not exp(mean of logs).
Intervals use empirical leave-one-out residual quantiles, not a normal assumption.
Validation: expanding-window out-of-sample forecasts against two benchmarks
(yesterday's window, 22-day average) with QLIKE loss (Patton 2011), plus
coverage of the 10-90 interval.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

NY = "America/New_York"
HERE = Path(__file__).resolve().parent
DATA = HERE / "nq_vol_data"
CACHE_5M = DATA / "nq_5m_cache.csv"
LOG_PATH = DATA / "forecast_log.csv"

# Minutes since the Globex open (18:00 NY). All intraday logic lives on this
# axis so the overnight session never crosses a calendar midnight.
M_PRE = 14 * 60 + 30      # 08:30
M_WIN0 = 15 * 60 + 30     # 09:30
M_WIN1 = 17 * 60          # 11:00
N_WIN_BARS = 18           # 90 min / 5 min
MIN_TRAIN_WIN = 25        # minimum days to fit layer 2


def sess_minute(t: time) -> int:
    return ((t.hour * 60 + t.minute) - 18 * 60) % 1440


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _yf():
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is missing. Run: pip install -r requirements.txt")
    return yf


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].astype(float)
    df = df.dropna()
    ok = (df.high >= df.low) & (df.low > 0)
    ok &= df[["open", "close"]].max(axis=1) <= df.high * 1.0001
    ok &= df[["open", "close"]].min(axis=1) >= df.low * 0.9999
    return df[ok]


def read_bars_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    col = next(c for c in df.columns if c.lower() in ("datetime", "date", "time", "timestamp"))
    raw = df[col].astype(str)
    if raw.str.contains(r"(?:[+-]\d\d:?\d\d|Z)$").any():
        # explicit offsets: going through UTC avoids the mixed-offset error
        # when the file crosses a DST change (-04:00 / -05:00)
        idx = pd.to_datetime(raw, utc=True).dt.tz_convert(NY)
    else:
        # no offset: assume New York time
        idx = pd.to_datetime(raw).dt.tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
    df.index = pd.DatetimeIndex(idx)
    df = df[df.index.notna()]
    return _clean_ohlc(df)


def load_5m(src_dir: Path | None, extra_csv: Path | None, now: pd.Timestamp) -> pd.DataFrame:
    frames = []
    if CACHE_5M.exists():
        frames.append(read_bars_csv(CACHE_5M))
    if extra_csv:
        frames.append(read_bars_csv(extra_csv))
    if src_dir:
        frames.append(read_bars_csv(src_dir / "nq_5m.csv"))
    else:
        h = _yf().Ticker("NQ=F").history(period="60d", interval="5m", auto_adjust=False)
        if h.empty:
            sys.exit("Yahoo returned no 5-minute bars for NQ=F. Check your connection.")
        h.index = h.index.tz_convert(NY)
        frames.append(_clean_ohlc(h))
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index + pd.Timedelta(minutes=5) <= now]   # closed bars only
    DATA.mkdir(exist_ok=True)
    df.rename_axis("datetime").to_csv(CACHE_5M)
    return df


def load_daily(src_dir: Path | None) -> pd.DataFrame:
    if src_dir:
        nq = pd.read_csv(src_dir / "nq_daily.csv", index_col=0, parse_dates=True)
        vx = pd.read_csv(src_dir / "vxn_daily.csv", index_col=0, parse_dates=True)
    else:
        yf = _yf()
        nq = yf.Ticker("NQ=F").history(period="10y", interval="1d", auto_adjust=False)
        vx = yf.Ticker("^VXN").history(period="10y", interval="1d", auto_adjust=False)
        if vx.empty:
            vx = yf.Ticker("^VIX").history(period="10y", interval="1d", auto_adjust=False)
            print("Note: ^VXN not available, using ^VIX as the implied vol proxy.")
    for d in (nq, vx):
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        d.index = d.index.normalize()
    nq = _clean_ohlc(nq)
    vx = vx.rename(columns=str.lower)["close"].astype(float)
    nq["vxn"] = vx.reindex(nq.index).ffill()
    return nq.dropna()


# --------------------------------------------------------------------------
# Layer 1: daily HAR-IV
# --------------------------------------------------------------------------
def daily_design(nq: pd.DataFrame) -> pd.DataFrame:
    """Row i uses information up to the close of day i (to forecast day i+1)."""
    lo, lh = np.log(nq.close / nq.open), np.log(nq.high / nq.low)
    gk = 0.5 * lh**2 - (2 * np.log(2) - 1) * lo**2
    gk = gk.clip(lower=gk[gk > 0].quantile(0.005))
    X = pd.DataFrame({
        "ld": np.log(gk),
        "lw": np.log(gk.rolling(5).mean()),
        "lm": np.log(gk.rolling(22).mean()),
        "liv": np.log((nq.vxn / 100) ** 2 / 252),
    }, index=nq.index)
    X["y_next"] = X["ld"].shift(-1)
    return X


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def daily_forecast(D: pd.DataFrame, target: pd.Timestamp) -> float:
    """Expected log GK variance for `target`, using only days < target."""
    hist = D[D.index < target].dropna(subset=["ld", "lw", "lm", "liv"])
    train = hist.dropna(subset=["y_next"])
    cols = ["ld", "lw", "lm", "liv"]
    beta = _ols(train[cols].values, train["y_next"].values)
    x = np.r_[1.0, hist[cols].iloc[-1].values]
    return float(x @ beta)


# --------------------------------------------------------------------------
# Layer 2: the 09:30-11:00 window
# --------------------------------------------------------------------------
def session_table(bars: pd.DataFrame, cut_m: int) -> pd.DataFrame:
    b = bars.copy()
    b["sess"] = (b.index + pd.Timedelta(hours=6)).tz_localize(None).normalize()
    b["m"] = [sess_minute(t) for t in b.index.time]
    r = np.log(b.close).diff()
    first = b.sess != b.sess.shift()
    r[first] = np.log(b.close / b.open)[first]       # no 18:00 gaps or contract rolls
    b["r2"] = r**2

    on_end = min(cut_m, M_PRE)
    rows = {}
    for s, g in b.groupby("sess"):
        w = g[(g.m >= M_WIN0) & (g.m < M_WIN1)]
        on = g[g.m < on_end]
        pre = g[(g.m >= M_PRE) & (g.m < cut_m)]
        p0 = g[g.m < M_WIN0].close
        rows[s] = {
            "n_win": len(w),
            "rv_w": w.r2.sum(),
            "rng_w": (w.high.max() - w.low.min()) / p0.iloc[-1] if len(p0) and len(w) else np.nan,
            "rv_on": on.r2.sum(), "n_on": len(on),
            "rv_pre": pre.r2.sum(), "n_pre": len(pre),
            "px": p0.iloc[-1] if len(p0) else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


RIDGE_LAMBDA = 10.0   # roughly 10 pseudo-observations pulling the slopes toward 0


def fit_predict(train: pd.DataFrame, x_new: pd.Series, feats: list[str], ycol: str) -> dict:
    """Ridge on standardized features (intercept not penalized).

    With 30-60 days and 3-4 coefficients, OLS overfits. Ridge shrinks the
    slopes toward zero, i.e. toward "tomorrow looks like the average day".
    Intervals use leave-one-out residuals e/(1-h), because in-sample
    residuals understate the tails when the sample is small.
    """
    X = train[feats].values.astype(float)
    y = np.log(train[ycol].values.astype(float))
    mu_x, sd_x = X.mean(0), X.std(0) + 1e-12
    A = np.column_stack([np.ones(len(X)), (X - mu_x) / sd_x])
    P = np.eye(A.shape[1]) * RIDGE_LAMBDA
    P[0, 0] = 0.0
    M = np.linalg.solve(A.T @ A + P, A.T)
    b = M @ y
    h = np.einsum("ij,ji->i", A, M)                # diagonal of the hat matrix
    resid = (y - A @ b) / (1 - h)                  # LOO residuals
    z = (x_new[feats].values.astype(float) - mu_x) / sd_x
    mu = float(b[0] + z @ b[1:])
    beta = np.r_[b[0] - (b[1:] / sd_x) @ mu_x, b[1:] / sd_x]   # in original units
    draws = np.exp(mu + resid)                     # empirical predictive distribution
    return {
        "mean": float(draws.mean()),               # Duan smearing
        "q10": float(np.exp(mu + np.quantile(resid, 0.10))),
        "q50": float(np.exp(mu + np.quantile(resid, 0.50))),
        "q90": float(np.exp(mu + np.quantile(resid, 0.90))),
        "beta": beta, "r2": 1 - resid.var() / y.var(),
    }


def qlike(rv, f):
    x = rv / f
    return x - np.log(x) - 1


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def resolve_target(now: pd.Timestamp) -> tuple[pd.Timestamp, int]:
    t = now.time()
    m = sess_minute(t)
    if time(11, 0) <= t < time(18, 0):           # today's window is already over
        d, m = now.normalize().tz_localize(None) + pd.Timedelta(days=1), 0
    else:
        d = (now + pd.Timedelta(hours=6)).normalize().tz_localize(None)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
        m = 0
    sess_open = (d - pd.Timedelta(days=1)).tz_localize(NY) + pd.Timedelta(hours=18)  # Monday: Sunday 18:00
    if now < sess_open:
        m = 0
    return d, min(m, M_WIN0)                      # never use data from inside the window


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for st in self.streams:
            st.write(x)

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    ap = argparse.ArgumentParser(description="Expected NQ volatility, 09:30-11:00 NY")
    ap.add_argument("--offline", type=Path, help="folder with nq_5m.csv, nq_daily.csv and vxn_daily.csv")
    ap.add_argument("--csv5m", type=Path, help="your own 5-min CSV (datetime,open,high,low,close) to extend history")
    ap.add_argument("--stop", type=float, help="stop size in points, to express it in units of 5-min sigma")
    ap.add_argument("--now", help="override the NY clock, e.g. '2026-10-26 09:20' (testing)")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    global DATA, CACHE_5M, LOG_PATH
    if a.offline:
        DATA = a.offline / "nq_vol_data"
        CACHE_5M, LOG_PATH = DATA / "nq_5m_cache.csv", DATA / "forecast_log.csv"

    now = pd.Timestamp(a.now, tz=NY) if a.now else pd.Timestamp.now(tz=NY)
    target, cut_m = resolve_target(now)

    bars = load_5m(a.offline, a.csv5m, now)
    nq = load_daily(a.offline)
    D = daily_design(nq)

    S = session_table(bars, cut_m)
    hist = S[(S.index < target) & (S.n_win >= N_WIN_BARS - 1) & (S.rv_w > 0)].copy()
    hist = hist[hist.index > D.index[30]]
    if len(hist) < MIN_TRAIN_WIN + 5:
        sys.exit(f"Only {len(hist)} complete 5-min sessions; {MIN_TRAIN_WIN + 5} are needed.")

    # Ex-ante daily forecast for every past session and for today
    hist["ldaily"] = [daily_forecast(D, d) for d in hist.index]
    ld_today = daily_forecast(D, target)

    # Which features exist today: >= 1 h of overnight bars, >= 15 min of premarket
    today = S.loc[target] if target in S.index else None
    feats = ["ldaily"]
    if today is not None and today.n_on >= 12 and cut_m >= 60:
        feats.append("lon")
    if today is not None and today.n_pre >= 3 and cut_m >= M_PRE + 15:
        feats.append("lpre")
    for f, c in (("lon", "rv_on"), ("lpre", "rv_pre")):
        if f in feats:
            floor = hist[c][hist[c] > 0].quantile(0.01)
            hist[f] = np.log(hist[c].clip(lower=floor))
    x_today = pd.Series({"ldaily": ld_today})
    if "lon" in feats:
        x_today["lon"] = np.log(max(today.rv_on, hist.rv_on[hist.rv_on > 0].quantile(0.01)))
    if "lpre" in feats:
        x_today["lpre"] = np.log(max(today.rv_pre, hist.rv_pre[hist.rv_pre > 0].quantile(0.01)))

    # ---- Out-of-sample validation (expanding window) ----
    oos = []
    for k in range(MIN_TRAIN_WIN, len(hist)):
        tr, te = hist.iloc[:k], hist.iloc[k]
        p = fit_predict(tr, te, feats, "rv_w")
        pr = fit_predict(tr, te, feats, "rng_w")
        oos.append({
            "date": hist.index[k], "rv": te.rv_w, "rng": te.rng_w,
            "f_model": p["mean"], "f_rw": tr.rv_w.iloc[-1], "f_22": tr.rv_w.iloc[-22:].mean(),
            "r10": pr["q10"], "r90": pr["q90"], "px": te.px,
        })
    oos = pd.DataFrame(oos).set_index("date")
    ql = {k: qlike(oos.rv, oos[f"f_{k}"]).mean() for k in ("model", "rw", "22")}
    cover = ((oos.rng >= oos.r10) & (oos.rng <= oos.r90)).mean()

    # ---- Today's forecast ----
    fr = fit_predict(hist, x_today, feats, "rv_w")
    fg = fit_predict(hist, x_today, feats, "rng_w")
    px = today.px if (today is not None and not np.isnan(today.px)) else bars.close.iloc[-1]
    sig_pts = np.sqrt(fr["mean"]) * px
    sig5 = sig_pts / np.sqrt(N_WIN_BARS)
    pct = (hist.rv_w < fr["mean"]).mean() * 100
    regime = ("LOW" if pct < 25 else "NORMAL" if pct < 75 else "HIGH" if pct < 90 else "VERY HIGH")

    # ---- Log and tracking of past forecasts ----
    DATA.mkdir(exist_ok=True)
    log = pd.read_csv(LOG_PATH, parse_dates=["target"]) if LOG_PATH.exists() else pd.DataFrame()
    row = {
        "run_ny": now.strftime("%Y-%m-%d %H:%M"), "target": target, "cutoff": _fmt_m(cut_m),
        "feats": "+".join(feats), "price": round(px, 2), "sigma_pts": round(sig_pts, 1),
        "rng_p10": round(fg["q10"] * px, 1), "rng_p50": round(fg["q50"] * px, 1),
        "rng_p90": round(fg["q90"] * px, 1), "regime": regime,
        "real_sigma_pts": np.nan, "real_rng_pts": np.nan,
    }
    log = pd.concat([log[log["target"] != target] if len(log) else log, pd.DataFrame([row])],
                    ignore_index=True)
    full = S[S.n_win >= N_WIN_BARS - 1]
    for i, r in log.iterrows():
        if r.target in full.index:
            s = full.loc[r.target]
            log.at[i, "real_sigma_pts"] = round(np.sqrt(s.rv_w) * s.px, 1)
            log.at[i, "real_rng_pts"] = round(s.rng_w * s.px, 1)
    log.to_csv(LOG_PATH, index=False)

    # ---- Report (screen + nq_vol_data/last_run.txt) ----
    out = open(DATA / "last_run.txt", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, out)
    bar = "=" * 62
    print(bar)
    print(f" NQ  |  Expected volatility 09:30-11:00 NY  |  {target:%a %d %b %Y}")
    print(bar)
    upto = f"{_fmt_m(cut_m)} NY" if cut_m > 0 else "no bars from this session yet"
    print(f" Run: {now:%a %H:%M} NY   data up to: {upto}   ref price: {px:,.2f}")
    print(f" Regime: {regime}  (percentile {pct:.0f} of the last {len(hist)} days)")
    print()
    print(f" Window sigma                 {sig_pts:7.1f} pts   ({np.sqrt(fr['mean'])*100:.2f}%)")
    print(f" Sigma per 5-min bar          {sig5:7.1f} pts")
    print(f" Expected high-low range      {fg['q50']*px:7.1f} pts  (median)")
    print(f"   80% interval               {fg['q10']*px:7.1f} - {fg['q90']*px:.1f} pts")
    if a.stop:
        print()
        print(f" Stop of {a.stop:g} pts = {a.stop/sig5:.1f} sigmas of a 5-min bar; "
              f"{a.stop/(fg['q50']*px)*100:.0f}% of the median range")
    print()
    print(f" Daily layer (HAR-IV): expected full-day sigma {np.sqrt(np.exp(ld_today))*100:.2f}%")
    print(" Window layer: " + "  ".join(f"{n}={v:+.2f}" for n, v in zip(["c"] + feats, fr["beta"]))
          + f"   R2 (LOO)={fr['r2']:.2f}")
    if len(feats) < 3:
        print(" Note: overnight/premarket not complete. For the best forecast run it 09:15-09:29 NY.")
    print()
    print(f" Out-of-sample backtest ({len(oos)} days, QLIKE, lower is better):")
    print(f"   model {ql['model']:.3f} | yesterday's window {ql['rw']:.3f} | 22-day average {ql['22']:.3f}")
    print(f"   improvement vs 22-day average: {(1-ql['model']/ql['22'])*100:+.0f}%   "
          f"10-90 range coverage: {cover*100:.0f}% (target 80%)")
    done = log.dropna(subset=["real_rng_pts"]).tail(5)
    if len(done):
        print()
        print(" Recent forecasts vs reality (range, pts):")
        for _, r in done.iterrows():
            hit = "in" if r.rng_p10 <= r.real_rng_pts <= r.rng_p90 else "OUT"
            print(f"   {pd.Timestamp(r.target):%d %b}  expected {r.rng_p10:.0f}-{r.rng_p90:.0f}"
                  f"  actual {r.real_rng_pts:.0f}  [{hit}]")
    print(bar)
    sys.stdout = sys.__stdout__
    out.close()

    if not a.no_plot:
        _plot(oos, target, fg, px)


def _fmt_m(m: int) -> str:
    h, mm = divmod((m + 18 * 60) % 1440, 60)
    return f"{h:02d}:{mm:02d}"


def _plot(oos, target, fg, px):
    """Show the chart in a window. Nothing is saved to disk."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(10, 4.2))
    try:
        fig.canvas.manager.set_window_title(f"NQ vol 09:30-11:00  |  {target:%d %b %Y}")
    except Exception:
        pass
    ax.fill_between(oos.index, oos.r10 * oos.px, oos.r90 * oos.px, alpha=0.25,
                    label="10-90 forecast (out of sample)")
    ax.plot(oos.index, oos.rng * oos.px, "o-", ms=3, lw=1, label="actual 09:30-11:00 range")
    ax.errorbar([target], [fg["q50"] * px],
                yerr=[[fg["q50"] * px - fg["q10"] * px], [fg["q90"] * px - fg["q50"] * px]],
                fmt="s", color="crimson", capsize=4, label="next session")
    ax.set_ylabel("NQ points")
    ax.set_title("NQ 09:30-11:00 range: forecast vs actual")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    print(" (close the chart window to finish)")
    plt.show()


if __name__ == "__main__":
    main()
