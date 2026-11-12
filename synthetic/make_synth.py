"""
Synthetic NQ data to test nq_vol_open.py offline, without Yahoo.

It simulates what the model has to deal with: long-memory daily volatility
(HAR dynamics), a U-shaped intraday profile with spikes at 08:30 and at the
cash open, "event day" shocks that hit both the premarket and the window,
an overnight shock, and fat-tailed t(5) returns.

Usage:
    python synthetic/make_synth.py OUT_DIR [N_5MIN_SESSIONS] [SEED]
    python nq_vol_open.py --offline OUT_DIR --now "2026-10-26 09:22"
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
k = int(sys.argv[2]) if len(sys.argv) > 2 else 60
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 7)

days = pd.bdate_range("2016-10-24", "2026-10-26")
n = len(days)

# Daily log variance with HAR dynamics
x = np.empty(n)
x[:22] = -8.9
for t in range(22, n):
    x[t] = (-8.9 * 0.12 + 0.30 * x[t - 1] + 0.35 * x[t - 5:t].mean() + 0.23 * x[t - 22:t].mean()
            + rng.normal(0, 0.45))

# Intraday profile: 276 bars from 18:00 to 17:00 NY
m = np.arange(0, 1380, 5)
w = 0.25 + 0.35 * np.exp(-((m - 540) / 60) ** 2)                    # London, 03:00
w += 1.8 * np.exp(-((m - 870) / 12) ** 2)                           # 08:30 releases
w += np.where(m >= 930, 3.2 * np.exp(-(m - 930) / 35) + 0.9, 0)     # cash open
w += np.where(m >= 1260, 0.8, 0)                                    # close
w /= w.sum()

ev = np.exp(rng.normal(0, 0.40, n))       # event shock: premarket + window
on = np.exp(rng.normal(0, 0.30, n))       # overnight shock
mult = np.ones((n, len(m)))
mult[:, m < 870] *= on[:, None]
mult[:, (m >= 870) & (m < 1020)] *= ev[:, None]
mult[:, (m >= 930) & (m < 1020)] *= np.sqrt(on)[:, None]
mult /= (mult * w).sum(1, keepdims=True)  # keeps the daily variance unchanged

var_bar = np.exp(x)[:, None] * w[None, :] * mult
z = rng.standard_t(5, size=var_bar.shape) / np.sqrt(5 / 3)
r = np.sqrt(var_bar) * z
logp = np.log(12000) + np.cumsum(r.ravel()).reshape(r.shape)
close = np.exp(logp)
open_ = np.exp(np.r_[np.log(12000), logp.ravel()[:-1]]).reshape(r.shape)
sb = np.sqrt(var_bar)
high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 1, r.shape)) * sb * 0.5)
low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 1, r.shape)) * sb * 0.5)

# Session for day D opens at 18:00 NY the previous evening
ts = np.array([d - pd.Timedelta(days=1) + pd.Timedelta(hours=18) + pd.to_timedelta(m, unit="min")
               for d in days])

# Daily bars (the last day is the one being forecast, so it is left out)
daily = pd.DataFrame({"open": open_[:, 0], "high": high.max(1), "low": low.min(1), "close": close[:, -1]},
                     index=days)[:-1]
daily.to_csv(out / "nq_daily.csv")
vxn = 100 * np.sqrt(252 * np.exp(0.5 * x + 0.5 * np.r_[x[1:], x[-1]] + rng.normal(0, 0.25, n))) * 1.1
pd.Series(vxn[:-1], index=days[:-1], name="close").to_frame().to_csv(out / "vxn_daily.csv")

# Last k sessions of 5-min bars; the forecast day is cut before 09:20
rows = []
for i in range(n - k, n):
    sel = slice(None) if i < n - 1 else (m < 920)
    idx = pd.DatetimeIndex(ts[i][sel]).tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    rows.append(pd.DataFrame({"open": open_[i][sel], "high": high[i][sel], "low": low[i][sel],
                              "close": close[i][sel]}, index=idx))
b = pd.concat(rows)
b = b[b.index.notna()]
b.rename_axis("datetime").to_csv(out / "nq_5m.csv")
print(f"ok: {len(daily)} daily bars, {len(b)} 5-min bars -> {out}")
