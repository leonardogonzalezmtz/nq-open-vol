# nq-open-vol

A daily volatility forecast for Nasdaq-100 futures (NQ) over the first 90 minutes of the cash session, from 09:30 to 11:00 New York time. You run it before the open and it tells you how much NQ should move in that window: the expected sigma in points, the sigma of a single 5-minute bar, the expected high-low range with an 80% interval and how the day ranks against recent history.

It does not forecast direction. Volatility clusters and can be predicted, direction mostly can't. The use is risk: knowing before the open if the session is likely to be a 200-point or a 500-point window.

## Model

The forecast has two layers because the data has two very different lengths. Free 5-minute data from Yahoo only goes back 60 days, while daily data goes back ten years.

**Layer 1 (daily, ~10 years).** A log HAR model (Corsi 2009) on the daily Garman-Klass variance of NQ, extended with VXN implied volatility as in Busch, Christensen and Nielsen (2011):

```
log GK[t] = b0 + bd·log GK[t-1] + bw·log mean5(GK) + bm·log mean22(GK) + bv·log VXN²[t-1] + e
```

The daily, weekly and monthly terms give the long memory of volatility and VXN adds the information that option prices already carry. It is fitted on an expanding window so every forecast only uses data from before that day.

**Layer 2 (the 09:30-11:00 window, 5-minute bars).** Intraday variance follows a U shape (Andersen and Bollerslev 1997), so the share of the daily variance that lands in the first 90 minutes is large and fairly stable. This layer maps the daily forecast to the window and conditions it on what already happened before the open:

```
log Y[t] = a + b·log GKhat[t] + c·log RV_overnight[t] + d·log RV_premarket[t] + u
```

`Y` is either the realized variance of the window (sum of squared 5-minute returns) or its high-low range. `RV_overnight` covers Globex from 18:00 to 08:30 and `RV_premarket` covers 08:30 to the time the script runs, which picks up the 08:30 macro releases. With only a few dozen days to fit, OLS overfits, so the coefficients are estimated with ridge on standardized features. Means in levels use Duan (1983) smearing and the intervals come from empirical leave-one-out residual quantiles instead of a normal assumption. Every run appends the 5-minute bars to a local cache, so the sample for this layer grows by one day each time the script runs.

## Validation

Every run prints an out-of-sample backtest on an expanding window. For each past day the model is refitted only on the days before it and scored against two naive benchmarks with QLIKE loss (Patton 2011), which is the standard loss for variance forecasts and penalizes under-forecasting more than over-forecasting. It also reports how often the actual range fell inside the 10-90 interval.

First run on real NQ data, 12 Nov 2026, 25 out-of-sample days (9 Oct to 12 Nov 2026), daily layer only:

| Forecast | QLIKE |
|---|---|
| This model | **0.334** |
| Yesterday's window | 0.402 |
| 22-day average | 0.414 |

That is 19% lower loss than the 22-day average, and the 10-90 interval held the actual range on 84% of days against a target of 80%. The one large miss is 5 Nov 2026, a 1,496-point window against an upper band near 470. The model adapted the next day and the following five sessions all landed inside the band.

Before touching real data the model was checked on synthetic NQ (`synthetic/make_synth.py`), which simulates HAR volatility, the intraday U shape, spikes at 08:30 and 09:30, event-day shocks, overnight shocks and fat-tailed returns. Over seven seeds with 60 days of 5-minute data the model beat the 22-day average in QLIKE on six and kept interval coverage between 71% and 85%. Testing on synthetic data first also found two real bugs: Monday's session opens on Sunday at 18:00, and CSV timestamps break when a file crosses a daylight-saving change.

## Usage

```
pip install -r requirements.txt
python nq_vol_open.py
```

On Windows you can double-click `run_nq_vol.bat`. Run it between 09:15 and 09:29 New York time so it has the full overnight and premarket. If you run it earlier the premarket term is dropped and the output says so. After 11:00 it forecasts the next session.

Options:

```
--stop PTS      express a stop size in units of 5-minute sigma
--csv5m FILE    add your own 5-minute bars (datetime,open,high,low,close) to extend history
--offline DIR   run on local CSVs instead of Yahoo (see synthetic/)
--now "..."     override the New York clock, for testing
--no-plot       skip the chart window
```

`run_nq_vol.bat` reads extra arguments from an optional `local_args.txt` next to it, so personal settings stay out of the repo.

To run fully offline on synthetic data:

```
python synthetic/make_synth.py synth_data 60 3
python nq_vol_open.py --offline synth_data --now "2026-10-26 09:22"
```

## Output

The run prints the forecast and opens a chart of the forecast band against the actual range for every out-of-sample day, with the next session marked in red. Nothing is saved except a local folder `nq_vol_data/` (ignored by git) with the 5-minute cache, a `forecast_log.csv` that fills in the actual range once each forecast day is over, and the text of the last run.

## Known limitations

The model has no event calendar. Releases at 10:00 (ISM, JOLTS, consumer sentiment) and FOMC days land inside or around the window and are the biggest source of error. A dummy for scheduled events is the first planned addition.

The window layer is fitted on whatever 5-minute history exists. With the Yahoo limit that starts at about 60 days, which makes the interval noisy. The `--csv5m` option exists to load longer history from a data vendor.

Yahoo futures data is delayed by about 10 minutes, so a run at 09:20 sees premarket data up to around 09:10. The backtest is built with the same cutoff as the live run, so the evaluation is consistent with what the model sees in practice.

Half-day sessions are dropped from the history because the window is incomplete, but the script does not know the exchange calendar and will still produce a forecast for a holiday.

## References

- Andersen, T. G. and Bollerslev, T. (1997). Intraday periodicity and volatility persistence in financial markets. *Journal of Empirical Finance* 4, 115-158.
- Busch, T., Christensen, B. J. and Nielsen, M. Ø. (2011). The role of implied volatility in forecasting future realized volatility and jumps in foreign exchange, stock and bond markets. *Journal of Econometrics* 160, 48-57.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics* 7, 174-196.
- Duan, N. (1983). Smearing estimate: a nonparametric retransformation method. *Journal of the American Statistical Association* 78, 605-610.
- Garman, M. B. and Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business* 53, 67-78.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics* 160, 246-256.

Built with AI assistance.
