#!/usr/bin/env python3
"""
TRT timeline plot – labs, injections & PK release-rate curve
with E₂ in its own subplot and Free-T reference band
"""

# ── imports ──────────────────────────────────────────────
import re, io, requests, numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse
from matplotlib.ticker import MultipleLocator
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── CONFIG ───────────────────────────────────────────────
# what to draw
PLOT_START_DATE     = '01.06.2025'
BUFFER_DAYS         = 2

# what to compute (for PK/Bateman baseline)
# - If COMPUTE_START_DATE is None, we auto-pick:
#     min(earliest dose date, PLOT_START_DATE - COMPUTE_LOOKBACK_DAYS)
COMPUTE_START_DATE  = None            # e.g. '01.03.2025' to force a date
COMPUTE_LOOKBACK_DAYS = 60            # used only when auto-picking
PRE_BORON_START = PLOT_START_DATE
BORON_START     = '07.03.2025'
FIGSIZE         = (13, 7)

# ── TRT PK model ─────────────────────────────────────────
TRT_PK_MODEL   = "elim"   # "elim" (current) or "abs"
ABS_HALF_DAYS  = 2.5      # only used if TRT_PK_MODEL == "abs"


PLOT_TT, PLOT_FT, PLOT_E2 = True, True, True
PLOT_INJ, PLOT_PK         = True, True



T_HALF_DAYS = 8
PK_FREQ     = '1h'
ACTIVE_FRAC = 288.424 / 412.61     # 0.699

# E2 limit of detection
E2_LOD_PG_ML   = 24.0     # your lab’s limit of detection, edit once here
E2_PLOT_FRACTION_OF_LOD = 0.60  # plot at 60% of LOD when censored (“<”)


# ── hCG → endogenous-T model (tunable) ───────────────────────
HCG_ENABLED = True

HCG_PARAMS = dict(
    half_life_hours = 36.0,   # hCG plasma t1/2 ~24–36 h; 36 h is a good starting point
    delay_hours     = 12.0,   # onset delay for Leydig steroidogenesis
    ec50_IU         = 450.0,  # "concentration-like" IU that gives 50% of Emax
    emax_mg_per_day = 7.5     # max extra T secretion (mg/day) at saturation
)

HCG_BAR_COLOR = 'goldenrod'   # bar color for hCG doses


PHASE_COLORS = {
    'Pre-Boron': 'tab:blue',
    'Boron Start': 'tab:orange',
    'TRT Start': 'tab:green'
}
LINE_COLORS  = {
    'TT':'tab:blue',
    'FT':'tab:green',
    'E2':'purple',
    'PK':'#cc4444'
}
BAR_COLOR    = 'gray'

INJECT_SHEET = "https://docs.google.com/spreadsheets/d/<spreadsheet_id>/edit"
MARKER_SHEET = "https://docs.google.com/spreadsheets/d/<spreadsheet_id>/edit"

# ── ensure PRE_BORON_START < BORON_START ─────────────────
dt_start = datetime.strptime(PRE_BORON_START, '%d.%m.%Y')
dt_boron = datetime.strptime(BORON_START,     '%d.%m.%Y')
if dt_start >= dt_boron:
    dt_start = dt_boron - timedelta(days=1)
PRE_BORON_START = dt_start.strftime('%d.%m.%Y')


# ── HELPERS ───────────────────────────────────────────────
def gsheet_csv(url):
    m = re.search(r'/d/([^/]+)/', url)
    return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv" if m else url

def read_any(src, **kw):
    if isinstance(src, str) and src.startswith('http'):
        txt = requests.get(gsheet_csv(src), timeout=10).text
        return pd.read_csv(io.StringIO(txt), **kw)
    return pd.read_csv(src, **kw)

def _float_from_any(x: str | float | int) -> float | None:
    """
    Robustly pull a numeric value out of whatever is in the sheet.
    Returns None if no usable number is found.
    """
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r'([\d.]+)', str(x))
    return float(m.group(1)) if m else None

def load_injections(src, *, hcg_threshold: float = 124) -> pd.DataFrame:
    """
    Read the injection sheet and return only *test-cyp* shots.

    A row is **dropped** (classified as HCG) when:
      • the “Substance” column (if present) contains “HCG”, **OR**
      • the extracted dose is > `hcg_threshold` (124 by default).

    The returned frame has two columns: datetime, mg – exactly as before.
    """
    df       = read_any(src)
    dt_col   = next(c for c in df.columns if 'date' in c.lower())
    mg_col   = next(c for c in df.columns if 'mg'   in c.lower())
    sub_col  = next((c for c in df.columns if 'substance' in c.lower()), None)

    rec = []
    for _, row in df.iterrows():
        # ----- substance check -------------------------------------------------
        substance = str(row[sub_col]).lower() if sub_col else ''
        is_hcg_by_name = 'hcg' in substance

        # ----- dose ------------------------------------------------------------
        mg = _float_from_any(row[mg_col])
        if mg is None or mg <= 0:
            continue
        is_hcg_by_dose = mg > hcg_threshold

        # ----- final decision --------------------------------------------------
        if is_hcg_by_name or is_hcg_by_dose:          # → skip (HCG)
            continue

        # ----- date ------------------------------------------------------------
        try:
            dt = parse(str(row[dt_col]), dayfirst=True, fuzzy=True)
        except Exception:
            continue
        if dt.time() == datetime.min.time():
            dt = dt.replace(hour=12)

        rec.append((dt, mg))

    return (
        pd.DataFrame(rec, columns=['datetime', 'mg'])
          .sort_values('datetime')
          .reset_index(drop=True)
    )

def fit_scale_to_tt(pk_df, tt_df, window_days=1.0):
    """
    Fit a scalar 'S' and offset 'B' so: TT_pred = S * (pk_df.value rolling-mean) + B
    minimizing squared error at TT sample times. Returns (S, B, N_pairs).
    """
    if pk_df.empty or tt_df.empty:
        return 1.0, 0.0, 0
    x = pk_df.set_index('datetime')['value'].rolling(
            pd.Timedelta(days=window_days), min_periods=1, center=True
        ).mean()
    X = x.reindex(tt_df.datetime, method='nearest',
                tolerance=pd.Timedelta(hours=12)).values
    Y = tt_df.value.values
    mask = np.isfinite(X) & np.isfinite(Y)
    n_pairs = int(mask.sum())
    if n_pairs < 2:
        return 1.0, 0.0, 0
    Xmat = np.c_[X, np.ones_like(X)]  # [value, 1]
    beta, *_ = np.linalg.lstsq(Xmat[mask], Y[mask], rcond=None)
    S, B = float(beta[0]), float(beta[1])
    if not np.isfinite(S): S = 1.0
    if not np.isfinite(B): B = 0.0
    return S, B, n_pairs

# ─────────────────────────────────────────────────────────────────────────────
# Bateman (absorption+elimination) + hCG contribution → TT prediction helpers
# ─────────────────────────────────────────────────────────────────────────────
def _bateman_conc(ts, inj_df, hcg_df, active_frac, ka_half_days, ke_half_days, HCG_PARAMS):
    """
    Return an unscaled 'concentration-like' time series for TRT + hCG on the
    timestamps `ts` (DatetimeIndex). Units are arbitrary; later mapped to ng/dL
    with a linear fit (scale + offset) against TT labs.
    """
    ts = pd.DatetimeIndex(ts)
    if len(ts) == 0:
        return np.zeros(0, dtype=float)

    ka = np.log(2) / float(ka_half_days)
    ke = np.log(2) / float(ke_half_days)
    conc = np.zeros(len(ts), dtype=float)

    # TRT depot → Bateman: (ka/(ka-ke)) * (e^{-ke t} - e^{-ka t})
    for t0, dose in inj_df[['datetime', 'mg']].itertuples(index=False):
        msk = ts >= t0
        if not np.any(msk):
            continue
        dt = (ts[msk] - t0).total_seconds() / 86400.0
        if abs(ka - ke) < 1e-9:
            term = dt * np.exp(-ke * dt)  # limit as ka→ke
        else:
            term = (np.exp(-ke * dt) - np.exp(-ka * dt)) * (ka / (ka - ke))
        conc[msk] += dose * active_frac * term

    # hCG → endogenous secretion (mg/day) with Emax; convolve with ke
    lam_hcg   = np.log(2) / (HCG_PARAMS['half_life_hours'] / 24.0)
    delay_h   = HCG_PARAMS['delay_hours']
    Emax      = HCG_PARAMS['emax_mg_per_day']
    EC50      = HCG_PARAMS['ec50_IU']
    C = np.zeros(len(ts), dtype=float)  # "effect-site" amount (IU)
    for t0, iu in hcg_df[['datetime', 'iu']].itertuples(index=False):
        t_eff = t0 + timedelta(hours=delay_h)
        msk = ts >= t_eff
        if not np.any(msk):
            continue
        dt = (ts[msk] - t_eff).total_seconds() / 86400.0
        C[msk] += iu * np.exp(-lam_hcg * dt)
    hcg_mgd = Emax * (C / (EC50 + C))  # mg/day secretion

    # discrete convolution with elimination to convert secretion→conc
    conc_hcg = np.zeros(len(ts), dtype=float)
    if len(ts) >= 2:
        dt_days = (ts[1] - ts[0]).total_seconds() / 86400.0
    else:
        dt_days = 1.0 / 24.0
    decay = np.exp(-ke * dt_days)
    for i in range(1, len(ts)):
        conc_hcg[i] = conc_hcg[i - 1] * decay + hcg_mgd[i] * dt_days

    return conc + conc_hcg


def fit_and_predict_tt_bateman(
    ts, inj_df, hcg_df, tt_df, active_frac, HCG_PARAMS,
    ka_grid=None, ke_grid=None, window_hours=24, tol_hours=12
):
    """
    Grid-search (coarse, fast) over ka, ke to minimize RMSE to TT labs.
    Returns (tt_pred_ts, metrics_dict) or (None, None) if not enough labs.
    metrics_dict: {'ka_half','ke_half','rmse','mape','r2','scale','offset','n'}
    """
    if tt_df.empty or len(tt_df) < 2:
        return None, None
    if ka_grid is None:
        ka_grid = np.linspace(3.0, 9.0, 13)     # half-life of absorption (days)
    if ke_grid is None:
        ke_grid = np.linspace(0.4, 1.4, 6)      # half-life of elimination (days)

    best = None
    best_conc = None

    for ka_half in ka_grid:
        for ke_half in ke_grid:
            conc = _bateman_conc(ts, inj_df, hcg_df, active_frac, ka_half, ke_half, HCG_PARAMS)
            x = (pd.Series(conc, index=ts)
                   .rolling(f'{int(window_hours)}h', min_periods=1, center=True)
                   .mean())
            X = x.reindex(tt_df.datetime, method='nearest', tolerance=pd.Timedelta(f'{int(tol_hours)}h'))
            mask = X.notna().values
            if mask.sum() < 2:
                continue
            Y = (pd.Series(tt_df.value.values, index=tt_df.datetime)
                   .reindex(X.index).values)
            Xv = X.values[mask]; Yv = Y[mask]
            A  = np.c_[Xv, np.ones_like(Xv)]
            beta, *_ = np.linalg.lstsq(A, Yv, rcond=None)
            S, B = float(beta[0]), float(beta[1])
            pred = S * Xv + B
            rmse = float(np.sqrt(np.mean((pred - Yv) ** 2)))
            mape = float(np.mean(np.abs((pred - Yv) / Yv)) * 100)
            r2   = float(1 - np.sum((Yv - pred) ** 2) / np.sum((Yv - np.mean(Yv)) ** 2))
            if (best is None) or (rmse < best['rmse']):
                best = {'ka_half': float(ka_half), 'ke_half': float(ke_half),
                        'rmse': rmse, 'mape': mape, 'r2': r2,
                        'scale': S, 'offset': B, 'n': int(mask.sum())}
                best_conc = conc

    if best is None:
        return None, None

    # Full-series TT prediction at best params
    tt_pred_ts = best['scale'] * best_conc + best['offset']
    return tt_pred_ts, best

# ─────────────────────────────────────────────────────────────────────────────
# Coarse→fine calibration of Bateman + hCG against TT labs (no SciPy needed)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_scale_offset_to_tt(ts_vals, tt_df, window_hours=24, tol_hours=12):
    """Return (scale, offset, n_pairs, rmse, mape, r2) aligning model→labs."""
    if tt_df.empty or len(tt_df) < 2:
        return 1.0, 0.0, 0, np.nan, np.nan, np.nan
    x = (pd.Series(ts_vals[1], index=ts_vals[0])
        .rolling(pd.Timedelta(hours=window_hours), min_periods=1, center=True)
        .mean())
    X = x.reindex(tt_df.datetime, method='nearest',
                tolerance=pd.Timedelta(hours=tol_hours))
    mask = X.notna().values
    if mask.sum() < 2:
        return 1.0, 0.0, 0, np.nan, np.nan, np.nan
    Y = (pd.Series(tt_df.value.values, index=tt_df.datetime)
           .reindex(X.index).values)
    Xv, Yv = X.values[mask], Y[mask]
    A = np.c_[Xv, np.ones_like(Xv)]
    beta, *_ = np.linalg.lstsq(A, Yv, rcond=None)
    S, B = float(beta[0]), float(beta[1])
    pred = S * Xv + B
    rmse = float(np.sqrt(np.mean((pred - Yv)**2)))
    mape = float(np.mean(np.abs((pred - Yv)/Yv)) * 100.0)
    r2   = float(1 - np.sum((Yv - pred)**2) / np.sum((Yv - np.mean(Yv))**2))
    return S, B, int(mask.sum()), rmse, mape, r2

def calibrate_bateman_and_hcg_fast(
    ts, inj_df, hcg_df, tt_df, active_frac, HCG_PARAMS,
    ka_range=(3.5, 5.5), ke_range=(0.7, 1.1),
    emax_range=(5.0, 10.0), ec50_range=(300.0, 600.0),
    stages=((9,7,7,7), (9,7,7,7)),
    tol_hours=6,  # match labs within ± tol_hours
    dt_hours=1.0, # internal grid step for full-series (plot)
    tune_hcg=True
):
    """
    Fast two-stage zoom:
      • During search: evaluate model ONLY at lab timestamps.
      • Reuse precomputed single-rate IIRs for all (ka,ke) pairs.
      • Precompute C(t) for hCG once; reuse for all (Emax,EC50).
      • For each ke: single IIR maps hcg_mgd -> conc_hcg at lab times.
    Then: build 1h full-series once at the end for plotting.

    Returns: (BEST dict, full_series_tt) as before.
    """
    import numpy as _np
    import pandas as _pd

    if tt_df.empty or len(tt_df) < 2:
        return None, None

    # --- Build a regular 1h grid that spans injections + labs (for end product)
    ts = _pd.DatetimeIndex(ts)
    grid = _pd.date_range(ts.min().floor('h'), ts.max().ceil('h'), freq=_pd.Timedelta(hours=dt_hours))
    N = len(grid)
    dt_days = dt_hours / 24.0

    # Map lab times to grid indices (nearest within tolerance)
    lab_times = _pd.DatetimeIndex(tt_df['datetime']).sort_values()
    lab_idx = _np.searchsorted(grid.values, lab_times.values)
    lab_idx = _np.clip(lab_idx, 0, N-1)
    # Drop labs too far from grid (shouldn't happen with 1h grid)
    # Keep the mask in case you later change dt_hours
    labs_ok = _np.ones(len(lab_idx), dtype=bool)

    # --- Build impulse trains on the grid
    trt_imp = _np.zeros(N, dtype=_np.float32)
    for t0, dose in inj_df[['datetime','mg']].itertuples(index=False):
        i = _np.searchsorted(grid.values, _np.datetime64(t0, 'ns'))
        if 0 <= i < N:
            trt_imp[i] += dose * active_frac

    # hCG impulses start at effect-time (delay)
    lam_hcg = _np.log(2.0) / (HCG_PARAMS['half_life_hours'] / 24.0)
    delay = _pd.to_timedelta(f"{int(HCG_PARAMS['delay_hours'])}h")
    hcg_imp = _np.zeros(N, dtype=_np.float32)
    for t0, iu in hcg_df[['datetime','iu']].itertuples(index=False):
        teff = _pd.Timestamp(t0) + delay
        i = _np.searchsorted(grid.values, _np.datetime64(teff, 'ns'))
        if 0 <= i < N:
            hcg_imp[i] += iu

    # --- Precompute C(t) for hCG once (independent of Emax/EC50)
    alpha_hcg = _np.exp(-lam_hcg * dt_days)
    C = _np.empty(N, dtype=_np.float32)
    c = 0.0
    for i in range(N):
        c = c * float(alpha_hcg) + float(hcg_imp[i])
        C[i] = c

    # --- TRT single-rate IIR cache (compute on demand; round key to avoid float-key misses)
    trt_resp = {}  # rounded(k_half) -> Y_k (float32)
    def _trt_response_for_khalf(k_half):
        k_key = float(round(k_half, 6))
        y = trt_resp.get(k_key)
        if y is None:
            k = _np.log(2.0) / float(k_half)
            alpha = _np.exp(-k * dt_days)
            y = _np.empty(N, dtype=_np.float32)
            acc = 0.0
            for i in range(N):
                acc = acc * float(alpha) + float(trt_imp[i])
                y[i] = acc
            trt_resp[k_key] = y
        return y


    # helper: score a candidate set at lab indices
    def score_candidate(ka_half, ke_half, Emax, EC50):
        # flip-flop guard
        if ka_half <= ke_half:
            return None

        # TRT at lab times from cached IIRs
        y_ke = _trt_response_for_khalf(ke_half)[lab_idx]
        y_ka = _trt_response_for_khalf(ka_half)[lab_idx]
        ka = _np.log(2.0) / float(ka_half)
        ke = _np.log(2.0) / float(ke_half)
        coef = ka / (ka - ke)  # correct Bateman coefficient (use rates, not half-lives)

        trt_lab = coef * (y_ke - y_ka)

        # hCG mg/day at lab times
        hmgd = Emax * (C / (EC50 + C))
        # map mg/day -> concentration at lab times via elimination IIR (depends only on ke)
        alpha_ke = _np.exp(-(_np.log(2.0) / float(ke_half)) * dt_days)
        z = 0.0
        z_lab = _np.empty(len(lab_idx), dtype=_np.float32)
        li = 0
        for i in range(N):
            z = z * float(alpha_ke) + float(hmgd[i]) * dt_days
            while li < len(lab_idx) and i == lab_idx[li]:
                z_lab[li] = z
                li += 1
            if li >= len(lab_idx):
                break


        # linear map to TT labs
        X = trt_lab + z_lab
        Y = tt_df['value'].values
        if len(X) != len(Y):  # should match
            m = min(len(X), len(Y))
            X, Y = X[:m], Y[:m]
        A = _np.c_[X, _np.ones_like(X)]
        beta, *_ = _np.linalg.lstsq(A, Y, rcond=None)
        S, B = float(beta[0]), float(beta[1])
        pred = S * X + B
        rmse = float(_np.sqrt(_np.mean((pred - Y) ** 2)))
        mape = float(_np.mean(_np.abs((pred - Y) / Y)) * 100.0)
        r2 = float(1 - _np.sum((Y - pred) ** 2) / _np.sum((Y - _np.mean(Y)) ** 2))
        return {'ka_half': float(ka_half), 'ke_half': float(ke_half),
                'Emax': float(Emax), 'EC50': float(EC50),
                'scale': S, 'offset': B, 'rmse': rmse, 'mape': mape, 'r2': r2, 'n': len(Y)}

    # --- Stagewise zoom over (ka, ke, Emax, EC50)
    def zoom(box, grids):
        (ka_lo, ka_hi), (ke_lo, ke_hi), (emax_lo, emax_hi), (ec50_lo, ec50_hi) = box
        (Nka, Nke, Nemax, Nec50) = grids
        kas   = _np.linspace(ka_lo,   ka_hi,   Nka)
        kes   = _np.linspace(ke_lo,   ke_hi,   Nke)
        emaxs = _np.linspace(emax_lo, emax_hi, Nemax) if tune_hcg else [HCG_PARAMS['emax_mg_per_day']]
        ec50s = _np.linspace(ec50_lo, ec50_hi, Nec50) if tune_hcg else [HCG_PARAMS['ec50_IU']]
        best = None
        for ka_half in kas:
            for ke_half in kes:
                for Emax in emaxs:
                    for EC50 in ec50s:
                        m = score_candidate(ka_half, ke_half, Emax, EC50)
                        if m is None:  # failed guard
                            continue
                        if (best is None) or (m['rmse'] < best['rmse']):
                            best = m
        # shrink box around best
        def shrink(lo, hi, c, frac=0.45):
            w = (hi - lo) * frac
            return max(c - w, lo), min(c + w, hi)
        out = ((shrink(ka_lo, ka_hi, best['ka_half'])),
               (shrink(ke_lo, ke_hi, best['ke_half'])),
               (shrink(emax_lo, emax_hi, best['Emax']) if tune_hcg else (emax_lo, emax_hi)),
               (shrink(ec50_lo, ec50_hi, best['EC50']) if tune_hcg else (ec50_lo, ec50_hi)))
        return best, out

    # initial box
    box = (ka_range, ke_range, emax_range, ec50_range)
    best = None
    for stage in stages:
        best, box = zoom(box, stage)

    if best is None:
        return None, None

    # --- Build full 1h series at BEST for plotting
    ka_best, ke_best = best['ka_half'], best['ke_half']
    Emax_best, EC50_best = best['Emax'], best['EC50']
    # TRT full
    y_ke_full = _trt_response_for_khalf(ke_best)
    y_ka_full = _trt_response_for_khalf(ka_best)
    ka = _np.log(2.0) / float(ka_best)
    ke = _np.log(2.0) / float(ke_best)
    coef_full = ka / (ka - ke)
    conc_trt_full = coef_full * (y_ke_full - y_ka_full)

    # hCG full
    hmgd_full = Emax_best * (C / (EC50_best + C))
    alpha_ke_full = _np.exp(-(_np.log(2.0) / float(ke_best)) * dt_days)
    z = _np.empty(N, dtype=_np.float32); acc = 0.0
    for i in range(N):
        acc = acc * float(alpha_ke_full) + float(hmgd_full[i]) * dt_days
        z[i] = acc
    conc_full = conc_trt_full + z
    tt_full = best['scale'] * conc_full + best['offset']

    # Bring back to pk_df datetime granularity: our 'grid' already matches 1h ticks
    # Return BEST + full TT series aligned to 'grid'
    BEST = best
    TT_SERIES = _pd.Series(tt_full, index=grid)
    # Align to pk_df datetime if needed by caller
    return BEST, TT_SERIES.reindex(ts, method='nearest', tolerance=_pd.Timedelta(hours=dt_hours))


def calibrate_bateman_and_hcg(
    ts, inj_df, hcg_df, tt_df, active_frac, HCG_PARAMS,
    ka_range=(3.0, 7.0), ke_range=(0.6, 1.4),
    emax_range=(4.0, 12.0), ec50_range=(250.0, 700.0),
    stages=((7, 5, 5, 5), (7, 5, 5, 5)),    # grid sizes per stage
    window_hours=24, tol_hours=12,
    tune_hcg=True

):
    """
    Successive grid zoom. Returns dict with best params & metrics and a full-series tt_pred.
    """
    ts = pd.DatetimeIndex(ts)

    def eval_one(ka_half, ke_half, emax, ec50):
        hcg_cfg = HCG_PARAMS.copy()
        hcg_cfg['emax_mg_per_day'] = emax
        hcg_cfg['ec50_IU']         = ec50
        conc = _bateman_conc(ts, inj_df, hcg_df, active_frac,
                             ka_half, ke_half, hcg_cfg)
        S, B, n, rmse, mape, r2 = _fit_scale_offset_to_tt(
            (ts, conc), tt_df,
            window_hours=window_hours, tol_hours=tol_hours
        )
        return {'ka_half':ka_half, 'ke_half':ke_half,
                'Emax':emax, 'EC50':ec50,
                'scale':S, 'offset':B,
                'rmse':rmse, 'mape':mape, 'r2':r2, 'n':n}, conc

    # Stagewise zoom
    lo_ka, hi_ka   = ka_range
    lo_ke, hi_ke   = ke_range
    lo_emax, hi_emax = emax_range
    lo_ec50, hi_ec50 = ec50_range

    best = None
    best_conc = None

    for (Nka, Nke, Nemax, Nec50) in stages:
        kas  = np.linspace(lo_ka,  hi_ka,  Nka)
        kes  = np.linspace(lo_ke,  hi_ke,  Nke)
        emaxs = np.linspace(lo_emax, hi_emax, Nemax) if tune_hcg else [HCG_PARAMS['emax_mg_per_day']]
        ec50s = np.linspace(lo_ec50, hi_ec50, Nec50) if tune_hcg else [HCG_PARAMS['ec50_IU']]

        for ka_half in kas:
            for ke_half in kes:
                for Emax in emaxs:
                    for EC50 in ec50s:
                        if ka_half <= ke_half:
                            continue
                        m, conc = eval_one(ka_half, ke_half, Emax, EC50)
                        if (best is None) or (m['rmse'] < best['rmse']):
                            best, best_conc = m, conc

        # shrink ranges around best (guard for identical lo/hi)
        def shrink(lo, hi, center, factor=0.45):
            width = (hi - lo) * factor
            return max(center - width, lo), min(center + width, hi)

        lo_ka, hi_ka   = shrink(lo_ka,  hi_ka,  best['ka_half'])
        lo_ke, hi_ke   = shrink(lo_ke,  hi_ke,  best['ke_half'])
        if tune_hcg:
            lo_emax, hi_emax = shrink(lo_emax, hi_emax, best['Emax'])
            lo_ec50, hi_ec50 = shrink(lo_ec50, hi_ec50, best['EC50'])

    # Final full-series TT prediction at best params
    tt_pred = best['scale'] * best_conc + best['offset']
    return best, tt_pred


def load_hcg_injections(src, *, hcg_threshold: float = 124) -> pd.DataFrame:
    """
    Read the same injection sheet and return *only* hCG rows.
    Detection:
      • 'Substance' contains 'HCG'  OR
      • numeric dose > hcg_threshold   (useful when IU are in that column)
    Returns: DataFrame[datetime, iu]
    """
    df       = read_any(src)
    dt_col   = next(c for c in df.columns if 'date' in c.lower())
    mg_col   = next(c for c in df.columns if 'mg'   in c.lower())
    sub_col  = next((c for c in df.columns if 'substance' in c.lower()), None)

    rec = []
    for _, row in df.iterrows():
        # --- date ---
        try:
            dt = parse(str(row[dt_col]), dayfirst=True, fuzzy=True)
        except Exception:
            continue
        if dt.time() == datetime.min.time():
            dt = dt.replace(hour=12)

        # --- substance / dose checks ---
        substance = str(row[sub_col]).lower() if sub_col else ''
        is_hcg_by_name = 'hcg' in substance

        dose_val = _float_from_any(row[mg_col])   # may actually be IU in your sheet
        if dose_val is None or dose_val <= 0:
            continue
        is_hcg_by_dose = dose_val > hcg_threshold

        if is_hcg_by_name or is_hcg_by_dose:
            rec.append((dt, dose_val))           # treat the numeric entry as IU

    return (
        pd.DataFrame(rec, columns=['datetime', 'iu'])
          .sort_values('datetime')
          .reset_index(drop=True)
    )


def load_markers(src: str, name: str) -> pd.DataFrame:
    df = read_any(src)
    fc = df.columns[0]
    rows = df[df[fc].astype(str).str.contains(name, case=False, na=False)]
    if rows.empty:
        return pd.DataFrame(columns=['datetime','value','raw'])
    key = name.lower()

    # Estradiol → prefer pg/mL
    if key in ('e2','estradiol'):
        is_pg = rows.apply(lambda r: r.astype(str).str.contains('pg', case=False, na=False).any(), axis=1)
        row    = rows[is_pg].iloc[0] if is_pg.any() else rows.iloc[0]
        factor = 1.0 if is_pg.any() else 0.272

    # Free T → also prefer pg/mL
    elif key in ('free t','free testosterone'):
        is_pg = rows.apply(lambda r: r.astype(str).str.contains('pg', case=False, na=False).any(), axis=1)
        row    = rows[is_pg].iloc[0] if is_pg.any() else rows.iloc[0]
        factor = 1.0

    # Total T → prefer ng/dL
    else:
        is_ng = rows[fc].str.contains('ng/dl', case=False, na=False)
        row    = rows[is_ng].iloc[0] if is_ng.any() else rows.iloc[0]
        factor = 1.0 if is_ng.any() else 28.818

    rec = []
    for col, val in zip(df.columns[1:], row.iloc[1:]):
        try:
            dt = parse(str(col), dayfirst=True, fuzzy=True)
        except:
            continue
        s   = str(val).strip()
        raw = None
        if key in ('e2','estradiol') and s.startswith('<'):
            try:
                v = float(s.lstrip('<')) * factor
            except:
                continue
            raw = s
        else:
            try:
                v = float(s) * factor
            except:
                continue
        if v > 0:
            rec.append((dt, v, raw))
    return pd.DataFrame(rec, columns=['datetime','value','raw']).sort_values('datetime')

def dynamic_fig_width(plot_start, plot_end, inj_count, lab_points,
                      base_width=FIGSIZE[0],        # keep your default as the floor
                      days_per_inch=4.0,            # ~4 days per inch target
                      events_per_inch=2.2,          # ~2.2 events per inch target
                      min_width=10.0,               # absolute min
                      max_width=32.0):              # absolute max (keeps files sane)
    """
    Choose a width large enough for:
      • the date span (days_per_inch)
      • the total plotted points (events_per_inch), with labs discounted a bit
    Then clamp between min_width and max_width, and never below base_width.
    """
    days = max(1, (plot_end - plot_start).days)
    w_from_days   = days / days_per_inch
    w_from_events = (inj_count + 0.5 * lab_points) / events_per_inch  # labs count half
    return float(np.clip(max(base_width, w_from_days, w_from_events), min_width, max_width))

# ── LOAD & FILTER ────────────────────────────────────────
plot_start = datetime.strptime(PLOT_START_DATE, '%d.%m.%Y')

# load ALL data first
inj_all = load_injections(INJECT_SHEET)
hcg_all = load_hcg_injections(INJECT_SHEET)

# choose compute_start
if COMPUTE_START_DATE:
    compute_start = datetime.strptime(COMPUTE_START_DATE, '%d.%m.%Y')
else:
    earliest = min(
        [d.datetime.min() for d in (inj_all, hcg_all) if not d.empty] or [plot_start]
    )
    compute_start = min(earliest, plot_start - timedelta(days=COMPUTE_LOOKBACK_DAYS))

# frames used for DRAWING (trimmed at plot start)
inj_df = inj_all[inj_all.datetime >= plot_start].copy()
hcg_df = hcg_all[hcg_all.datetime >= plot_start].copy()

tt_df = load_markers(MARKER_SHEET, "Total T")
ft_df = load_markers(MARKER_SHEET, "Free T")
e2_df = load_markers(MARKER_SHEET, "Estradiol")
# drops prestart data usage for calibration
# for df in (tt_df, ft_df, e2_df):
#     df.drop(df[df.datetime < plot_start].index, inplace=True)

TRT_START_DT = inj_all.datetime.min() if not inj_all.empty else plot_start
last_point   = max(
    [d.datetime.max() for d in (inj_df, hcg_df, tt_df, ft_df, e2_df) if not d.empty] 
    or [plot_start]
)
plot_end = last_point + timedelta(days=BUFFER_DAYS)

# count points to decide how wide the figure should be
total_lab_points = sum(len(df) for df in (tt_df, ft_df, e2_df) if not df.empty)
FIG_W = dynamic_fig_width(plot_start, plot_end,
                          inj_count=(len(inj_df) + len(hcg_df)),
                          lab_points=total_lab_points)


# for E2 LOD handling
if not e2_df.empty:
    e2_df['plot_val'] = e2_df.apply(
        lambda r: (E2_PLOT_FRACTION_OF_LOD * E2_LOD_PG_ML) if isinstance(r.raw, str) and r.raw.strip().startswith('<')
                  else r.value,
        axis=1
    )
    e2_df['disp_text'] = e2_df.apply(
        lambda r: r.raw if isinstance(r.raw, str) and r.raw.strip().startswith('<')
                  else str(int(r.value)),
        axis=1
    )

# PK curve (mg/day active T)
pk_df = pd.DataFrame()
if PLOT_PK and (not inj_all.empty or (HCG_ENABLED and not hcg_all.empty)):
    ts  = pd.date_range(compute_start, plot_end, freq=PK_FREQ)

    # total release and hCG-only component
    rel       = np.zeros(len(ts), dtype=float)
    hcg_only  = np.zeros(len(ts), dtype=float)

    # --- Add endogenous T secretion driven by hCG ---
    if HCG_ENABLED and not hcg_all.empty:
        lam_hcg = np.log(2) / (HCG_PARAMS['half_life_hours'] / 24.0)  # [1/day]
        Emax    = HCG_PARAMS['emax_mg_per_day']                       # mg/day
        EC50    = HCG_PARAMS['ec50_IU']                               # IU

        # "Concentration-like" effect-compartment amount
        C = np.zeros(len(ts), dtype=float)
        for t0, iu in hcg_all[['datetime','iu']].itertuples(index=False):
            t_effect = t0 + timedelta(hours=HCG_PARAMS['delay_hours'])
            msk = ts >= t_effect
            if not np.any(msk):
                continue
            dt = (ts[msk] - t_effect).total_seconds() / 86400.0       # days since effect start
            C[msk] += iu * np.exp(-lam_hcg * dt)

        # Emax nonlinearity → extra T secretion (mg/day)
        hcg_only = Emax * (C / (EC50 + C))
        rel += hcg_only

    # --- TRT release-rate impulses (always computed if injections exist) ---
    ke = np.log(2) / T_HALF_DAYS
    if TRT_PK_MODEL == "abs":
        ka = np.log(2) / ABS_HALF_DAYS

    for t0, dose in inj_all[['datetime','mg']].itertuples(index=False):
        msk = ts >= t0
        if not np.any(msk):
            continue
        dt = (ts[msk] - t0).total_seconds() / 86400.0

        if TRT_PK_MODEL == "elim":
            rel[msk] += dose * ACTIVE_FRAC * np.exp(-ke * dt) * ke
        else:
            rel[msk] += dose * ACTIVE_FRAC * np.exp(-ka * dt) * ka


    pk_df = pd.DataFrame({
        'datetime': ts,
        'value':    rel,
        'hcg_extra': hcg_only,   # keep for plotting/tuning later
    })
    # Fit scale/offset so PK (mg/day) tracks Total T labs (ng/dL)
    TT_SCALE, TT_BASE, TT_N = fit_scale_to_tt(pk_df, tt_df, window_days=1.0)
    if TT_N >= 2:
        pk_df['tt_pred'] = TT_SCALE * pk_df['value'] + TT_BASE

# ── Bateman model: fit (ka, ke, scale, offset) and add 'tt_bateman' ──
if PLOT_PK and not pk_df.empty and not inj_df.empty and (not tt_df.empty):
    tt_bateman, BATEMAN_METRICS = fit_and_predict_tt_bateman(
        ts=pd.DatetimeIndex(pk_df['datetime']),
        inj_df=inj_all, hcg_df=hcg_all,
        tt_df=tt_df, active_frac=ACTIVE_FRAC,
        HCG_PARAMS=HCG_PARAMS,
        ka_grid=np.linspace(3.0, 9.0, 13),     # tune if you wish
        ke_grid=np.linspace(0.4, 1.4, 6),      # tune if you wish
        window_hours=24, tol_hours=12
    )
    if tt_bateman is not None:
        pk_df['tt_bateman'] = tt_bateman
        # Optional: console summary for quick sanity check
        print(
            "Bateman fit:",
            f"ka½={BATEMAN_METRICS['ka_half']:.2f} d,",
            f"ke½={BATEMAN_METRICS['ke_half']:.2f} d,",
            f"n={BATEMAN_METRICS['n']},",
            f"RMSE={BATEMAN_METRICS['rmse']:.1f} ng/dL,",
            f"MAPE={BATEMAN_METRICS['mape']:.1f}%,",
            f"R²={BATEMAN_METRICS['r2']:.3f}"
        )

# ── Bateman model: full calibration (ka½, ke½, and optionally hCG Emax/EC50)
CALIBRATE_BATEMAN = True      # toggle
if CALIBRATE_BATEMAN and PLOT_PK and not pk_df.empty and not inj_df.empty and (not tt_df.empty):
    E0 = HCG_PARAMS['emax_mg_per_day']
    EC = HCG_PARAMS['ec50_IU']
    BEST, tt_bate_cal = calibrate_bateman_and_hcg_fast(
        ts=pd.DatetimeIndex(pk_df['datetime']),
        inj_df=inj_all, hcg_df=hcg_all, tt_df=tt_df,
        active_frac=ACTIVE_FRAC, HCG_PARAMS=HCG_PARAMS,
        ka_range=(3.5, 5.5), ke_range=(0.7, 1.1),
        emax_range=(5.0, 10.0), ec50_range=(300.0, 600.0),
        stages=((9,7,7,7), (9,7,7,7)),
        tol_hours=6, dt_hours=1.0,
        tune_hcg=True
    )
    if tt_bate_cal is not None:
        pk_df['tt_bateman'] = tt_bate_cal.values


    # Optional: print a clean summary in console
    print("Calibrated Bateman:",
          f"ka½={BEST['ka_half']:.2f} d,",
          f"ke½={BEST['ke_half']:.2f} d,",
          f"Emax={BEST['Emax']:.2f} mg/day,",
          f"EC50={BEST['EC50']:.0f} IU,",
          f"scale={BEST['scale']:.2f}, offset={BEST['offset']:.1f},",
          f"n={BEST['n']}, RMSE={BEST['rmse']:.1f}, MAPE={BEST['mape']:.1f}%, R²={BEST['r2']:.3f}")




# ── PLOTTING ─────────────────────────────────────────────
has_e2 = PLOT_E2 and not e2_df.empty
if has_e2:
    fig, (ax, ax_e2) = plt.subplots(
        2, 1, sharex=True,
        figsize=(FIG_W, FIGSIZE[1]),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.04}
    )
else:
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIGSIZE[1]))
    ax_e2 = None
ax_inj = None
ax.grid(True, which='major', alpha=0.18)
ax.grid(True, which='minor', alpha=0.09)
ax.yaxis.set_minor_locator(MultipleLocator(50))



# — Total-T & Free-T on top axis —
if PLOT_TT and not tt_df.empty:
    ax.plot(tt_df.datetime, tt_df.value,
            'o-', lw=2,
            color=LINE_COLORS['TT'],
            label='Total T (ng/dL)')
    ax.axhspan(500, 1200, alpha=0.25, color=LINE_COLORS['TT'], label='Total-T Ref 500–1200')
    ax.axhline(300, color='orange', linestyle='--', label='Low-T 300')
    ax.axhline(1500, color='red', linestyle='--', label='Supra-T 1500')
    for d, v in zip(tt_df.datetime, tt_df.value):
        ax.annotate(f"{int(v)}", (d, v),
                    xytext=(0, 6), textcoords='offset points',
                    ha='center', fontsize=7)

if PLOT_FT and not ft_df.empty:
    ax.plot(ft_df.datetime, ft_df.value,
            'o-', lw=2,
            color=LINE_COLORS['FT'],
            label='Free T (pg/mL)')
    # Free-T reference band 144–288
    ax.axhspan(144, 288, alpha=0.25, color=LINE_COLORS['FT'], label='Free-T Ref 144–288 pg/mL')
    for d, v in zip(ft_df.datetime, ft_df.value):
        if pd.notna(v):
            ax.annotate(f"{int(v)}", (d, v),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='top',
                        fontsize=7, color=LINE_COLORS['FT'])


# — injections (bars) & PK on twin of top —
if PLOT_INJ and not inj_df.empty:
    ax_inj = ax.twinx()
    ax_inj.spines['right'].set_position(('outward', 0))
    bars = ax_inj.bar(inj_df.datetime, inj_df.mg,
                      width=0.6, alpha=0.5,
                      color=BAR_COLOR,
                      label='Injected Dose (mg)')
    ax_inj.set_ylabel('TRT Dose (mg) / Release (mg/day)')
    ax_inj.set_ylim(0, max(inj_df.mg) * 1.3)
    # bar labels & faint verticals
    for bar, dt in zip(bars, inj_df.datetime):
        x = bar.get_x() + bar.get_width() / 2
        h = bar.get_height()
        # dose number
        ax_inj.annotate(f"{int(h)}", (x, h),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)
        # faint full-height line
        ax.axvline(dt, color=BAR_COLOR, alpha=0.1, linewidth=1, zorder=0, linestyle='--')

# — PK curve on the same twin axis as injections —
if PLOT_PK and not pk_df.empty:
    pk_view = pk_df[pk_df['datetime'] >= plot_start]
    if ax_inj is None:
        ax_inj = ax.twinx()
        ax_inj.set_ylabel('Release (mg/day)')

    # PK release line
    ax_inj.plot(pk_view.datetime, pk_view.value,
                '-', lw=1.8,
                color=LINE_COLORS['PK'],
                label='Predicted release (mg/day)')
    ax_inj.fill_between(pk_view.datetime, pk_view.value,
                        alpha=0.08, color=LINE_COLORS['PK'])

    # overlay PK→TT fitted prediction (thin line on main axis)
    if 'tt_pred' in pk_df.columns:
        ax.plot(pk_view.datetime, pk_view['tt_pred'],
                '-', lw=1.0, alpha=0.5, color='grey',
                label='TT fit from PK')

    # Bateman (abs+elim) TT fit overlay
    if 'tt_bateman' in pk_df.columns:
        s_view = (pd.Series(pk_view['tt_bateman'].values, index=pk_view['datetime'])
                .rolling(pd.Timedelta(hours=12), center=True, min_periods=1)
                .mean())
        # align smoothed view back to the FULL pk_df index (same length as pk_df)
        s_full = s_view.reindex(pd.DatetimeIndex(pk_df['datetime']))
        pk_df['tt_bateman_smooth'] = s_full.values

        ax.plot(pk_view.datetime,
                s_view.reindex(pk_view['datetime']).values,
                '-', lw=1.3, alpha=0.9, color='black',
                label='TT Bateman (smoothed)')


    # ±RMSE band around Bateman (use BEST['rmse'] from calibration)
    if 'tt_bateman_smooth' in pk_df.columns and 'BEST' in globals():
        try:
            rmse_band = float(BEST.get('rmse', np.nan))
        except Exception:
            rmse_band = np.nan
        if np.isfinite(rmse_band):
            yb = (pd.Series(pk_df['tt_bateman_smooth'].values, index=pk_df['datetime'])
                    .reindex(pk_view['datetime'])
                    .values)
            ax.fill_between(pk_view['datetime'], yb - rmse_band, yb + rmse_band,
                            alpha=0.12, color='black', label='Bateman ±RMSE')



    # hCG-only contribution (optional, dashed)
    if 'hcg_extra' in pk_df.columns and float(pk_df['hcg_extra'].max()) > 0:
        ax_inj.plot(pk_view.datetime, pk_view['hcg_extra'], '--', lw=1.4, label='hCG-derived T (mg/day)')

    # ensure y-limits fit both bars and PK/hCG curves
    y_max = 0.0
    if PLOT_INJ and not inj_df.empty:
        y_max = max(y_max, float(np.nanmax(inj_df.mg.values)))
    y_max = max(y_max, float(np.nanmax(pk_view.value.values)))
    if 'hcg_extra' in pk_df.columns:
        y_max = max(y_max, float(np.nanmax(pk_view['hcg_extra'].values)))

    ax_inj.set_ylim(0, y_max * 1.15)


    # annotate each peak
    inj_times = list(inj_df.datetime) + [plot_end]
    for t0,t1 in zip(inj_times[:-1], inj_times[1:]):
        seg = pk_df[(pk_df.datetime >= t0) & (pk_df.datetime < t1)]
        if seg.empty:
            continue
        idx = seg.value.idxmax()
        tp  = seg.loc[idx,'datetime']
        vp  = seg.loc[idx,'value']
        ax_inj.annotate(f"{vp:.1f}", (tp, vp),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=7,
                        color=LINE_COLORS['PK'])

# — hCG dose bars on a second right axis —
if not hcg_df.empty:
    ax_hcg = ax.twinx()
    ax_hcg.spines['right'].set_position(('outward', 36))  # shove out so scales don't overlap
    bars_hcg = ax_hcg.bar(hcg_df.datetime, hcg_df.iu,
                          width=0.45, alpha=0.35,
                          color=HCG_BAR_COLOR, label='hCG Dose (IU)')
    ax_hcg.set_ylabel('hCG Dose (IU)')
    ax_hcg.set_ylim(0, max(hcg_df.iu) * 1.3)

# — Phase shading on top —
phases = {
    'Pre-Boron': datetime.strptime(PRE_BORON_START, '%d.%m.%Y'),
    'Boron Start': datetime.strptime(BORON_START,     '%d.%m.%Y'),
}
if not inj_df.empty:
    phases['TRT Start'] = TRT_START_DT

for i, (lbl, raw_start) in enumerate(phases.items()):
    start = max(raw_start, plot_start)
    end   = list(phases.values())[i+1] if i+1 < len(phases) else plot_end
    col   = PHASE_COLORS[lbl]
    ax.axvspan(start, end, color=col, alpha=0.05)
    for tgt in (ax, ax_inj) if ax_inj else (ax,):
        tgt.axvline(start, color=col, linestyle='--', linewidth=1.3)
    if raw_start >= plot_start:
        ax.text(start, ax.get_ylim()[1]*0.98,
                f"{lbl}\n{raw_start.strftime('%d-%m-%Y')}",
                ha='center', va='bottom',
                fontsize=7, color=col,
                backgroundcolor='white')


# — E₂ subplot at bottom —
if PLOT_E2 and not e2_df.empty:
    ax_e2.plot(e2_df.datetime, e2_df.plot_val,
               's--', lw=1.5,
               color=LINE_COLORS['E2'],
               label='E2 (pg/mL)')
    ax_e2.axhspan(20, 40,
                  alpha=0.25,
                  color=LINE_COLORS['E2'],
                  label='E₂ Ref 20–40')
    ax_e2.set_ylabel('E₂ (pg/mL)', color=LINE_COLORS['E2'])
    ax_e2.tick_params(axis='y', labelcolor=LINE_COLORS['E2'])
    ax_e2.set_xlabel('Date')

    for d, y, txt in zip(e2_df.datetime,
                         e2_df.plot_val,
                         e2_df.disp_text):
        ax_e2.annotate(txt, (d, y),
                       xytext=(0, -8),
                       textcoords='offset points',
                       ha='center',
                       fontsize=7,
                       color=LINE_COLORS['E2'])


# — X-axis formatting & legend —
locator = mdates.AutoDateLocator(minticks=6, maxticks=int(FIG_W * 1.8))
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
ax.set_xlim(plot_start, plot_end)

#ax.set_xlabel('Date')

# ax.tick_params(
#     axis='x',
#     which='major',
#     labelbottom=True,
# )


# build a unique legend across axes
handles, labels = [], []
for axis in (ax, ax_inj, (ax_hcg if 'ax_hcg' in locals() else None), ax_e2):
    if axis is None:
        continue
    h, l = axis.get_legend_handles_labels()
    for hh, ll in zip(h, l):
        if ll and ll not in labels:
            handles.append(hh); labels.append(ll)

# place the legend OUTSIDE the bottom E2 panel on the right
leg = ax_e2.legend(
    handles, labels,
    fontsize=6, frameon=True,
    loc='upper left', bbox_to_anchor=(1.005, 1.0),  # just to the right of ax_e2
    borderaxespad=0.0,
    ncol=1  # set to 2 if you want a wider but shorter legend
)

# ensure there's room on the right edge for the legend
fig.subplots_adjust(right=0.84)



# Re-order z to keep lines above bars
for line in ax.lines:
    try:
        line.set_zorder(3)
    except Exception:
        pass
if 'ax_inj' in locals() and ax_inj is not None:
    for c in ax_inj.collections:
        try:
            c.set_zorder(1)
        except Exception:
            pass
if 'ax_hcg' in locals():
    for c in ax_hcg.collections:
        try:
            c.set_zorder(1)   # hCG bars behind lines
        except Exception:
            pass
plt.suptitle('Hormone Levels, Injections & Predicted Testosterone Release')
plt.savefig("TRT_timeline_plot.png", dpi=300, bbox_inches='tight')
plt.show()