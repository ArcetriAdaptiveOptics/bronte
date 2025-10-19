
import os
import csv
import math
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib.ticker import AutoMinorLocator, MaxNLocator


# =========================
# 1) Lettura CSV DAQami
# =========================
def get_data_from_DAQamiCSVfile(csv_file_name, Nrows_to_skip=7):
    time_in_sec, pd1_voltage_in_volt, pd2_voltage_in_volt = [], [], []
    with open(csv_file_name, newline='') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx < Nrows_to_skip or not row or len(row) < 4:
                continue
            try:
                t = float(row[1]); v1 = float(row[2]); v2 = float(row[3])
            except Exception:
                to_float = lambda x: float(str(x).replace(',', '.'))
                try:
                    t = to_float(row[1]); v1 = to_float(row[2]); v2 = to_float(row[3])
                except Exception:
                    continue
            time_in_sec.append(t); pd1_voltage_in_volt.append(v1); pd2_voltage_in_volt.append(v2)
    return np.array(time_in_sec), np.array(pd1_voltage_in_volt), np.array(pd2_voltage_in_volt)


# =========================
# 2) Utility di segnale
# =========================
def _moving_average(x: np.ndarray, N: int) -> np.ndarray:
    if N <= 1:
        return x.copy()
    k = np.ones(N, float) / N
    return np.convolve(x, k, mode="same")


def _linear_interp_time(t: np.ndarray, y: np.ndarray, level: float, i0: int, i1: int) -> Optional[float]:
    if i1 < i0:
        i0, i1 = i1, i0
    tt = t[i0:i1+1]; yy = y[i0:i1+1]
    diff = yy - level
    for k in range(len(diff) - 1):
        a, b = diff[k], diff[k+1]
        if a == 0.0:
            return float(tt[k])
        if a * b <= 0.0:
            y0, y1 = yy[k], yy[k+1]
            t0, t1 = tt[k], tt[k+1]
            if y1 == y0:
                return float(t0)
            frac = (level - y0) / (y1 - y0)
            return float(t0 + frac * (t1 - t0))
    return None


def _estimate_slope(t: np.ndarray, y: np.ndarray, t_cross: float, window_ms: float = 1.0) -> Optional[float]:
    half = (window_ms / 1000.0) / 2.0
    lo, hi = t_cross - half, t_cross + half
    m = (t >= lo) & (t <= hi)
    if np.sum(m) >= 3:
        tt = t[m]; yy = y[m]
        tt0 = tt - np.mean(tt)
        denom = np.sum(tt0 * tt0)
        if denom <= 0:
            return None
        a = np.sum(tt0 * (yy - np.mean(yy))) / denom
        return float(a)
    i = int(np.argmin(np.abs(t - t_cross)))
    i0, i1 = max(0, i-1), min(len(t)-1, i+1)
    if i1 == i0:
        return None
    return float((y[i1] - y[i0]) / (t[i1] - t[i0]))


def _noise_std_baseline(sig: np.ndarray, i_center: int, fs: float, pre_ms: float = 5.0) -> float:
    n_pre = max(3, int(round(fs * (pre_ms / 1000.0))))
    lo = max(0, i_center - n_pre); hi = i_center
    if hi - lo < 3:
        return float(np.std(sig[:max(5, len(sig)//100)]))
    return float(np.std(sig[lo:hi]))


def _fit_exp_linearized(t: np.ndarray, y: np.ndarray, kind: str, pre_level: float, post_level: float) -> Tuple[Optional[float], Optional[float]]:
    # Fit mono-esponenziale verso l'asintoto finale y_inf = post_level.
    # rising:  ln(y_inf - y) = a + b t  -> b ≈ -1/tau
    # falling: ln(y - y_inf) = a + b t  -> b ≈ -1/tau
    y_inf = post_level
    z = (y_inf - y) if kind == "rising" else (y - y_inf)
    m = z > 0
    if np.sum(m) < 5:
        return None, None
    tt = t[m]; lnz = np.log(z[m])
    p_lo, p_hi = np.percentile(lnz, [10, 90])
    m2 = (lnz >= p_lo) & (lnz <= p_hi)
    if np.sum(m2) < 5:
        m2 = np.ones_like(lnz, dtype=bool)
    X = np.vstack([np.ones(np.sum(m2)), tt[m2]]).T
    yvec = lnz[m2]
    beta, _, _, _ = np.linalg.lstsq(X, yvec, rcond=None)
    b = beta[1]
    if b >= 0:
        return None, None
    tau = -1.0 / b
    yhat = X @ beta
    ss_res = float(np.sum((yvec - yhat)**2))
    ss_tot = float(np.sum((yvec - np.mean(yvec))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else None
    return float(tau), (float(r2) if r2 is not None else np.nan)


def _latency_diagnostic_plots(t, v, t50, T, delta, label="PD1"):
    if len(t50) == 0 or not np.isfinite(T) or not np.isfinite(delta):
        print("[WARN] nothing to plot for latency diagnostics.")
        return
    g_first = math.floor((t[0] - delta) / T)
    g_last  = math.ceil((t[-1] - delta) / T)
    grid_times = delta + np.arange(g_first, g_last + 1) * T

    plt.figure()
    plt.plot(t, v, label=label)
    plt.scatter(t50, np.interp(t50, t, v), marker='o', s=25, label="t50")
    for gt in grid_times:
        plt.axvline(gt, alpha=0.25)
    plt.title(f"{label}: grid overlay (δ={delta*1e3:.2f} ms, T={T*1e3:.2f} ms)")
    plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)"); plt.legend(); plt.show()

    phase = np.mod(t50 - delta, T)
    plt.figure()
    plt.plot(np.arange(len(phase)), phase*1e3, 'o-')
    plt.title(f"{label}: phase-fold t50 (mean={np.mean(phase)*1e3:.2f} ms, std={np.std(phase, ddof=1)*1e3:.2f} ms)")
    plt.xlabel("edge index"); plt.ylabel("phase (ms)"); plt.show()

    plt.figure()
    bins = max(6, int(round(math.sqrt(len(phase)))))
    plt.hist(phase*1e3, bins=bins)
    plt.title(f"{label}: phase histogram"); plt.xlabel("phase (ms)"); plt.ylabel("count"); plt.show()

    n = np.round((t50 - delta) / T).astype(int)
    t_fit = delta + n * T
    r = t50 - t_fit
    plt.figure()
    plt.plot(n, t50*1e3, 'o', label="t50 data")
    plt.plot(n, t_fit*1e3, '-', label="δ + n·T")
    plt.title(f"{label}: t50 vs index (RMS resid={np.sqrt(np.mean(r*r))*1e3:.2f} ms)")
    plt.xlabel("toggle index n"); plt.ylabel("time (ms)"); plt.legend(); plt.show()

    plt.figure()
    plt.plot(t50, r*1e3, 'o-')
    plt.axhline(0, color='k', alpha=0.3)
    s = np.std(r, ddof=1) if len(r) > 1 else np.nan
    plt.axhline(+s*1e3, color='r', alpha=0.3, linestyle='--')
    plt.axhline(-s*1e3, color='r', alpha=0.3, linestyle='--')
    plt.title(f"{label}: residuals vs time (std={s*1e3:.2f} ms)")
    plt.xlabel("time (s)"); plt.ylabel("residual (ms)"); plt.show()


# ========= Helpers per latenza assoluta (t0 noto) =========
def _robust_dwell_from_edges(r_times: np.ndarray, f_times: np.ndarray) -> float:
    Td_r = 0.5*np.median(np.diff(np.sort(r_times))) if len(r_times) >= 2 else np.nan
    Td_f = 0.5*np.median(np.diff(np.sort(f_times))) if len(f_times) >= 2 else np.nan
    Td = np.nanmedian([Td_r, Td_f])
    return float(Td) if np.isfinite(Td) and Td > 0 else np.nan

def _build_trigger_grids_from_t0(t0: float, Td: float, tmin: float, tmax: float, first_kind: str):
    assert first_kind in ("rise", "fall")
    pad = 2*Td
    lo = tmin - pad; hi = tmax + pad
    if first_kind == "rise":
        phi_r = t0;      phi_f = t0 + Td
    else:
        phi_f = t0;      phi_r = t0 + Td
    def grid(phi):
        n_lo = int(math.floor((lo - phi)/Td)) - 1
        n_hi = int(math.ceil ((hi - phi)/Td)) + 1
        return phi + Td*np.arange(n_lo, n_hi+1)
    C_rise = grid(phi_r); C_fall = grid(phi_f)
    m_r = (C_rise >= (tmin - 0.25*Td)) & (C_rise <= (tmax + 0.25*Td))
    m_f = (C_fall >= (tmin - 0.25*Td)) & (C_fall <= (tmax + 0.25*Td))
    return C_rise[m_r], C_fall[m_f]

def _nearest_offsets(events: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ritorna (offsets, assigned_grid) con offsets = events - trigger_assegnato."""
    if events.size == 0 or grid.size == 0:
        return np.array([], float), np.array([], float)
    idx = np.searchsorted(grid, events)
    idx0 = np.clip(idx-1, 0, len(grid)-1)
    idx1 = np.clip(idx,   0, len(grid)-1)
    g0 = grid[idx0]; g1 = grid[idx1]
    pick = np.where(np.abs(events-g0) <= np.abs(events-g1), g0, g1)
    return (events - pick), pick

def _get_rise_fall_times(out: Dict, rise_key: str = "t90", fall_key: str = "t10") -> Tuple[np.ndarray, np.ndarray]:
    r_times = np.array([m[rise_key] for m in out["rise"]], float) if len(out["rise"]) else np.array([], float)
    f_times = np.array([m[fall_key] for m in out["fall"]], float) if len(out["fall"]) else np.array([], float)
    if r_times.size: r_times.sort()
    if f_times.size: f_times.sort()
    return r_times, f_times

def estimate_absolute_latency_with_t0_levels(out: Dict,
                                             t0: float,
                                             Td_hint: Optional[float] = None,
                                             rise_key: str = "t90",
                                             fall_key: str = "t10") -> Dict:
    """
    Usa t0 (primo trigger noto) e i crossing selezionati:
      - rise_key per i fronti di salita (default 't90')
      - fall_key per i fronti di discesa (default 't10')
    per stimare:
      Td, il tipo del primo comando (rise/fall), Lr, Lf, jitter RMS, e le griglie di trigger.
    Definizioni:
      Lr = mediana( t90_rise - trigger_rise_associato )
      Lf = mediana( t10_fall - trigger_fall_associato )
    """
    r_times, f_times = _get_rise_fall_times(out, rise_key=rise_key, fall_key=fall_key)
    if r_times.size + f_times.size < 2:
        return dict(ok=False, reason="Too few events for absolute latency", t0=t0)

    # 1) Td stimato dai dati; se c'è un hint lo combino robustamente
    Td_est = _robust_dwell_from_edges(r_times, f_times)
    if np.isfinite(Td_hint) and Td_hint > 0:
        Td = 0.5*(Td_est + Td_hint) if (np.isfinite(Td_est) and Td_est > 0) else float(Td_hint)
    else:
        Td = float(Td_est)
    if not (np.isfinite(Td) and Td > 0):
        return dict(ok=False, reason="Cannot estimate Td", t0=t0)

    tmin = np.nanmin(np.concatenate([r_times, f_times]) if f_times.size else r_times)
    tmax = np.nanmax(np.concatenate([r_times, f_times]) if f_times.size else r_times)

    # 2) prova due ipotesi su t0: rise o fall → scegli quella con residui minori
    best = None
    for hyp in ("rise", "fall"):
        C_rise, C_fall = _build_trigger_grids_from_t0(t0, Td, tmin, tmax, hyp)

        # OFFSETS E ASSEGNAZIONI (!!! spacchettati correttamente)
        off_r, assign_r = _nearest_offsets(r_times, C_rise)
        off_f, assign_f = _nearest_offsets(f_times, C_fall)

        # Lr/Lf come mediane degli offset
        Lr = float(np.nanmedian(off_r)) if off_r.size else np.nan
        Lf = float(np.nanmedian(off_f)) if off_f.size else np.nan

        res_r = off_r - Lr if np.isfinite(Lr) else np.array([], float)
        res_f = off_f - Lf if np.isfinite(Lf) else np.array([], float)
        rms = np.sqrt(np.nanmean(np.concatenate([res_r**2, res_f**2]))) if (res_r.size+res_f.size) else np.inf

        cand = dict(
            ok=True, t0=t0, first_kind=hyp, Td=Td,
            C_rise=C_rise, C_fall=C_fall,
            Lr=Lr, Lf=Lf, rms=rms,
            rise_key=rise_key, fall_key=fall_key,
            rise_times=r_times, fall_times=f_times,  # utili per il plot "barre"
            assigned_rise=assign_r, assigned_fall=assign_f
        )
        if (best is None) or (cand["rms"] < best["rms"]):
            best = cand

    return best


# ========= Plot a barre: trigger e livelli (t90 rise, t10 fall) =========
def plot_triggers_and_levels_bars(rta, out, absres, channel_title="PD1", show_markers=False):
    """
    Barre verticali su:
      - trigger rise (sottili)
      - trigger fall (sottili)
      - t90 (rise) -- tratteggiate
      - t10 (fall) -- tratteggiate
    Opzionale: marker sui punti evento.
    """
    t = rta.t
    v = rta.v1 if out["channel"]=="PD1" else rta.v2

    # eventi ESATTI usati per la latenza (come richiesto)
    r_times = absres["rise_times"]   # t90 dei rise
    f_times = absres["fall_times"]   # t10 dei fall

    C_r = absres["C_rise"]; C_f = absres["C_fall"]
    Lr = absres["Lr"]; Lf = absres["Lf"]

    plt.figure(figsize=(12,5))
    plt.plot(t, v, lw=0.8, alpha=0.65, label=f"{channel_title} (V)")

    # Trigger puri
    for c in C_r:
        if t[0] <= c <= t[-1]: plt.axvline(c, alpha=0.35, linewidth=1.0, label="trigger rise")
    for c in C_f:
        if t[0] <= c <= t[-1]: plt.axvline(c, alpha=0.35, linewidth=1.0, label="trigger fall")

    # t90 (rise) e t10 (fall) — tratteggiate
    for tr in r_times:
        if t[0] <= tr <= t[-1]: plt.axvline(tr, alpha=0.85, linestyle="--", linewidth=1.3, label="rise t90")
    for tf in f_times:
        if t[0] <= tf <= t[-1]: plt.axvline(tf, alpha=0.85, linestyle="--", linewidth=1.3, label="fall t10")

    if show_markers:
        if r_times.size: plt.scatter(r_times, np.interp(r_times, t, v), s=30, marker="^", label="marker t90 (rise)")
        if f_times.size: plt.scatter(f_times, np.interp(f_times, t, v), s=30, marker="v", label="marker t10 (fall)")

    # legenda unica (senza duplicati)
    h, l = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(l, h))
    plt.legend(uniq.values(), uniq.keys(), loc="best")

    Lr_ms = (Lr*1e3) if np.isfinite(Lr) else float('nan')
    Lf_ms = (Lf*1e3) if np.isfinite(Lf) else float('nan')
    plt.title(f"{channel_title}: Trigger | t90(rise) | t10(fall) — Lr≈{Lr_ms:.2f} ms, Lf≈{Lf_ms:.2f} ms")
    plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)")
    plt.tight_layout(); plt.show()


# =========================
# 3) Classe di analisi (come tua versione)
# =========================
class ResponseTimeAnalyzer:
    WINDOW_SIZE     = 8
    TARGET_FS_HZ    = 5000.0
    TRIM_HEAD_N     = 100
    TRIM_TAIL_N     = 100
    BOUND_GUARD_MS  = 20.0

    MIN_EDGE_AMP_V  = 0.003
    MIN_EDGE_SNR    = 3.0

    def __init__(self, csv_file_name: str):
        self._fname = csv_file_name
        t_raw, v1_raw, v2_raw = get_data_from_DAQamiCSVfile(csv_file_name)
        dt = np.diff(t_raw)
        self._fs_raw = 1.0 / np.median(dt) if np.all(dt > 0) else np.nan

        v1s = _moving_average(v1_raw, self.WINDOW_SIZE)
        v2s = _moving_average(v2_raw, self.WINDOW_SIZE)

        decim = max(1, int(round(self._fs_raw / self.TARGET_FS_HZ))) if np.isfinite(self._fs_raw) and self._fs_raw > 0 else 1
        t_d, v1_d, v2_d = t_raw[::decim], v1s[::decim], v2s[::decim]
        fs_d = self._fs_raw / decim if np.isfinite(self._fs_raw) else np.nan

        a, b = self.TRIM_HEAD_N, self.TRIM_TAIL_N
        if a + b < len(t_d):
            self._t  = t_d[a:len(t_d)-b]
            self._v1 = v1_d[a:len(v1_d)-b]
            self._v2 = v2_d[a:len(v2_d)-b]
            print(f"[INFO] Trim: removed first {a} and last {b} samples. New length: {len(self._t)}")
        else:
            self._t, self._v1, self._v2 = t_d, v1_d, v2_d
            print("[WARN] Not enough samples to trim; no samples removed.")
        self._fs = fs_d

        eps = 1e-12
        denom = self._v1 + self._v2
        denom = np.where(np.abs(denom) < eps, eps*np.sign(denom+eps), denom)
        self._s = (self._v1 - self._v2) / denom

        self._g_s  = np.diff(self._s,  prepend=self._s[0])
        self._g_v1 = np.diff(self._v1, prepend=self._v1[0])
        self._g_v2 = np.diff(self._v2, prepend=self._v2[0])

    def detect_transitions_s(self, grad_threshold_k: float = 6.0, min_separation_ms: float = 70.0) -> List[int]:
        fs = self._fs
        g = self._g_s
        med = np.median(g)
        mad = np.median(np.abs(g - med)) + 1e-12
        thr = grad_threshold_k * mad
        cand = np.where(np.abs(g) > thr)[0].tolist()
        if not cand:
            return []
        min_sep = int(round((min_separation_ms / 1000.0) * fs))
        grouped, cur = [], [cand[0]]
        for idx in cand[1:]:
            if idx - cur[-1] <= min_sep:
                cur.append(idx)
            else:
                peak = cur[int(np.argmax(np.abs(g[cur])))]
                grouped.append(peak)
                cur = [idx]
        if cur:
            peak = cur[int(np.argmax(np.abs(g[cur])))]
            grouped.append(peak)
        guard = int(round(self._fs * (self.BOUND_GUARD_MS / 1000.0)))
        grouped = [i for i in grouped if (i > guard and i < (len(self._s) - 1 - guard))]
        return sorted(grouped)

    def refine_index_abs(self, t_guess: float, channel: str, search_ms: float = 30.0) -> Optional[int]:
        half = int(round(min(search_ms/1000.0, 0.5) * self._fs))
        j0 = int(np.searchsorted(self._t, t_guess))
        lo = max(1, j0 - half); hi = min(len(self._t)-2, j0 + half)
        if hi <= lo:
            return None
        g = self._g_v1 if channel == "PD1" else self._g_v2
        j = lo + int(np.argmax(np.abs(g[lo:hi+1])))
        return j

    def measure_edge_on_channel(self, idx: int, channel: str, pre_ms: float = 5.0, post_ms: float = 25.0) -> Optional[Dict]:
        fs = self._fs
        n_pre  = max(3, int(round(fs * (pre_ms / 1000.0))))
        n_post = max(3, int(round(fs * (post_ms / 1000.0))))
        i0 = int(idx)
        y  = self._v1 if channel == "PD1" else self._v2

        guard = int(round(self._fs * (self.BOUND_GUARD_MS / 1000.0)))
        if i0 < max(n_pre, guard) or i0 + max(n_post, guard) >= len(y):
            return None

        i_start = i0 - n_pre
        i_end   = i0 + n_post

        pre  = np.median(y[i_start:i0])
        post = np.median(y[i0+1:i_end])
        delta = post - pre
        if np.isclose(delta, 0.0):
            return None

        noise = _noise_std_baseline(y, i0, self._fs, pre_ms=pre_ms)
        if (abs(delta) < self.MIN_EDGE_AMP_V) or (abs(delta) < self.MIN_EDGE_SNR * max(noise, 1e-12)):
            return None

        y10 = pre + 0.1 * delta
        y50 = pre + 0.5 * delta
        y90 = pre + 0.9 * delta

        t10 = _linear_interp_time(self._t, y, y10, i_start, i_end)
        t50 = _linear_interp_time(self._t, y, y50, i_start, i_end)
        t90 = _linear_interp_time(self._t, y, y90, i_start, i_end)
        if (t10 is None) or (t50 is None) or (t90 is None):
            return None

        if delta > 0:
            kind = "rising"
            if not (t10 < t50 < t90):
                return None
            width_ms = (t90 - t10) * 1000.0
        else:
            kind = "falling"
            if not (t10 < t50 < t90):
                return None
            width_ms = (t90 - t10) * 1000.0

        a10 = _estimate_slope(self._t, y, t10, window_ms=1.0)
        a90 = _estimate_slope(self._t, y, t90, window_ms=1.0)
        sigma_t10 = abs(noise / a10) if (a10 is not None and a10 != 0) else np.nan
        sigma_t90 = abs(noise / a90) if (a90 is not None and a90 != 0) else np.nan
        sigma_ms = 1000.0 * math.sqrt(sigma_t10**2 + sigma_t90**2) if (np.isfinite(sigma_t10) and np.isfinite(sigma_t90)) else np.nan

        tau_s, r2 = _fit_exp_linearized(self._t[i_start:i_end+1], y[i_start:i_end+1], kind, pre, post)
        tr_exp_ms = (math.log(9.0) * tau_s * 1000.0) if (tau_s is not None) else np.nan

        return dict(channel=channel, kind=kind, idx=i0, t_center=float(self._t[i0]),
                    t10=float(t10), t50=float(t50), t90=float(t90),
                    width_10_90_ms=float(width_ms),
                    sigma_width_ms=float(sigma_ms),
                    tau_ms=(float(tau_s*1000.0) if tau_s is not None else np.nan),
                    width_exp_ms=float(tr_exp_ms) if not np.isnan(tr_exp_ms) else np.nan,
                    exp_r2=(float(r2) if r2 is not None else np.nan))

    def analyze_one_channel_with_dwell(self, channel: str, dwell_time_s: float,
                                       grad_threshold_k=6.0, min_separation_ms=70.0,
                                       pre_ms=5.0, post_ms=25.0, search_ms=30.0,
                                       expected_rise: int = 9, expected_fall: int = 10) -> Dict:

        idxs_s = self.detect_transitions_s(grad_threshold_k=grad_threshold_k,
                                           min_separation_ms=min_separation_ms)

        direct = []
        for idx in idxs_s:
            j = self.refine_index_abs(self._t[idx], channel, search_ms=search_ms)
            if j is None:
                j = self.refine_index_abs(self._t[idx], channel, search_ms=40.0)
            if j is None:
                continue
            m = self.measure_edge_on_channel(j, channel, pre_ms=pre_ms, post_ms=post_ms)
            if m is not None:
                direct.append(m)

        rise = sorted([m for m in direct if m["kind"] == "rising"], key=lambda r: r["t_center"])
        fall = sorted([m for m in direct if m["kind"] == "falling"], key=lambda r: r["t_center"])

        def _add_missing(target_list, seeds_list, want_kind):
            need = (expected_fall if want_kind=="falling" else expected_rise) - len(target_list)
            if need <= 0 or len(seeds_list) == 0:
                return
            min_sep = int(round(self._fs * (min_separation_ms / 1000.0)))
            for mseed in seeds_list:
                t_pred = mseed["t_center"] + dwell_time_s
                j = self.refine_index_abs(t_pred, channel, search_ms=30.0)
                if j is None: j = self.refine_index_abs(t_pred, channel, search_ms=40.0)
                if j is None: continue
                if any(abs(j - mm["idx"]) < min_sep for mm in (rise+fall)):
                    continue
                m = self.measure_edge_on_channel(j, channel, pre_ms=pre_ms, post_ms=post_ms)
                if m is not None and m["kind"] == want_kind:
                    target_list.append(m)
                if len(target_list) >= (expected_fall if want_kind=="falling" else expected_rise):
                    break

        if len(fall) < expected_fall:
            _add_missing(fall, rise, "falling")
        if len(rise) < expected_rise:
            _add_missing(rise, fall, "rising")

        rise = sorted(rise, key=lambda r: r["t_center"])[:expected_rise]
        fall = sorted(fall, key=lambda r: r["t_center"])[:expected_fall]
        kept = sorted(rise + fall, key=lambda r: r["t_center"])

        def _stats(x):
            if len(x)==0: return dict(n=0, mean=np.nan, std=np.nan, median=np.nan)
            v = np.array(x, float)
            return dict(n=len(v), mean=float(np.mean(v)), std=float(np.std(v, ddof=1) if len(v)>1 else 0.0), median=float(np.median(v)))

        rise_w      = [m["width_10_90_ms"] for m in rise]
        fall_w      = [m["width_10_90_ms"] for m in fall]
        rise_w_exp  = [m["width_exp_ms"]    for m in rise if np.isfinite(m["width_exp_ms"])]
        fall_w_exp  = [m["width_exp_ms"]    for m in fall if np.isfinite(m["width_exp_ms"])]

        def _rel(frames, key):
            return np.array([m[key] - m["t_center"] for m in frames]) if len(frames) else np.array([])
        mk = dict(
            t10_rise_rel = float(np.median(_rel(rise, "t10"))) if len(rise) else np.nan,
            t90_rise_rel = float(np.median(_rel(rise, "t90"))) if len(rise) else np.nan,
            t10_fall_rel = float(np.median(_rel(fall, "t10"))) if len(fall) else np.nan,
            t90_fall_rel = float(np.median(_rel(fall, "t90"))) if len(fall) else np.nan,
        )

        return dict(
            channel = channel,
            idxs_s = idxs_s,
            rise = rise,
            fall = fall,
            kept = kept,
            markers = mk,
            summary = dict(
                fs_raw_Hz=float(self._fs_raw),
                fs_Hz=float(self._fs),
                n_detected_on_s=len(idxs_s),
                n_rise=len(rise),
                n_fall=len(fall),
                rise_10_90_stats=_stats(rise_w),
                fall_10_90_stats=_stats(fall_w),
                rise_exp_stats=_stats(rise_w_exp),
                fall_exp_stats=_stats(fall_w_exp),
            )
        )

    @staticmethod
    def _grid_phase_latency(t_events: np.ndarray, T: float, nsteps: int = 4001) -> Tuple[float, float, float]:
        if (t_events is None) or (len(t_events) < 2) or not np.isfinite(T) or T <= 0:
            return (np.nan, np.nan, np.nan)
        t = np.sort(np.array(t_events, float))
        deltas = np.linspace(0.0, T, nsteps, endpoint=False)
        best_idx, best_rms = None, np.inf
        best_resid = None
        for j, d in enumerate(deltas):
            x = t - d
            k = np.round(x / T)
            r = x - k*T
            rms = float(np.sqrt(np.mean(r*r)))
            if rms < best_rms:
                best_idx, best_rms, best_resid = j, rms, r
        delta = float(deltas[best_idx])
        residual_std = float(np.std(best_resid, ddof=1)) if len(t) > 1 else best_rms
        return (delta, best_rms, residual_std)

    @staticmethod
    def _linear_fit_latency(t_events: np.ndarray) -> Tuple[float, float, float, float]:
        if (t_events is None) or (len(t_events) < 2):
            return (np.nan, np.nan, np.nan, np.nan)
        t = np.sort(np.array(t_events, float))
        N = len(t)
        n = np.arange(N, dtype=float)
        n0 = n - np.mean(n)
        t0 = t - np.mean(t)
        denom = np.sum(n0*n0)
        if denom <= 0:
            return (np.nan, np.nan, np.nan, np.nan)
        T_eff = float(np.sum(n0*t0) / denom)
        delta = float(np.mean(t) - T_eff*np.mean(n))
        r = t - (delta + n*T_eff)
        jitter_rms   = float(np.sqrt(np.mean(r*r)))
        residual_std = float(np.std(r, ddof=1) if N>1 else 0.0)
        if np.isfinite(T_eff) and T_eff > 0:
            delta = float(delta % T_eff)
        return (delta, T_eff, jitter_rms, residual_std)

    @staticmethod
    def _bootstrap_delta_A(t_events: np.ndarray, T: float, B: int = 200) -> Tuple[float, float]:
        if (t_events is None) or (len(t_events) < 2) or not np.isfinite(T) or T <= 0:
            return (np.nan, np.nan)
        t = np.sort(np.array(t_events, float))
        N = len(t)
        ds = []
        for _ in range(B):
            idx = np.random.randint(0, N, size=N)
            tb = np.sort(t[idx])
            d, _, _ = ResponseTimeAnalyzer._grid_phase_latency(tb, T)
            ds.append(d)
        ds = np.array(ds)
        return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
        # -------- pairing robusto per simmetria: accoppia per tempo (±tol_ms)
    @staticmethod
    def _pair_by_time(A_times: List[float], B_times: List[float], tol_ms: float = 10.0) -> List[Tuple[int,int]]:
        A = np.array(A_times); B = np.array(B_times)
        pairs = []
        used_B = set()
        for i, ta in enumerate(A):
            if len(B)==0: break
            j = int(np.argmin(np.abs(B - ta)))
            if abs(B[j] - ta) <= tol_ms/1000.0 and j not in used_B:
                pairs.append((i, j))
                used_B.add(j)
        return pairs

    def symmetry_pairs(self, out_PD1: Dict, out_PD2: Dict, tol_ms: float = 10.0) -> Dict:
        # →PD1: PD1 rise vs PD2 fall
        t_PD1_r = [m["t_center"] for m in out_PD1["rise"]]
        t_PD2_f = [m["t_center"] for m in out_PD2["fall"]]
        p_r_f = self._pair_by_time(t_PD1_r, t_PD2_f, tol_ms)
        diffs_to_PD1 = []
        for i,j in p_r_f:
            diffs_to_PD1.append(out_PD1["rise"][i]["width_10_90_ms"] - out_PD2["fall"][j]["width_10_90_ms"])

        # →PD2: PD1 fall vs PD2 rise
        t_PD1_f = [m["t_center"] for m in out_PD1["fall"]]
        t_PD2_r = [m["t_center"] for m in out_PD2["rise"]]
        p_f_r = self._pair_by_time(t_PD1_f, t_PD2_r, tol_ms)
        diffs_to_PD2 = []
        for i,j in p_f_r:
            diffs_to_PD2.append(out_PD1["fall"][i]["width_10_90_ms"] - out_PD2["rise"][j]["width_10_90_ms"])

        def _stats(v):
            if len(v)==0: 
                return dict(n=0, mean=np.nan, std=np.nan, median=np.nan, abs_mean=np.nan)
            arr = np.array(v, float)
            return dict(
                n=len(arr),
                mean=float(np.mean(arr)),
                std=float(np.std(arr, ddof=1) if len(arr)>1 else 0.0),
                median=float(np.median(arr)),
                abs_mean=float(np.mean(np.abs(arr)))
            )

        return dict(
            pairs_summary = dict(
                to_PD1 = _stats(diffs_to_PD1),
                to_PD2 = _stats(diffs_to_PD2),
                n_pairs_to_PD1 = len(p_r_f),
                n_pairs_to_PD2 = len(p_f_r),
            ),
            pairs_idx = dict(to_PD1=p_r_f, to_PD2=p_f_r)
        )

    @property
    def t(self) -> np.ndarray:
        return self._t

    @property
    def v1(self) -> np.ndarray:
        return self._v1

    @property
    def v2(self) -> np.ndarray:
        return self._v2

    @property
    def s(self) -> np.ndarray:
        return self._s

    @property
    def fs(self) -> float:
        return self._fs


# =========================
# 4) Main
# =========================
def main_blink_data_dwt100ms_with_cute_plots():
    FDIR = "D:\\phd_slm_edo\\old_data\\slm_time_response\\photodiode\\"
    fname = FDIR + "20240906_1441_blink_loop10_dwell100ms\\Analog - 9-6-2024 2-41-15.63336 PM.csv"

    # dwell teorico (hint)
    dwell_time_in_s = 106.5e-3

    # t0 del primo comando (trigger noto)
    t0_first_cmd_s = 1.102010

    if not os.path.exists(fname):
        alt = os.path.join(os.getcwd(), "Analog - 9-6-2024 2-41-15.63336 PM.csv")
        if os.path.exists(alt):
            fname = alt

    rta = ResponseTimeAnalyzer(fname)

    # ====== Stile globale “da tesi” ======
    mpl.rcParams.update({
        "figure.dpi": 170,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "lines.linewidth": 1.6,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
    })

    # helper per stile assi
    def _beautify(ax, xlabel=None, ylabel=None, title=None, xmaj=7, ymaj=6):
        if title:  ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(xmaj))
        ax.yaxis.set_major_locator(MaxNLocator(ymaj))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        # minor grid con tratteggio più fine
        ax.grid(which="minor", linestyle=":", alpha=0.25)
        ax.tick_params(direction="out", length=4, width=0.9)
        # bordi puliti
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ====== Analisi canali ======
    out1 = rta.analyze_one_channel_with_dwell(
        "PD1", dwell_time_in_s,
        grad_threshold_k=6.0, min_separation_ms=70.0,
        pre_ms=5.0, post_ms=25.0, search_ms=30.0,
        expected_rise=9, expected_fall=10
    )
    out2 = rta.analyze_one_channel_with_dwell(
        "PD2", dwell_time_in_s,
        grad_threshold_k=6.0, min_separation_ms=70.0,
        pre_ms=5.0, post_ms=25.0, search_ms=30.0,
        expected_rise=9, expected_fall=10
    )

    # Pairing simmetria
    sym = rta.symmetry_pairs(out1, out2, tol_ms=10.0)

    # ----- STAMPA RIEPILOGO (immutata) -----
    def _fmt_block(name, summ):
        print(f"\n[{name}]  Fs(raw)={summ['fs_raw_Hz']:.2f} Hz | Fs(dec)={summ['fs_Hz']:.2f} Hz | toggles(s)={summ['n_detected_on_s']}")
        def _fmt_stats(title, st):
            if st["n"] == 0:
                print(f"  {title}: n=0")
            else:
                print(f"  {title}: n={st['n']}, mean={st['mean']:.3f} ms, median={st['median']:.3f} ms, std={st['std']:.3f} ms")
        _fmt_stats("Rise 10–90% (linear)", summ["rise_10_90_stats"])
        _fmt_stats("Fall 10–90% (linear)", summ["fall_10_90_stats"])
        _fmt_stats("Rise 10–90% (exp-fit ≈ τ ln 9)", summ["rise_exp_stats"])
        _fmt_stats("Fall 10–90% (exp-fit ≈ τ ln 9)", summ["fall_exp_stats"])

    print("\n=== SLM Response Time Analysis — PD1 & PD2 (same robust procedure on each channel) ===")
    print(f"File: {fname}")
    _fmt_block("PD1", out1["summary"])
    _fmt_block("PD2", out2["summary"])

    # ----- STAMPA PER-EDGE (immutata) -----
    def _print_edges(label, edges):
        if len(edges)==0:
            print(f"\n-- {label} --\n(nessun fronte)")
            return
        print(f"\n-- {label} --")
        for k, m in enumerate(edges, 1):
            unc = (m['sigma_width_ms'] if np.isfinite(m['sigma_width_ms']) else float('nan'))
            print(f"{label[0]}{k:02d} @ t={m['t_center']:.6f}s | 10–90={m['width_10_90_ms']:.3f} ms ±{unc:.3f} ms"
                  f" | exp: τ={m['tau_ms']:.3f} ms, 10–90≈{m['width_exp_ms']:.3f} ms | R²={m['exp_r2']:.3f}")

    _print_edges("PD1 Rising", out1["rise"])
    _print_edges("PD1 Falling", out1["fall"])
    _print_edges("PD2 Rising", out2["rise"])
    _print_edges("PD2 Falling", out2["fall"])

    # ----- SYMMETRY (immutata) -----
    ps = sym["pairs_summary"]
    print("\n--- Symmetry test (paired by time, ±10 ms) ---")
    print(f"→PD1 pairs (PD1 rise vs PD2 fall): n={ps['n_pairs_to_PD1']}, mean Δ={ps['to_PD1']['mean']:.3f} ms, ⟨|Δ|⟩={ps['to_PD1']['abs_mean']:.3f} ms, std={ps['to_PD1']['std']:.3f} ms")
    print(f"→PD2 pairs (PD1 fall vs PD2 rise): n={ps['n_pairs_to_PD2']}, mean Δ={ps['to_PD2']['mean']:.3f} ms, ⟨|Δ|⟩={ps['to_PD2']['abs_mean']:.3f} ms, std={ps['to_PD2']['std']:.3f} ms")

    # ---------- LATENZA (senza trigger) sui t50 di PD1 ----------
    t50_pd1_all  = [m["t50"] for m in out1["rise"]] + [m["t50"] for m in out1["fall"]]
    t50_pd1_rise = [m["t50"] for m in out1["rise"]]
    t50_pd1_fall = [m["t50"] for m in out1["fall"]]

    dA_all,  jA_all,  sA_all  = ResponseTimeAnalyzer._grid_phase_latency(t50_pd1_all,  dwell_time_in_s)
    dA_rise, jA_rise, sA_rise = ResponseTimeAnalyzer._grid_phase_latency(t50_pd1_rise, 2*dwell_time_in_s)
    dA_fall, jA_fall, sA_fall = ResponseTimeAnalyzer._grid_phase_latency(t50_pd1_fall, 2*dwell_time_in_s)

    dB_all,  T_eff_all,  jB_all,  sB_all  = ResponseTimeAnalyzer._linear_fit_latency(t50_pd1_all)
    dB_rise, T_eff_rise, jB_rise, sB_rise = ResponseTimeAnalyzer._linear_fit_latency(t50_pd1_rise)
    dB_fall, T_eff_fall, jB_fall, sB_fall = ResponseTimeAnalyzer._linear_fit_latency(t50_pd1_fall)

    lo_all, hi_all = 0, 0  # Bootstrap opzionale

    print("\n--- Command latency estimate on PD1 (no trigger) ---")
    print(f"[A-grid, T]      all:  N={len(t50_pd1_all):2d},  delta≈{dA_all*1000:.2f} ms,  jitter_RMS≈{jA_all*1000:.2f} ms, residual_std≈{sA_all*1000:.2f} ms")
    print(f"[A-grid, 2T]     rise: N={len(t50_pd1_rise):2d}, delta≈{dA_rise*1000:.2f} ms,  jitter_RMS≈{jA_rise*1000:.2f} ms, residual_std≈{sA_rise*1000:.2f} ms")
    print(f"[A-grid, 2T]     fall: N={len(t50_pd1_fall):2d}, delta≈{dA_fall*1000:.2f} ms,  jitter_RMS≈{jA_fall*1000:.2f} ms, residual_std≈{sA_fall*1000:.2f} ms")
    print(f"[B-linear]       all:  N={len(t50_pd1_all):2d},  delta≈{dB_all*1000:.2f} ms,  T_eff≈{T_eff_all*1000:.2f} ms  (~dwell {dwell_time_in_s*1000:.2f} ms), jitter_RMS≈{jB_all*1000:.2f} ms")
    print(f"[B-linear]       rise: N={len(t50_pd1_rise):2d}, delta≈{dB_rise*1000:.2f} ms,  T_eff≈{T_eff_rise*1000:.2f} ms  (~2*dwell {2*dwell_time_in_s*1000:.2f} ms), jitter_RMS≈{jB_rise*1000:.2f} ms")
    print(f"[B-linear]       fall: N={len(t50_pd1_fall):2d}, delta≈{dB_fall*1000:.2f} ms,  T_eff≈{T_eff_fall*1000:.2f} ms  (~2*dwell {2*dwell_time_in_s*1000:.2f} ms), jitter_RMS≈{jB_fall*1000:.2f} ms")
    print(f"Bootstrap 95% CI on δ (A, all): [{lo_all*1e3:.2f}, {hi_all*1e3:.2f}] ms")

    # ---------- NUOVO: latenza ASSOLUTA da t0 (rise=t90, fall=t10) ----------
    absres = estimate_absolute_latency_with_t0_levels(
        out1, t0_first_cmd_s, Td_hint=dwell_time_in_s, rise_key="t90", fall_key="t90"
    )
    if absres.get("ok", False):
        Td_ms = absres["Td"]*1e3
        Lr_ms = absres["Lr"]*1e3 if np.isfinite(absres["Lr"]) else float('nan')
        Lf_ms = absres["Lf"]*1e3 if np.isfinite(absres["Lf"]) else float('nan')

        print("\n=== ABSOLUTE LATENCY from first trigger t0 (levels: rise=t90, fall=t10) ===")
        print(f"First command kind: {absres['first_kind']}")
        print(f"Dwell Td ≈ {Td_ms:.3f} ms")
        print(f"Latency L_r (t90 - trigger_rise) ≈ {Lr_ms:.3f} ms")
        print(f"Latency L_f (t10 - trigger_fall) ≈ {Lf_ms:.3f} ms")
        print(f"Residual RMS (jitter) ≈ {absres['rms']*1e3:.3f} ms")

        plot_triggers_and_levels_bars(rta, out1, absres, channel_title="PD1", show_markers=True)
    else:
        print("\n[WARN] Absolute latency estimation (t90/t10) failed:", absres.get("reason", "unknown"))

    # ---------- PLOT 0: PD1 & PD2 insieme ----------
    fig0, ax0 = plt.subplots(figsize=(10.0, 4.6))
    ax0.plot(rta.t, rta.v1, label="PD1 (SMA)")
    ax0.plot(rta.t, rta.v2, label="PD2 (SMA)")
    _beautify(ax0, xlabel="Time [s]", ylabel="Signal [V]", title="Photodiodes Signals — Dwell time = 100 ms")
    ax0.legend(loc="best", frameon=True)
    ax0.set_xlim(1.690, 2.550)
    fig0.tight_layout()
    plt.show()

    # ---------- PLOT A: PD1 con marker ----------
    figA, axA = plt.subplots(figsize=(10.0, 4.6))
    axA.plot(rta.t, rta.v1, label="PD1 (V)")
    idx_r_PD1 = [m["idx"] for m in out1["rise"]]
    idx_f_PD1 = [m["idx"] for m in out1["fall"]]
    tr1 = [rta.t[i] for i in idx_r_PD1]; tf1 = [rta.t[i] for i in idx_f_PD1]
    yr1 = [rta.v1[i] for i in idx_r_PD1]; yf1 = [rta.v1[i] for i in idx_f_PD1]
    axA.scatter(tr1, yr1, marker="^", color="tab:green", s=28, label="Rising on PD1", zorder=3)
    axA.scatter(tf1, yf1, marker="v", color="tab:red",   s=28, label="Falling on PD1", zorder=3)
    _beautify(axA, xlabel="Time [s]", ylabel="Voltage [V]", title="PD1 voltage with detected edges")
    axA.legend(loc="best", frameon=True)
    figA.tight_layout()
    plt.show()

    # ---------- PLOT B: PD2 con marker ----------
    figB, axB = plt.subplots(figsize=(10.0, 4.6))
    axB.plot(rta.t, rta.v2, label="PD2 (V)")
    idx_r_PD2 = [m["idx"] for m in out2["rise"]]
    idx_f_PD2 = [m["idx"] for m in out2["fall"]]
    tr2 = [rta.t[i] for i in idx_r_PD2]; tf2 = [rta.t[i] for i in idx_f_PD2]
    yr2 = [rta.v2[i] for i in idx_r_PD2]; yf2 = [rta.v2[i] for i in idx_f_PD2]
    axB.scatter(tr2, yr2, marker="^", color="tab:green", s=28, label="Rising on PD2", zorder=3)
    axB.scatter(tf2, yf2, marker="v", color="tab:red",   s=28, label="Falling on PD2", zorder=3)
    _beautify(axB, xlabel="Time [s]", ylabel="Voltage [V]", title="PD2 voltage with detected edges")
    axB.legend(loc="best", frameon=True)
    figB.tight_layout()
    plt.show()

    # ---------- overlay locali (come tua “figura 3”) ----------
    def _overlay(ax, channel: str, meas: List[Dict], fs, t, v, title, labelx):
        win_ms = 6.5
        n_win = max(1, int(round(fs * (win_ms / 1000.0))))
        if len(meas) > 0:
            t10_rel = float(np.median([m["t10"] - m["t_center"] for m in meas]))
            t90_rel = float(np.median([m["t90"] - m["t_center"] for m in meas]))
        else:
            t10_rel = t90_rel = np.nan

        for m in meas:
            i0 = m["idx"]
            lo = max(0, i0 - n_win); hi = min(len(v) - 1, i0 + n_win)
            tt = t[lo:hi+1] - t[i0]; ss = v[lo:hi+1]
            ax.plot((tt - t10_rel) / 1e-3, ss, alpha=0.80)

        if np.isfinite(t10_rel):
            ax.axvline((t10_rel - t10_rel) / 1e-3, linestyle="--", label=r"$\bar{t}_{10}$")
        if np.isfinite(t90_rel):
            ax.axvline((t90_rel - t10_rel) / 1e-3, linestyle="--", label=r"$\bar{t}_{90}$")

        _beautify(ax, xlabel=f"Time rel. to {labelx} [ms]", ylabel=channel, title=title)

        # legenda compatta se ci sono le due linee medie
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", frameon=True)

    # fig 5: PD1 rising/falling overlay
    fig5, axes5 = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True, sharey=True)
    _overlay(axes5[0], "Signal on PD1 [V]", out1["rise"], rta.fs, rta.t, rta.v1, "Rising Events", r"$\bar{t}_{10}$")
    _overlay(axes5[1], "",                   out1["fall"], rta.fs, rta.t, rta.v1, "Falling Events", r"$\bar{t}_{90}$")
    fig5.suptitle("(A) Transient signals on PD1", y=0.98)
    fig5.tight_layout()
    plt.show()

    # fig 6: PD2 rising/falling overlay
    fig6, axes6 = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True, sharey=True)
    _overlay(axes6[0], "Signal on PD2 [V]", out2["rise"], rta.fs, rta.t, rta.v2, "Rising Events", r"$\bar{t}_{10}$")
    _overlay(axes6[1], "",                   out2["fall"], rta.fs, rta.t, rta.v2, "Falling Events", r"$\bar{t}_{90}$")
    fig6.suptitle("(B) Transient signals on PD2", y=0.98)
    fig6.tight_layout()
    plt.show()
    
def main_blink_data_dwt100ms():
    FDIR = "D:\\phd_slm_edo\\old_data\\slm_time_response\\photodiode\\"
    fname = FDIR + "20240906_1441_blink_loop10_dwell100ms\\Analog - 9-6-2024 2-41-15.63336 PM.csv"

    # dwell teorico (hint)
    dwell_time_in_s = 106.5e-3

    # t0 del primo comando (trigger noto)
    t0_first_cmd_s = 1.102010

    if not os.path.exists(fname):
        alt = os.path.join(os.getcwd(), "Analog - 9-6-2024 2-41-15.63336 PM.csv")
        if os.path.exists(alt):
            fname = alt

    rta = ResponseTimeAnalyzer(fname)

    # PD1 e PD2, stessa pipeline
    out1 = rta.analyze_one_channel_with_dwell("PD1", dwell_time_in_s,
            grad_threshold_k=6.0, min_separation_ms=70.0,
            pre_ms=5.0, post_ms=25.0, search_ms=30.0,
            expected_rise=9, expected_fall=10)
    out2 = rta.analyze_one_channel_with_dwell("PD2", dwell_time_in_s,
            grad_threshold_k=6.0, min_separation_ms=70.0,
            pre_ms=5.0, post_ms=25.0, search_ms=30.0,
            expected_rise=9, expected_fall=10)

    # Pairing simmetria
    sym = rta.symmetry_pairs(out1, out2, tol_ms=10.0)

    # ----- STAMPA RIEPILOGO -----
    def _fmt_block(name, summ):
        print(f"\n[{name}]  Fs(raw)={summ['fs_raw_Hz']:.2f} Hz | Fs(dec)={summ['fs_Hz']:.2f} Hz | toggles(s)={summ['n_detected_on_s']}")
        def _fmt_stats(title, st):
            if st["n"] == 0:
                print(f"  {title}: n=0")
            else:
                print(f"  {title}: n={st['n']}, mean={st['mean']:.3f} ms, median={st['median']:.3f} ms, std={st['std']:.3f} ms")
        _fmt_stats("Rise 10–90% (linear)", summ["rise_10_90_stats"])
        _fmt_stats("Fall 10–90% (linear)", summ["fall_10_90_stats"])
        _fmt_stats("Rise 10–90% (exp-fit ≈ τ ln 9)", summ["rise_exp_stats"])
        _fmt_stats("Fall 10–90% (exp-fit ≈ τ ln 9)", summ["fall_exp_stats"])

    print("\n=== SLM Response Time Analysis — PD1 & PD2 (same robust procedure on each channel) ===")
    print(f"File: {fname}")
    _fmt_block("PD1", out1["summary"])
    _fmt_block("PD2", out2["summary"])

    # ----- STAMPA PER-EDGE -----
    def _print_edges(label, edges):
        if len(edges)==0:
            print(f"\n-- {label} --\n(nessun fronte)")
            return
        print(f"\n-- {label} --")
        for k, m in enumerate(edges, 1):
            unc = (m['sigma_width_ms'] if np.isfinite(m['sigma_width_ms']) else float('nan'))
            print(f"{label[0]}{k:02d} @ t={m['t_center']:.6f}s | 10–90={m['width_10_90_ms']:.3f} ms ±{unc:.3f} ms"
                  f" | exp: τ={m['tau_ms']:.3f} ms, 10–90≈{m['width_exp_ms']:.3f} ms | R²={m['exp_r2']:.3f}")

    _print_edges("PD1 Rising", out1["rise"])
    _print_edges("PD1 Falling", out1["fall"])
    _print_edges("PD2 Rising", out2["rise"])
    _print_edges("PD2 Falling", out2["fall"])

    # ----- SYMMETRY -----
    ps = sym["pairs_summary"]
    print("\n--- Symmetry test (paired by time, ±10 ms) ---")
    print(f"→PD1 pairs (PD1 rise vs PD2 fall): n={ps['n_pairs_to_PD1']}, mean Δ={ps['to_PD1']['mean']:.3f} ms, ⟨|Δ|⟩={ps['to_PD1']['abs_mean']:.3f} ms, std={ps['to_PD1']['std']:.3f} ms")
    print(f"→PD2 pairs (PD1 fall vs PD2 rise): n={ps['n_pairs_to_PD2']}, mean Δ={ps['to_PD2']['mean']:.3f} ms, ⟨|Δ|⟩={ps['to_PD2']['abs_mean']:.3f} ms, std={ps['to_PD2']['std']:.3f} ms")

    # ---------- LATENZA (senza trigger) sui t50 di PD1 ----------
    t50_pd1_all  = [m["t50"] for m in out1["rise"]] + [m["t50"] for m in out1["fall"]]
    t50_pd1_rise = [m["t50"] for m in out1["rise"]]
    t50_pd1_fall = [m["t50"] for m in out1["fall"]]

    dA_all,  jA_all,  sA_all  = ResponseTimeAnalyzer._grid_phase_latency(t50_pd1_all,  dwell_time_in_s)
    dA_rise, jA_rise, sA_rise = ResponseTimeAnalyzer._grid_phase_latency(t50_pd1_rise, 2*dwell_time_in_s)
    dA_fall, jA_fall, sA_fall = ResponseTimeAnalyzer._grid_phase_latency(t50_pd1_fall, 2*dwell_time_in_s)

    dB_all,  T_eff_all,  jB_all,  sB_all  = ResponseTimeAnalyzer._linear_fit_latency(t50_pd1_all)
    dB_rise, T_eff_rise, jB_rise, sB_rise = ResponseTimeAnalyzer._linear_fit_latency(t50_pd1_rise)
    dB_fall, T_eff_fall, jB_fall, sB_fall = ResponseTimeAnalyzer._linear_fit_latency(t50_pd1_fall)

    lo_all, hi_all =0,0 #ResponseTimeAnalyzer._bootstrap_delta_A(np.array(t50_pd1_all), dwell_time_in_s, B=400)

    print("\n--- Command latency estimate on PD1 (no trigger) ---")
    print(f"[A-grid, T]      all:  N={len(t50_pd1_all):2d},  delta≈{dA_all*1000:.2f} ms,  jitter_RMS≈{jA_all*1000:.2f} ms, residual_std≈{sA_all*1000:.2f} ms")
    print(f"[A-grid, 2T]     rise: N={len(t50_pd1_rise):2d}, delta≈{dA_rise*1000:.2f} ms,  jitter_RMS≈{jA_rise*1000:.2f} ms, residual_std≈{sA_rise*1000:.2f} ms")
    print(f"[A-grid, 2T]     fall: N={len(t50_pd1_fall):2d}, delta≈{dA_fall*1000:.2f} ms,  jitter_RMS≈{jA_fall*1000:.2f} ms, residual_std≈{sA_fall*1000:.2f} ms")
    print(f"[B-linear]       all:  N={len(t50_pd1_all):2d},  delta≈{dB_all*1000:.2f} ms,  T_eff≈{T_eff_all*1000:.2f} ms  (~dwell {dwell_time_in_s*1000:.2f} ms), jitter_RMS≈{jB_all*1000:.2f} ms")
    print(f"[B-linear]       rise: N={len(t50_pd1_rise):2d}, delta≈{dB_rise*1000:.2f} ms,  T_eff≈{T_eff_rise*1000:.2f} ms  (~2*dwell {2*dwell_time_in_s*1000:.2f} ms), jitter_RMS≈{jB_rise*1000:.2f} ms")
    print(f"[B-linear]       fall: N={len(t50_pd1_fall):2d}, delta≈{dB_fall*1000:.2f} ms,  T_eff≈{T_eff_fall*1000:.2f} ms  (~2*dwell {2*dwell_time_in_s*1000:.2f} ms), jitter_RMS≈{jB_fall*1000:.2f} ms")
    print(f"Bootstrap 95% CI on δ (A, all): [{lo_all*1e3:.2f}, {hi_all*1e3:.2f}] ms")

    # ---------- NUOVO: latenza ASSOLUTA da t0 (rise=t90, fall=t10) ----------
    absres = estimate_absolute_latency_with_t0_levels(out1, t0_first_cmd_s,
                                                      Td_hint=dwell_time_in_s,
                                                      rise_key="t90", fall_key="t90")
    if absres.get("ok", False):
        Td_ms = absres["Td"]*1e3
        Lr_ms = absres["Lr"]*1e3 if np.isfinite(absres["Lr"]) else float('nan')
        Lf_ms = absres["Lf"]*1e3 if np.isfinite(absres["Lf"]) else float('nan')

        print("\n=== ABSOLUTE LATENCY from first trigger t0 (levels: rise=t90, fall=t10) ===")
        print(f"First command kind: {absres['first_kind']}")
        print(f"Dwell Td ≈ {Td_ms:.3f} ms")
        print(f"Latency L_r (t90 - trigger_rise) ≈ {Lr_ms:.3f} ms")
        print(f"Latency L_f (t10 - trigger_fall) ≈ {Lf_ms:.3f} ms")
        print(f"Residual RMS (jitter) ≈ {absres['rms']*1e3:.3f} ms")

        # Plot pulito: solo barre su trigger e su t90/t10 (marker opzionali)
        plot_triggers_and_levels_bars(rta, out1, absres, channel_title="PD1", show_markers=True)
    else:
        print("\n[WARN] Absolute latency estimation (t90/t10) failed:", absres.get("reason", "unknown"))

    # ---------- PLOT 0: PD1 & PD2 insieme ----------
    #figure 2
    plt.figure()
    plt.plot(rta.t, rta.v1, label="PD1 (SMA)")
    plt.plot(rta.t, rta.v2, label="PD2 (SMA)")
    plt.xlabel("Time [s]"); plt.ylabel("Signal [V]"); plt.title("Photodiodes Signals: Dwell time = 100 ms")
    plt.legend(); plt.show();plt.xlim(1.690,2.550)

    # ---------- PLOT A: PD1 con marker ----------
    plt.figure()
    plt.plot(rta.t, rta.v1, label="PD1 (V)")
    idx_r_PD1 = [m["idx"] for m in out1["rise"]]
    idx_f_PD1 = [m["idx"] for m in out1["fall"]]
    tr1 = [rta.t[i] for i in idx_r_PD1]; tf1 = [rta.t[i] for i in idx_f_PD1]
    yr1 = [rta.v1[i] for i in idx_r_PD1]; yf1 = [rta.v1[i] for i in idx_f_PD1]
    plt.scatter(tr1, yr1, marker="^", color="g", label="Rising on PD1")
    plt.scatter(tf1, yf1, marker="v", color="r", label="Falling on PD1")
    plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)")
    plt.title("PD1 voltage with detected edges")
    plt.legend(); plt.show()

    # ---------- PLOT B: PD2 con marker ----------
    plt.figure()
    plt.plot(rta.t, rta.v2, label="PD2 (V)")
    idx_r_PD2 = [m["idx"] for m in out2["rise"]]
    idx_f_PD2 = [m["idx"] for m in out2["fall"]]
    tr2 = [rta.t[i] for i in idx_r_PD2]; tf2 = [rta.t[i] for i in idx_f_PD2]
    yr2 = [rta.v2[i] for i in idx_r_PD2]; yf2 = [rta.v2[i] for i in idx_f_PD2]
    plt.scatter(tr2, yr2, marker="^", color="g", label="Rising on PD2")
    plt.scatter(tf2, yf2, marker="v", color="r", label="Falling on PD2")
    plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)")
    plt.title("PD2 voltage with detected edges")
    plt.legend(); plt.show()

    # ---------- overlay locali (come tua “figura 3”) ----------
    def _overlay(ax, channel: str, meas: List[Dict], fs, t, v, title, labelx):
        win_ms = 6.5
        n_win = max(1, int(round(fs * (win_ms / 1000.0))))
        if len(meas)>0:
            t10_rel = float(np.median([m["t10"] - m["t_center"] for m in meas]))
            t90_rel = float(np.median([m["t90"] - m["t_center"] for m in meas]))
        else:
            t10_rel = t90_rel = np.nan
        for m in meas:
            i0 = m["idx"]
            lo = max(0, i0 - n_win); hi = min(len(v) - 1, i0 + n_win)
            tt = t[lo:hi+1] - t[i0]; ss = v[lo:hi+1]
            ax.plot((tt-t10_rel)/1e-3, ss, alpha=0.75)
        if np.isfinite(t10_rel): ax.axvline((t10_rel-t10_rel)/1e-3, linestyle="--", label=r"$\bar{t}_{10}$")
        if np.isfinite(t90_rel): ax.axvline((t90_rel-t10_rel)/1e-3, linestyle="--", label=r"$\bar{t}_{90}$")
        ax.set_title(title); ax.set_xlabel("Time rel. to "+f"{labelx}"+" [ms]"); ax.set_ylabel(f"{channel}")
        if ax.get_legend_handles_labels()[0]: ax.legend()

    #fig 5
    fig, axes = plt.subplots(1, 2, figsize=(11, 4),sharex=True, sharey=True)
    _overlay(axes[0], "Signal on PD1 [V]", out1["rise"], rta.fs, rta.t, rta.v1, "Rising Events", r"$\bar{t}_{10}$")
    _overlay(axes[1], "", out1["fall"], rta.fs, rta.t, rta.v1, "Falling Events", r"$\bar{t}_{90}$")
    fig.suptitle("(A) Transient signals on PD1", y=0.98)
    plt.tight_layout(); plt.show()
    #fig 6
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True,sharey=True)
    _overlay(axes[0], "Signal on PD2 [V]", out2["rise"], rta.fs, rta.t, rta.v2, "Rising Events", r"$\bar{t}_{10}$")
    _overlay(axes[1], "", out2["fall"], rta.fs, rta.t, rta.v2, "Falling Events", r"$\bar{t}_{90}$")
    fig.suptitle("(B) Transient signals on PD2", y=0.98)
    plt.tight_layout(); plt.show()

    # ---------- CSV ----------
    # def _dump_csv(out, fname_csv):
    #     rows = []
    #     for m in out["kept"]:
    #         rows.append(dict(
    #             channel = out["channel"],
    #             kind = m["kind"], t_center_s=m["t_center"],
    #             t10_s=m["t10"], t50_s=m["t50"], t90_s=m["t90"],
    #             width_10_90_ms=m["width_10_90_ms"],
    #             sigma_width_ms=m["sigma_width_ms"],
    #             tau_ms=m["tau_ms"], width_exp_ms=m["width_exp_ms"],
    #             exp_r2=m["exp_r2"]
    #         ))
    #     pd.DataFrame(rows).to_csv(fname_csv, index=False)
    #     print(f"Saved per-edge report: {fname_csv}")
    #
    # _dump_csv(out1, os.path.join(os.getcwd(), "slm_edge_timings_PD1.csv"))
    # _dump_csv(out2, os.path.join(os.getcwd(), "slm_edge_timings_PD2.csv"))


# if __name__ == "__main__":
#     main_blink_data_dwt100ms()

def main_blink_data_dwt0ms():
    FDIR = "D:\\phd_slm_edo\\old_data\\slm_time_response\\photodiode\\"
    fname = FDIR + "20240906_1422_blink_loop10_dwell0ms\\Analog - 9-6-2024 2-22-21.39271 PM.csv"

    # dwell teorico (hint)
    dwell_time_in_s = 2e-3

    # t0 del primo comando (trigger noto)
    t0_first_cmd_s = 1.239

    if not os.path.exists(fname):
        alt = os.path.join(os.getcwd(), "Analog - 9-6-2024 2-41-15.63336 PM.csv")
        if os.path.exists(alt):
            fname = alt

    rta = ResponseTimeAnalyzer(fname)

    # PD1 e PD2, stessa pipeline
    out1 = rta.analyze_one_channel_with_dwell("PD1", dwell_time_in_s,
            grad_threshold_k=6.0, min_separation_ms=1,
            pre_ms=0.6, post_ms=0.6, search_ms=0.5,
            expected_rise=9, expected_fall=10)
    out2 = rta.analyze_one_channel_with_dwell("PD2", dwell_time_in_s,
            grad_threshold_k=6.0, min_separation_ms=1,
            pre_ms=0.6, post_ms=0.6, search_ms=0.5,
            expected_rise=9, expected_fall=10)

    # Pairing simmetria
    sym = rta.symmetry_pairs(out1, out2, tol_ms=10.0)

    # ----- STAMPA RIEPILOGO -----
    def _fmt_block(name, summ):
        print(f"\n[{name}]  Fs(raw)={summ['fs_raw_Hz']:.2f} Hz | Fs(dec)={summ['fs_Hz']:.2f} Hz | toggles(s)={summ['n_detected_on_s']}")
        def _fmt_stats(title, st):
            if st["n"] == 0:
                print(f"  {title}: n=0")
            else:
                print(f"  {title}: n={st['n']}, mean={st['mean']:.3f} ms, median={st['median']:.3f} ms, std={st['std']:.3f} ms")
        _fmt_stats("Rise 10–90% (linear)", summ["rise_10_90_stats"])
        _fmt_stats("Fall 10–90% (linear)", summ["fall_10_90_stats"])
        _fmt_stats("Rise 10–90% (exp-fit ≈ τ ln 9)", summ["rise_exp_stats"])
        _fmt_stats("Fall 10–90% (exp-fit ≈ τ ln 9)", summ["fall_exp_stats"])

    print("\n=== SLM Response Time Analysis — PD1 & PD2 (same robust procedure on each channel) ===")
    print(f"File: {fname}")
    _fmt_block("PD1", out1["summary"])
    _fmt_block("PD2", out2["summary"])

    # ----- STAMPA PER-EDGE -----
    def _print_edges(label, edges):
        if len(edges)==0:
            print(f"\n-- {label} --\n(nessun fronte)")
            return
        print(f"\n-- {label} --")
        for k, m in enumerate(edges, 1):
            unc = (m['sigma_width_ms'] if np.isfinite(m['sigma_width_ms']) else float('nan'))
            print(f"{label[0]}{k:02d} @ t={m['t_center']:.6f}s | 10–90={m['width_10_90_ms']:.3f} ms ±{unc:.3f} ms"
                  f" | exp: τ={m['tau_ms']:.3f} ms, 10–90≈{m['width_exp_ms']:.3f} ms | R²={m['exp_r2']:.3f}")

    _print_edges("PD1 Rising", out1["rise"])
    _print_edges("PD1 Falling", out1["fall"])
    _print_edges("PD2 Rising", out2["rise"])
    _print_edges("PD2 Falling", out2["fall"])

    # ----- SYMMETRY -----
    ps = sym["pairs_summary"]
    print("\n--- Symmetry test (paired by time, ±10 ms) ---")
    print(f"→PD1 pairs (PD1 rise vs PD2 fall): n={ps['n_pairs_to_PD1']}, mean Δ={ps['to_PD1']['mean']:.3f} ms, ⟨|Δ|⟩={ps['to_PD1']['abs_mean']:.3f} ms, std={ps['to_PD1']['std']:.3f} ms")
    print(f"→PD2 pairs (PD1 fall vs PD2 rise): n={ps['n_pairs_to_PD2']}, mean Δ={ps['to_PD2']['mean']:.3f} ms, ⟨|Δ|⟩={ps['to_PD2']['abs_mean']:.3f} ms, std={ps['to_PD2']['std']:.3f} ms")

    # ---------- LATENZA (senza trigger) sui t50 di PD1 ----------
    t50_pd1_all  = [m["t50"] for m in out1["rise"]] + [m["t50"] for m in out1["fall"]]
    t50_pd1_rise = [m["t50"] for m in out1["rise"]]
    t50_pd1_fall = [m["t50"] for m in out1["fall"]]

    dA_all,  jA_all,  sA_all  = ResponseTimeAnalyzer._grid_phase_latency(t50_pd1_all,  dwell_time_in_s)
    dA_rise, jA_rise, sA_rise = ResponseTimeAnalyzer._grid_phase_latency(t50_pd1_rise, 2*dwell_time_in_s)
    dA_fall, jA_fall, sA_fall = ResponseTimeAnalyzer._grid_phase_latency(t50_pd1_fall, 2*dwell_time_in_s)

    dB_all,  T_eff_all,  jB_all,  sB_all  = ResponseTimeAnalyzer._linear_fit_latency(t50_pd1_all)
    dB_rise, T_eff_rise, jB_rise, sB_rise = ResponseTimeAnalyzer._linear_fit_latency(t50_pd1_rise)
    dB_fall, T_eff_fall, jB_fall, sB_fall = ResponseTimeAnalyzer._linear_fit_latency(t50_pd1_fall)

    lo_all, hi_all =0,0 #ResponseTimeAnalyzer._bootstrap_delta_A(np.array(t50_pd1_all), dwell_time_in_s, B=400)

    print("\n--- Command latency estimate on PD1 (no trigger) ---")
    print(f"[A-grid, T]      all:  N={len(t50_pd1_all):2d},  delta≈{dA_all*1000:.2f} ms,  jitter_RMS≈{jA_all*1000:.2f} ms, residual_std≈{sA_all*1000:.2f} ms")
    print(f"[A-grid, 2T]     rise: N={len(t50_pd1_rise):2d}, delta≈{dA_rise*1000:.2f} ms,  jitter_RMS≈{jA_rise*1000:.2f} ms, residual_std≈{sA_rise*1000:.2f} ms")
    print(f"[A-grid, 2T]     fall: N={len(t50_pd1_fall):2d}, delta≈{dA_fall*1000:.2f} ms,  jitter_RMS≈{jA_fall*1000:.2f} ms, residual_std≈{sA_fall*1000:.2f} ms")
    print(f"[B-linear]       all:  N={len(t50_pd1_all):2d},  delta≈{dB_all*1000:.2f} ms,  T_eff≈{T_eff_all*1000:.2f} ms  (~dwell {dwell_time_in_s*1000:.2f} ms), jitter_RMS≈{jB_all*1000:.2f} ms")
    print(f"[B-linear]       rise: N={len(t50_pd1_rise):2d}, delta≈{dB_rise*1000:.2f} ms,  T_eff≈{T_eff_rise*1000:.2f} ms  (~2*dwell {2*dwell_time_in_s*1000:.2f} ms), jitter_RMS≈{jB_rise*1000:.2f} ms")
    print(f"[B-linear]       fall: N={len(t50_pd1_fall):2d}, delta≈{dB_fall*1000:.2f} ms,  T_eff≈{T_eff_fall*1000:.2f} ms  (~2*dwell {2*dwell_time_in_s*1000:.2f} ms), jitter_RMS≈{jB_fall*1000:.2f} ms")
    print(f"Bootstrap 95% CI on δ (A, all): [{lo_all*1e3:.2f}, {hi_all*1e3:.2f}] ms")

    # ---------- NUOVO: latenza ASSOLUTA da t0 (rise=t90, fall=t10) ----------
    absres = estimate_absolute_latency_with_t0_levels(out1, t0_first_cmd_s,
                                                      Td_hint=dwell_time_in_s,
                                                      rise_key="t90", fall_key="t90")
    if absres.get("ok", False):
        Td_ms = absres["Td"]*1e3
        Lr_ms = absres["Lr"]*1e3 if np.isfinite(absres["Lr"]) else float('nan')
        Lf_ms = absres["Lf"]*1e3 if np.isfinite(absres["Lf"]) else float('nan')

        print("\n=== ABSOLUTE LATENCY from first trigger t0 (levels: rise=t90, fall=t10) ===")
        print(f"First command kind: {absres['first_kind']}")
        print(f"Dwell Td ≈ {Td_ms:.3f} ms")
        print(f"Latency L_r (t90 - trigger_rise) ≈ {Lr_ms:.3f} ms")
        print(f"Latency L_f (t10 - trigger_fall) ≈ {Lf_ms:.3f} ms")
        print(f"Residual RMS (jitter) ≈ {absres['rms']*1e3:.3f} ms")

        # Plot pulito: solo barre su trigger e su t90/t10 (marker opzionali)
        plot_triggers_and_levels_bars(rta, out1, absres, channel_title="PD1", show_markers=True)
    else:
        print("\n[WARN] Absolute latency estimation (t90/t10) failed:", absres.get("reason", "unknown"))

    # ---------- PLOT 0: PD1 & PD2 insieme ----------
    plt.figure()
    plt.plot(rta.t, rta.v1, label="PD1 (V)")
    plt.plot(rta.t, rta.v2, label="PD2 (V)")
    plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)"); plt.title("Photodiode signals (PD1 & PD2)")
    plt.legend(); plt.show()

    # ---------- PLOT A: PD1 con marker ----------
    plt.figure()
    plt.plot(rta.t, rta.v1, label="PD1 (V)")
    idx_r_PD1 = [m["idx"] for m in out1["rise"]]
    idx_f_PD1 = [m["idx"] for m in out1["fall"]]
    tr1 = [rta.t[i] for i in idx_r_PD1]; tf1 = [rta.t[i] for i in idx_f_PD1]
    yr1 = [rta.v1[i] for i in idx_r_PD1]; yf1 = [rta.v1[i] for i in idx_f_PD1]
    plt.scatter(tr1, yr1, marker="^", color="g", label="Rising on PD1")
    plt.scatter(tf1, yf1, marker="v", color="r", label="Falling on PD1")
    plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)")
    plt.title("PD1 voltage with detected edges")
    plt.legend(); plt.show()

    # ---------- PLOT B: PD2 con marker ----------
    plt.figure()
    plt.plot(rta.t, rta.v2, label="PD2 (V)")
    idx_r_PD2 = [m["idx"] for m in out2["rise"]]
    idx_f_PD2 = [m["idx"] for m in out2["fall"]]
    tr2 = [rta.t[i] for i in idx_r_PD2]; tf2 = [rta.t[i] for i in idx_f_PD2]
    yr2 = [rta.v2[i] for i in idx_r_PD2]; yf2 = [rta.v2[i] for i in idx_f_PD2]
    plt.scatter(tr2, yr2, marker="^", color="g", label="Rising on PD2")
    plt.scatter(tf2, yf2, marker="v", color="r", label="Falling on PD2")
    plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)")
    plt.title("PD2 voltage with detected edges")
    plt.legend(); plt.show()

    # ---------- overlay locali (come tua “figura 3”) ----------
    def _overlay(ax, channel: str, meas: List[Dict], fs, t, v, title):
        win_ms = 2.5#20.0
        n_win = max(1, int(round(fs * (win_ms / 1000.0))))
        if len(meas)>0:
            t10_rel = float(np.median([m["t10"] - m["t_center"] for m in meas]))
            t90_rel = float(np.median([m["t90"] - m["t_center"] for m in meas]))
        else:
            t10_rel = t90_rel = np.nan
        for m in meas:
            i0 = m["idx"]
            lo = max(0, i0 - n_win); hi = min(len(v) - 1, i0 + n_win)
            tt = t[lo:hi+1] - t[i0]; ss = v[lo:hi+1]
            ax.plot(tt, ss, alpha=0.75)
        if np.isfinite(t10_rel): ax.axvline(t10_rel, linestyle="--", label="t10 (median)")
        if np.isfinite(t90_rel): ax.axvline(t90_rel, linestyle="--", label="t90 (median)")
        ax.set_title(title); ax.set_xlabel("Time rel. to edge (s)"); ax.set_ylabel(f"{channel} (V)")
        if ax.get_legend_handles_labels()[0]: ax.legend()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    _overlay(axes[0], "PD1", out1["rise"], rta.fs, rta.t, rta.v1, "PD1 rising")
    _overlay(axes[1], "PD1", out1["fall"], rta.fs, rta.t, rta.v1, "PD1 falling")
    fig.suptitle("Edges overlay (PD1) with median t10/t90", y=1.02)
    plt.tight_layout(); plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    _overlay(axes[0], "PD2", out2["rise"], rta.fs, rta.t, rta.v2, "PD2 rising")
    _overlay(axes[1], "PD2", out2["fall"], rta.fs, rta.t, rta.v2, "PD2 falling")
    fig.suptitle("Edges overlay (PD2) with median t10/t90", y=1.02)
    plt.tight_layout(); plt.show()

    # ---------- CSV ----------
    def _dump_csv(out, fname_csv):
        rows = []
        for m in out["kept"]:
            rows.append(dict(
                channel = out["channel"],
                kind = m["kind"], t_center_s=m["t_center"],
                t10_s=m["t10"], t50_s=m["t50"], t90_s=m["t90"],
                width_10_90_ms=m["width_10_90_ms"],
                sigma_width_ms=m["sigma_width_ms"],
                tau_ms=m["tau_ms"], width_exp_ms=m["width_exp_ms"],
                exp_r2=m["exp_r2"]
            ))
        pd.DataFrame(rows).to_csv(fname_csv, index=False)
        print(f"Saved per-edge report: {fname_csv}")

    _dump_csv(out1, os.path.join(os.getcwd(), "slm_edge_timings_PD1.csv"))
    _dump_csv(out2, os.path.join(os.getcwd(), "slm_edge_timings_PD2.csv"))
