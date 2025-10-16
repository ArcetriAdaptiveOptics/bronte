
import os
import csv
import math
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


# =========================
# 3) Classe di analisi
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

    # ---------- Detection su s(t) ----------
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

        # livelli per 10%, 50%, 90%
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
            width_ms = (t90 - t10) * 1000.0  # definizione uniforme: 10→90 sempre positivo

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

    def analyze_one_channel_with_dwell(
        self,
        channel: str,
        dwell_time_s: float,
        grad_threshold_k=6.0,
        min_separation_ms=70.0,
        pre_ms=5.0,
        post_ms=25.0,
        search_ms=30.0,
        expected_rise: int = 9,
        expected_fall: int = 10,
    ) -> Dict:

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
            if len(v)==0: return dict(n=0, mean=np.nan, std=np.nan, median=np.nan, abs_mean=np.nan)
            arr = np.array(v, float)
            return dict(n=len(arr), mean=float(np.mean(arr)), std=float(np.std(arr, ddof=1) if len(arr)>1 else 0.0),
                        median=float(np.median(arr)), abs_mean=float(np.mean(np.abs(arr))))
        return dict(
            pairs_summary = dict(
                to_PD1 = _stats(diffs_to_PD1),
                to_PD2 = _stats(diffs_to_PD2),
                n_pairs_to_PD1 = len(p_r_f),
                n_pairs_to_PD2 = len(p_f_r),
            ),
            pairs_idx = dict(to_PD1=p_r_f, to_PD2=p_f_r)
        )

    # ===== Latenza: metodo A (grid) =====
    @staticmethod
    def _grid_phase_latency(t_events: np.ndarray, T: float, nsteps: int = 4001) -> Tuple[float, float, float]:
        """
        Minimizziamo l'RMS dei residui rispetto alla griglia {delta + k*T}.
        Ritorna (delta_s in [0,T), jitter_rms_s, residual_std_s).
        """
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

    # ===== Latenza: metodo B (fit lineare) =====
    @staticmethod
    def _linear_fit_latency(t_events: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Fit lineare: t_n = delta + n*T_eff, con t ordinati e n=0..N-1.
        Ritorna (delta_s, T_eff_s, jitter_rms_s, residual_std_s).
        """
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

    # ===== (Opzionale) Bootstrap su latenza =====
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

        # ------- getters (mancavano!) -------
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
def main_blink_data_dwt100ms():
    FDIR = "D:\\phd_slm_edo\\old_data\\slm_time_response\\photodiode\\"
    fname = FDIR + "20240906_1441_blink_loop10_dwell100ms\\Analog - 9-6-2024 2-41-15.63336 PM.csv"

    # *** imposta qui il dwell teorico (p.es. 0.1065 se usi 106.5 ms) ***
    dwell_time_in_s = 106.5e-3

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

    # Pairing simmetria per tempo (±10 ms)
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

    # ----- STAMPA PER-EDGE (PD1 & PD2) -----
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

    # ----- SYMMETRY PAIRS -----
    ps = sym["pairs_summary"]
    print("\n--- Symmetry test (paired by time, ±10 ms) ---")
    print(f"→PD1 pairs (PD1 rise vs PD2 fall): n={ps['n_pairs_to_PD1']}, mean Δ={ps['to_PD1']['mean']:.3f} ms, ⟨|Δ|⟩={ps['to_PD1']['abs_mean']:.3f} ms, std={ps['to_PD1']['std']:.3f} ms")
    print(f"→PD2 pairs (PD1 fall vs PD2 rise): n={ps['n_pairs_to_PD2']}, mean Δ={ps['to_PD2']['mean']:.3f} ms, ⟨|Δ|⟩={ps['to_PD2']['abs_mean']:.3f} ms, std={ps['to_PD2']['std']:.3f} ms")

    # ---------- LATENZA (senza trigger) sui t50 di PD1 ----------
    t50_pd1_all  = [m["t50"] for m in out1["rise"]] + [m["t50"] for m in out1["fall"]]
    t50_pd1_rise = [m["t50"] for m in out1["rise"]]
    t50_pd1_fall = [m["t50"] for m in out1["fall"]]

    # Metodo A: griglia con T (dwell teorico scelto) e con 2T
    dA_all,  jA_all,  sA_all  = ResponseTimeAnalyzer._grid_phase_latency(t50_pd1_all,  dwell_time_in_s)
    dA_rise, jA_rise, sA_rise = ResponseTimeAnalyzer._grid_phase_latency(t50_pd1_rise, 2*dwell_time_in_s)
    dA_fall, jA_fall, sA_fall = ResponseTimeAnalyzer._grid_phase_latency(t50_pd1_fall, 2*dwell_time_in_s)

    # Metodo B: fit lineare (stima T_eff e delta)
    dB_all,  T_eff_all,  jB_all,  sB_all  = ResponseTimeAnalyzer._linear_fit_latency(t50_pd1_all)
    dB_rise, T_eff_rise, jB_rise, sB_rise = ResponseTimeAnalyzer._linear_fit_latency(t50_pd1_rise)
    dB_fall, T_eff_fall, jB_fall, sB_fall = ResponseTimeAnalyzer._linear_fit_latency(t50_pd1_fall)

    # (Opzionale) bootstrap 95% CI su delta (Metodo A)
    # lo/hi in secondi
    lo_all, hi_all = ResponseTimeAnalyzer._bootstrap_delta_A(np.array(t50_pd1_all), dwell_time_in_s, B=400)

    print("\n--- Command latency estimate on PD1 (no trigger) ---")
    print(f"[A-grid, T]      all:  N={len(t50_pd1_all):2d},  delta≈{dA_all*1000:.2f} ms,  jitter_RMS≈{jA_all*1000:.2f} ms, residual_std≈{sA_all*1000:.2f} ms")
    print(f"[A-grid, 2T]     rise: N={len(t50_pd1_rise):2d}, delta≈{dA_rise*1000:.2f} ms,  jitter_RMS≈{jA_rise*1000:.2f} ms, residual_std≈{sA_rise*1000:.2f} ms")
    print(f"[A-grid, 2T]     fall: N={len(t50_pd1_fall):2d}, delta≈{dA_fall*1000:.2f} ms,  jitter_RMS≈{jA_fall*1000:.2f} ms, residual_std≈{sA_fall*1000:.2f} ms")
    print(f"[B-linear]       all:  N={len(t50_pd1_all):2d},  delta≈{dB_all*1000:.2f} ms,  T_eff≈{T_eff_all*1000:.2f} ms  (~dwell {dwell_time_in_s*1000:.2f} ms), jitter_RMS≈{jB_all*1000:.2f} ms")
    print(f"[B-linear]       rise: N={len(t50_pd1_rise):2d}, delta≈{dB_rise*1000:.2f} ms,  T_eff≈{T_eff_rise*1000:.2f} ms  (~2*dwell {2*dwell_time_in_s*1000:.2f} ms), jitter_RMS≈{jB_rise*1000:.2f} ms")
    print(f"[B-linear]       fall: N={len(t50_pd1_fall):2d}, delta≈{dB_fall*1000:.2f} ms,  T_eff≈{T_eff_fall*1000:.2f} ms  (~2*dwell {2*dwell_time_in_s*1000:.2f} ms), jitter_RMS≈{jB_fall*1000:.2f} ms")
    print (lo_all)
    print(hi_all)
    # Stima "errore su delta": std residual / sqrt(N) (errore tipo standard)
    if len(t50_pd1_all) >= 2:
        se_delta_ms = (sA_all*1000.0)/math.sqrt(len(t50_pd1_all))
        print(f"→ Uncertainty on δ (A, all) ~ residual_std/√N ≈ {se_delta_ms:.3f} ms  (jitter per-event ≈ {sA_all*1000:.3f} ms)")

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

    # ---------- PLOT C: overlay stile “figura 3” per PD1 e PD2 ----------
    def _overlay(ax, channel: str, meas: List[Dict], fs, t, v, title):
        win_ms = 20.0
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




