# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
# from bronte.mains.main241014_slm_time_resp_blink import (
#     ResponseTimeAnalyzer,
#     _get_rise_fall_times,
# )
#
# # ========= utilità robuste =========
# def _se_median_no_boot(x: np.ndarray) -> float:
#     x = np.asarray(x, float)
#     x = x[np.isfinite(x)]
#     n = x.size
#     if n < 2: return float('nan')
#     med = np.median(x)
#     mad = np.median(np.abs(x - med))
#     return 1.4826 * mad / np.sqrt(n)
#
# def _periods_from_times(ts: np.ndarray) -> np.ndarray:
#     ts = np.sort(np.array(ts, float))
#     return np.diff(ts) if ts.size >= 2 else np.array([], float)
#
# def _sdesc_ms(x):
#     if x.size == 0: return "n=0"
#     return f"n={x.size}, mean={np.mean(x)*1e3:.3f} ms, median={np.median(x)*1e3:.3f} ms, std={np.std(x, ddof=1)*1e3:.3f} ms"
#
# # ========= assegnazione al comando PRECEDENTE =========
# def _previous_offsets(events: np.ndarray, grid: np.ndarray):
#     """
#     Per ogni evento, prende il comando PRECEDENTE della stessa griglia (stesso tipo).
#     offsets = evento - comando_precedente. Scarta eventi prima del primo comando.
#     """
#     events = np.asarray(events, float)
#     grid   = np.asarray(grid, float)
#     if events.size == 0 or grid.size == 0:
#         return np.array([], float), np.array([], float)
#     idx  = np.searchsorted(grid, events, side="right") - 1
#     mask = (idx >= 0)
#     ev   = events[mask]
#     tr   = grid[idx[mask]]
#     return (ev - tr, tr)
#
# # ========= griglia uniforme (stampella) =========
# def _build_uniform_grids_from_t0(t0: float, Td: float, tmin: float, tmax: float, first_kind: str):
#     """
#     Griglia uniforme alternata a partire da t0.
#     first_kind in {'rise','fall'}.
#     """
#     assert first_kind in ("rise","fall")
#     lo = tmin - 2*Td; hi = tmax + 2*Td
#     C_rise, C_fall = [], []
#
#     # avanti
#     t = t0; kind = first_kind
#     while t <= hi:
#         (C_rise if kind=="rise" else C_fall).append(t)
#         t += Td
#         kind = "fall" if kind=="rise" else "rise"
#
#     # indietro
#     t = t0; kind = first_kind
#     while t >= lo:
#         (C_rise if kind=="rise" else C_fall).append(t)
#         t -= Td
#         kind = "fall" if kind=="rise" else "rise"
#
#     C_rise = np.array([c for c in C_rise if lo <= c <= hi], float)
#     C_fall = np.array([c for c in C_fall if lo <= c <= hi], float)
#     C_rise = np.unique(np.round(C_rise, 12)); C_rise.sort()
#     C_fall = np.unique(np.round(C_fall, 12)); C_fall.sort()
#     return C_rise, C_fall
#
# # ========= recovery (se il detector ha perso transiente) =========
# def _scan_first_edge_in_window(rta: ResponseTimeAnalyzer, channel: str,
#                                kind_wanted: str, t_start: float, t_end: float,
#                                step_samples: int = 5, pre_ms: float = 5.0, post_ms: float = 25.0):
#     assert kind_wanted in ("rising","falling")
#     i0 = int(np.searchsorted(rta.t, max(t_start, rta.t[0])))
#     i1 = int(np.searchsorted(rta.t, min(t_end,   rta.t[-1])))
#     i0 = max(1,i0); i1 = min(len(rta.t)-2, i1)
#     if i1 <= i0: return None
#     best = None
#     for j in range(i0, i1, max(1, step_samples)):
#         m = rta.measure_edge_on_channel(j, channel, pre_ms=pre_ms, post_ms=post_ms)
#         if m is not None and m["kind"] == kind_wanted:
#             if (best is None) or (m["t_center"] < best["t_center"]):
#                 best = m
#     return best
#
# def _recover_missing_events_with_windows(
#     rta: ResponseTimeAnalyzer,
#     out_channel: dict,
#     channel_name: str,
#     t0: float,
#     Td_hint: float,
#     first_kind: str,
#     want_rise: int,
#     want_fall: int,
#     search_after_cmd_ms: float = 80.0,
#     safety_lead_ms: float = 0.5
# ):
#     """
#     Usa griglia UNIFORME come guida. Per ogni comando atteso:
#       rise: cerca il primo 'rising' in [cmd+lead, cmd+search]
#       fall: cerca il primo 'falling' in [cmd+lead, cmd+search]
#     Aggiunge se manca.
#     """
#     pre_ms = 5.0; post_ms = 25.0
#     r_times = np.array([m["t90"] for m in out_channel["rise"]], float) if len(out_channel["rise"]) else np.array([], float)
#     f_times = np.array([m["t10"] for m in out_channel["fall"]], float) if len(out_channel["fall"]) else np.array([], float)
#
#     tmin = float(rta.t[0]); tmax = float(rta.t[-1])
#     C_rise_uni, C_fall_uni = _build_uniform_grids_from_t0(t0, Td_hint, tmin, tmax, first_kind)
#
#     def _already_has_event(existing_times, t_new, tol_ms=8.0):
#         return existing_times.size>0 and (np.min(np.abs(existing_times - t_new)) <= tol_ms/1e3)
#
#     # RISE
#     for c in C_rise_uni:
#         if len(out_channel["rise"]) >= want_rise: break
#         w_lo = c + safety_lead_ms/1e3
#         w_hi = c + search_after_cmd_ms/1e3
#         m = _scan_first_edge_in_window(rta, channel_name, "rising", w_lo, w_hi,
#                                        step_samples=3, pre_ms=pre_ms, post_ms=post_ms)
#         if (m is not None) and (not _already_has_event(r_times, m["t90"])):  # t90 per rise
#             out_channel["rise"].append(m); r_times = np.append(r_times, m["t90"])
#
#     # FALL
#     for c in C_fall_uni:
#         if len(out_channel["fall"]) >= want_fall: break
#         w_lo = c + safety_lead_ms/1e3
#         w_hi = c + search_after_cmd_ms/1e3
#         m = _scan_first_edge_in_window(rta, channel_name, "falling", w_lo, w_hi,
#                                        step_samples=3, pre_ms=pre_ms, post_ms=post_ms)
#         if (m is not None) and (not _already_has_event(f_times, m["t10"])):  # t10 per fall
#             out_channel["fall"].append(m); f_times = np.append(f_times, m["t10"])
#
#     out_channel["rise"].sort(key=lambda m: m["t90"])
#     out_channel["fall"].sort(key=lambda m: m["t10"])
#
# # ========= latenze con definizione RICHIESTA =========
# def compute_latencies_previous(
#     rise_events_t90: np.ndarray,
#     fall_events_t10: np.ndarray,
#     C_rise_cmd: np.ndarray,   # comandi low→high (che portano a rise)
#     C_fall_cmd: np.ndarray,   # comandi high→low (che portano a fall)
# ):
#     """
#     L_r = t90_rise − cmd_low→high PRECEDENTE
#     L_f = t10_fall − cmd_high→low PRECEDENTE
#
#     Ritorna sia stime a MEDIANA (robuste) sia a MEDIA (classiche),
#     con errori standard, più statistiche per-evento.
#     Tutte le quantità sono in secondi (la stampa le converte in ms).
#     """
#     # offsets per-evento rispetto al comando PRECEDENTE
#     off_r, _ = _previous_offsets(np.asarray(rise_events_t90, float),
#                                  np.asarray(C_rise_cmd, float))
#     off_f, _ = _previous_offsets(np.asarray(fall_events_t10, float),
#                                  np.asarray(C_fall_cmd, float))
#
#     # ---------- stime a MEDIANA ----------
#     Lr_med = float(np.nanmedian(off_r)) if off_r.size else np.nan
#     Lf_med = float(np.nanmedian(off_f)) if off_f.size else np.nan
#     se_Lr_med = _se_median_no_boot(off_r)
#     se_Lf_med = _se_median_no_boot(off_f)
#
#     # ---------- stime a MEDIA ----------
#     def _safe_mean(x): return float(np.mean(x)) if x.size else np.nan
#     def _safe_std(x):  return float(np.std(x, ddof=1)) if x.size > 1 else (0.0 if x.size == 1 else np.nan)
#     def _safe_min(x):  return float(np.min(x)) if x.size else np.nan
#     def _safe_max(x):  return float(np.max(x)) if x.size else np.nan
#
#     Lr_mean = _safe_mean(off_r)
#     Lf_mean = _safe_mean(off_f)
#     std_Lr  = _safe_std(off_r)
#     std_Lf  = _safe_std(off_f)
#     se_Lr_mean = (std_Lr / np.sqrt(off_r.size)) if off_r.size > 1 else (0.0 if off_r.size == 1 else np.nan)
#     se_Lf_mean = (std_Lf / np.sqrt(off_f.size)) if off_f.size > 1 else (0.0 if off_f.size == 1 else np.nan)
#
#     # ---------- range & diagnostica ----------
#     min_Lr = _safe_min(off_r); max_Lr = _safe_max(off_r)
#     min_Lf = _safe_min(off_f); max_Lf = _safe_max(off_f)
#     rng_Lr = (max_Lr - min_Lr) if np.isfinite(min_Lr) and np.isfinite(max_Lr) else np.nan
#     rng_Lf = (max_Lf - min_Lf) if np.isfinite(min_Lf) and np.isfinite(max_Lf) else np.nan
#
#     # residui rispetto alle mediane (diagnostica combinata)
#     res = np.concatenate([(off_r - Lr_med) if np.isfinite(Lr_med) else np.array([]),
#                           (off_f - Lf_med) if np.isfinite(Lf_med) else np.array([])])
#     mad = 1.4826 * np.median(np.abs(res - np.median(res))) if res.size else np.nan
#     rms = float(np.sqrt(np.mean(res**2))) if res.size else np.nan
#
#     return dict(
#         # offset per-evento (se vuoi ispezionarli)
#         off_r=off_r, off_f=off_f,
#         n_r=int(off_r.size), n_f=int(off_f.size),
#
#         # stime a MEDIANA (robuste)
#         Lr=Lr_med, Lf=Lf_med,           # alias: Lr/Lf = mediana (per compatibilità retro)
#         Lr_med=Lr_med, Lf_med=Lf_med,
#         se_Lr=se_Lr_med, se_Lf=se_Lf_med,
#         se_Lr_med=se_Lr_med, se_Lf_med=se_Lf_med,
#
#         # stime a MEDIA (classiche)
#         Lr_mean=Lr_mean, Lf_mean=Lf_mean,
#         se_Lr_mean=se_Lr_mean, se_Lf_mean=se_Lf_mean,
#
#         # dispersioni/estremi per-evento
#         std_Lr=std_Lr, std_Lf=std_Lf,
#         min_Lr=min_Lr, max_Lr=max_Lr, rng_Lr=rng_Lr,
#         min_Lf=min_Lf, max_Lf=max_Lf, rng_Lf=rng_Lf,
#
#         # diagnostiche globali sui residui vs mediana
#         residual_mad=mad, residual_rms=rms,
#     )
#
#
# def print_latency_report(lat: dict, prefix: str = ""):
#     """Stampa compatta (in ms) di mediana vs media, errori standard, std e range."""
#     p = (prefix + " ").strip()
#     ms = lambda x: (x * 1e3) if np.isfinite(x) else np.nan
#
#     # --- MEDIANA (robusta) ---
#     print(f"{p}MEDIAN  L_r ≈ {ms(lat['Lr_med']):.3f} ms  ± {ms(lat['se_Lr_med']):.3f} ms [SE(median)]")
#     print(f"{p}MEDIAN  L_f ≈ {ms(lat['Lf_med']):.3f} ms  ± {ms(lat['se_Lf_med']):.3f} ms [SE(median)]")
#
#     # --- MEDIA (classica) ---
#     print(f"{p}MEAN    L_r ≈ {ms(lat['Lr_mean']):.3f} ms  ± {ms(lat['se_Lr_mean']):.3f} ms [SE(mean)]")
#     print(f"{p}MEAN    L_f ≈ {ms(lat['Lf_mean']):.3f} ms  ± {ms(lat['se_Lf_mean']):.3f} ms [SE(mean)]")
#
#     # --- dispersioni e range per-evento ---
#     if 'n_r' in lat and 'n_f' in lat:
#         print(f"{p}Per-event L_r: N={lat['n_r']}, std≈{ms(lat['std_Lr']):.3f} ms, "
#               f"min={ms(lat['min_Lr']):.3f} ms, max={ms(lat['max_Lr']):.3f} ms, "
#               f"range={ms(lat['rng_Lr']):.3f} ms")
#         print(f"{p}Per-event L_f: N={lat['n_f']}, std≈{ms(lat['std_Lf']):.3f} ms, "
#               f"min={ms(lat['min_Lf']):.3f} ms, max={ms(lat['max_Lf']):.3f} ms, "
#               f"range={ms(lat['rng_Lf']):.3f} ms")
#
#     # --- diagnostiche globali sui residui vs mediana ---
#     if 'residual_mad' in lat and 'residual_rms' in lat:
#         print(f"{p}Residuals (rise+fall vs median): MAD≈{ms(lat['residual_mad']):.3f} ms | RMS≈{ms(lat['residual_rms']):.3f} ms")
#
#
# # ========= PLOT con marker come da tua specifica (FIX) =========
# def plot_pd_with_triggers_and_events(rta, out_channel, C_rise, C_fall, channel_title, t0=None):
#     """
#     - Linee sottili: trigger rise/fall
#     - Linee tratteggiate: tempi usati per le latenze (rise=t90, fall=t10)
#     - Marker (come richiesto): 'v' = t10 dei FALL, '^' = t90 dei RISE
#     """
#     if channel_title.upper() == "PD1":
#         t, v = rta.t, rta.v1
#     else:
#         t, v = rta.t, rta.v2
#
#     # tempi per LATENZE
#     #rise_t90 = np.array([m["t90"] for m in out_channel["rise"]], float) if len(out_channel["rise"]) else np.array([], float)
#     #fall_t10 = np.array([m["t10"] for m in out_channel["fall"]], float) if len(out_channel["fall"]) else np.array([], float)
#
#     rise_t90 = np.array([m["t90"] for m in out_channel["rise"]], float) if len(out_channel["rise"]) else np.array([], float)
#     fall_t10 = np.array([m["t90"] for m in out_channel["fall"]], float) if len(out_channel["fall"]) else np.array([], float)
#
#     plt.figure(figsize=(12,5))
#     plt.plot(t, v, lw=0.9, alpha=0.75, label=f"{channel_title} (V)")
#
#     # if (t0 is not None) and (t[0] <= t0 <= t[-1]):
#     #     plt.axvline(t0, linewidth=2.0, alpha=0.7, label="t0 (command)")
#
#     # trigger
#     for c in C_rise:
#         if t[0] <= c <= t[-1]: plt.axvline(c, alpha=0.35, linewidth=1.0, label="trigger rise")
#     for c in C_fall:
#         if t[0] <= c <= t[-1]: plt.axvline(c, alpha=0.35, linewidth=1.0, label="trigger fall")
#
#     # linee tratteggiate = metrica latenze
#     for tr in rise_t90:
#         if t[0] <= tr <= t[-1]: plt.axvline(tr, linestyle="--", alpha=0.9, linewidth=1.3, label="rise t90 (lat)")
#     for tf in fall_t10:
#         if t[0] <= tf <= t[-1]: plt.axvline(tf, linestyle="--", alpha=0.9, linewidth=1.3, label="fall t10 (lat)")
#
#     # marker: 'v' = t10 FALL, '^' = t90 RISE
#     if fall_t10.size:
#         plt.scatter(fall_t10, np.interp(fall_t10, t, v), s=36, marker="v", label="marker: fall t10")
#     if rise_t90.size:
#         plt.scatter(rise_t90, np.interp(rise_t90, t, v), s=36, marker="^", label="marker: rise t90")
#
#     # legenda unica
#     h, l = plt.gca().get_legend_handles_labels()
#     uniq = dict(zip(l, h))
#     plt.legend(uniq.values(), uniq.keys(), loc="best")
#     plt.title(f"{channel_title}: TRIGGER + latenze (---) | marker: 'v'=fall t10, '^'=rise t90")
#     plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)")
#     plt.tight_layout(); plt.show()
#
# # ========= (NUOVO) — SIMULAZIONE TRIGGER dai tuoi eventi =========
# def simulate_triggers_from_events(out_channel: dict,
#                                   advance_ms: float = 6.5,   # usa 5.0 se vuoi il set "classico"
#                                   jitter_ms: float  = 0.3,   # usa 3.0 per jitter più grande
#                                   seed: int | None  = 42,
#                                   rise_key: str = "t90",
#                                   fall_key: str = "t10"):
#     """
#     Crea tempi di trigger sintetici per ogni evento:
#       - per i RISE: trigger = t90 − advance_ms + N(0, jitter_ms)
#       - per i FALL: trigger = t10 − advance_ms + N(0, jitter_ms)
#     Restituisce anche gli offset (evento − trigger), attesi ≈ +advance_ms.
#     """
#     rng = np.random.default_rng(seed)
#     r_times = np.array([m[rise_key] for m in out_channel["rise"]], float) if len(out_channel["rise"]) else np.array([], float)
#     f_times = np.array([m[fall_key] for m in out_channel["fall"]], float) if len(out_channel["fall"]) else np.array([], float)
#
#     adv_s    = advance_ms / 1e3
#     jitter_s = jitter_ms  / 1e3
#
#     C_rise_sim = r_times - adv_s + rng.normal(0.0, jitter_s, size=r_times.size) if r_times.size else np.array([], float)
#     C_fall_sim = f_times - adv_s + rng.normal(0.0, jitter_s, size=f_times.size) if f_times.size else np.array([], float)
#
#     off_r = r_times - C_rise_sim
#     off_f = f_times - C_fall_sim
#
#     return dict(
#         rise_times=r_times, fall_times=f_times,
#         C_rise_sim=C_rise_sim, C_fall_sim=C_fall_sim,
#         off_r=off_r, off_f=off_f,
#         advance_ms=advance_ms, jitter_ms=jitter_ms
#     )
#
# def plot_simulated_triggers_PD1(rta, sim: dict, t0_first_cmd_s: float | None = None, title_suffix: str = ""):
#     """Mostra segnale PD1, eventi (t90/t10) e trigger simulati con anticipo e jitter."""
#     t = rta.t; v = rta.v1
#     plt.figure(figsize=(12,5))
#     plt.plot(t, v, lw=0.9, alpha=0.75, label="PD1 (V)")
#
#     if t0_first_cmd_s is not None and (t[0] <= t0_first_cmd_s <= t[-1]):
#         plt.axvline(t0_first_cmd_s, linewidth=2.0, alpha=0.6, label="t0 (command)")
#
#     # eventi (tratteggiati)
#     for tr in sim["rise_times"]:
#         if t[0] <= tr <= t[-1]:
#             plt.axvline(tr, linestyle="--", alpha=0.85, linewidth=1.3, label="rise t90")
#     for tf in sim["fall_times"]:
#         if t[0] <= tf <= t[-1]:
#             plt.axvline(tf, linestyle="--", alpha=0.85, linewidth=1.3, label="fall t10")
#
#     # trigger simulati (sottili)
#     for c in sim["C_rise_sim"]:
#         if t[0] <= c <= t[-1]:
#             plt.axvline(c, alpha=0.35, linewidth=1.0, label="trigger (sim) rise")
#     for c in sim["C_fall_sim"]:
#         if t[0] <= c <= t[-1]:
#             plt.axvline(c, alpha=0.35, linewidth=1.0, label="trigger (sim) fall")
#
#     # marker: '^' = rise t90, 'v' = fall t10
#     if sim["rise_times"].size:
#         plt.scatter(sim["rise_times"], np.interp(sim["rise_times"], t, v), s=28, marker="^", label="marker t90 (rise)")
#     if sim["fall_times"].size:
#         plt.scatter(sim["fall_times"], np.interp(sim["fall_times"], t, v), s=28, marker="v", label="marker t10 (fall)")
#
#     h, l = plt.gca().get_legend_handles_labels()
#     uniq = dict(zip(l, h))
#     plt.legend(uniq.values(), uniq.keys(), loc="best")
#
#     plt.title(f"PD1: eventi & trigger simulati (−{sim['advance_ms']:.1f} ms, σ≈{sim['jitter_ms']:.1f} ms){title_suffix}")
#     plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)")
#     plt.tight_layout(); plt.show()
#
# def print_sim_stats(sim: dict, label: str = "PD1"):
#     def _sdesc(x):
#         x_ms = np.asarray(x)*1e3
#         if x_ms.size == 0:
#             return "n=0"
#         mu = float(np.mean(x_ms)); sd = float(np.std(x_ms, ddof=1) if x_ms.size > 1 else 0.0)
#         med = float(np.median(x_ms))
#         return f"n={x_ms.size}, mean={mu:.3f} ms, median={med:.3f} ms, std={sd:.3f} ms"
#     print(f"\n--- Simulated trigger offsets (evento − trigger) [{label}] ---")
#     print(f"Rise: {_sdesc(sim['off_r'])}  (target ≈ +{sim['advance_ms']:.1f} ms, std ≈ {sim['jitter_ms']:.1f} ms)")
#     print(f"Fall: {_sdesc(sim['off_f'])}  (target ≈ +{sim['advance_ms']:.1f} ms, std ≈ {sim['jitter_ms']:.1f} ms)")
#
# # ========= MAIN =========
# def main_plico_data():
#     """
#     - Detection su PD1/PD2
#     - Recovery per centrare i conteggi (PD1: 9 rise, 10 fall)
#     - Griglia UNIFORME da t0 (Td ≈ Tsum/2) per assegnazione PRECEDENTE
#     - L_r, L_f coerenti con le definizioni
#     - Plot con marker FIX: 'v' = t10 fall, '^' = t90 rise
#     - (NUOVO) Simulazione trigger dai tuoi eventi con anticipo + jitter
#     """
#     # === file ===
#     FDIR = "D:\\phd_slm_edo\\old_data\\slm_time_response\\photodiode\\"
#     fname = FDIR + "20240906_1232_1\\Analog - 9-6-2024 12-32-43.52892 PM.csv"
#
#     dwell_time_in_s_hint = 117e-3
#     t0_first_cmd_s = 3.0404      # primo comando = FALL
#     FIRST_KIND = "fall"          # sequenza alternata da t0
#
#     if not os.path.exists(fname):
#         alt = os.path.join(os.getcwd(), "Analog - 9-6-2024 12-32-43.52892 PM.csv")
#         if os.path.exists(alt): fname = alt
#
#     rta = ResponseTimeAnalyzer(fname)
#
#     out1 = rta.analyze_one_channel_with_dwell(
#         "PD1", dwell_time_in_s_hint,
#         grad_threshold_k=6.0, min_separation_ms=70.0,
#         pre_ms=5.0, post_ms=25.0, search_ms=40.0,
#         expected_rise=9, expected_fall=10
#     )
#     out2 = rta.analyze_one_channel_with_dwell(
#         "PD2", dwell_time_in_s_hint,
#         grad_threshold_k=6.0, min_separation_ms=70.0,
#         pre_ms=5.0, post_ms=25.0, search_ms=40.0,
#         expected_rise=9, expected_fall=10
#     )
#
#     print("\n=== SLM Response Time Analysis — PLICO (marker fix + latencies previous + trigger sim) ===")
#     print(f"File: {fname}")
#
#     # Tsum hint da PD1
#     rise_t90_pd1, fall_t10_pd1 = _get_rise_fall_times(out1, rise_key="t90", fall_key="t10")
#     per_r = _periods_from_times(rise_t90_pd1)
#     per_f = _periods_from_times(fall_t10_pd1)
#     Tsum_hint = np.nanmedian(np.concatenate([per_r, per_f])) if (per_r.size + per_f.size) else (2*dwell_time_in_s_hint)
#     Td_hint_uniform = Tsum_hint/2.0
#     print("\n--- Tsum hint (PD1) ---")
#     print(f"rise→rise: {_sdesc_ms(per_r)}")
#     print(f"fall→fall: {_sdesc_ms(per_f)}")
#     print(f"=> Tsum_hint ≈ {Tsum_hint*1e3:.3f} ms  ⇒ Td_hint_uniform ≈ {Td_hint_uniform*1e3:.3f} ms")
#
#     # Recovery PD1 per arrivare a 9 rise, 10 fall
#     _recover_missing_events_with_windows(
#         rta=rta, out_channel=out1, channel_name="PD1",
#         t0=t0_first_cmd_s, Td_hint=Td_hint_uniform, first_kind=FIRST_KIND,
#         want_rise=9, want_fall=10,
#         search_after_cmd_ms=80.0, safety_lead_ms=0.5
#     )
#     print(f"[CHECK] PD1 after recovery: rise={len(out1['rise'])} (atteso 9), fall={len(out1['fall'])} (atteso 10)")
#
#     # Griglia UNIFORME da t0 per assegnazione PRECEDENTE
#     tmin = float(rta.t[0]); tmax = float(rta.t[-1])
#     C_rise_uni, C_fall_uni = _build_uniform_grids_from_t0(t0_first_cmd_s, Td_hint_uniform, tmin, tmax, FIRST_KIND)
#
#     # L_r / L_f con definizione richiesta (PREVIOUS)
#     rise_events_t90 = np.array([m["t90"] for m in out1["rise"]], float)
#     fall_events_t10 = np.array([m["t10"] for m in out1["fall"]], float)
#     lat = compute_latencies_previous(
#         rise_events_t90=rise_events_t90,
#         fall_events_t10=fall_events_t10,
#         C_rise_cmd=C_rise_uni,
#         C_fall_cmd=C_fall_uni,
#     )
#     print_latency_report(lat, prefix="[UNIFORM/PREVIOUS]")
#
#     # === PLOT 1: con marker (v = t10 fall, ^ = t90 rise) ===
#     plot_pd_with_triggers_and_events(rta, out1, C_rise_uni, C_fall_uni, "PD1", t0=t0_first_cmd_s)
#
#     # === PLOT 2: panoramica PD1/PD2 ===
#     plt.figure()
#     plt.plot(rta.t, rta.v1, label="PD1 (V)")
#     plt.plot(rta.t, rta.v2, label="PD2 (V)")
#     plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)"); plt.title("Photodiode signals (PD1 & PD2)")
#     plt.legend(); plt.show()
#
#     # === PLOT 3: solo marker su PD1 (v = t10 fall, ^ = t90 rise) ===
#     plt.figure()
#     plt.plot(rta.t, rta.v1, label="PD1 (V)")
#     fall_t10 = [m["t10"] for m in out1["fall"]]
#     rise_t90 = [m["t90"] for m in out1["rise"]]
#     if len(fall_t10): plt.scatter(fall_t10, np.interp(fall_t10, rta.t, rta.v1), marker="v", s=38, label="fall t10 (marker)")
#     if len(rise_t90): plt.scatter(rise_t90, np.interp(rise_t90, rta.t, rta.v1), marker="^", s=38, label="rise t90 (marker)")
#     plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)")
#     plt.title("PD1 — markers: 'v'=fall t10, '^'=rise t90")
#     plt.legend(); plt.show()
#
#     # === PLOT 4: overlay locale (come prima) ===
#     def _overlay(ax, meas, fs, t, v, title):
#         win_ms = 20.0
#         n_win = max(1, int(round(fs * (win_ms / 1000.0))))
#         t10_rel = float(np.median([m["t10"] - m["t_center"] for m in meas])) if len(meas) else np.nan
#         t90_rel = float(np.median([m["t90"] - m["t_center"] for m in meas])) if len(meas) else np.nan
#         for m in meas:
#             i0 = m["idx"]; lo = max(0, i0 - n_win); hi = min(len(v)-1, i0 + n_win)
#             tt = t[lo:hi+1] - t[i0]; ss = v[lo:hi+1]
#             ax.plot(tt, ss, alpha=0.75)
#         if np.isfinite(t10_rel): ax.axvline(t10_rel, linestyle="--", label="t10 (median)")
#         if np.isfinite(t90_rel): ax.axvline(t90_rel, linestyle="--", label="t90 (median)")
#         ax.set_title(title); ax.set_xlabel("Time rel. to edge (s)"); ax.set_ylabel("Voltage (V)")
#         if ax.get_legend_handles_labels()[0]: ax.legend()
#
#     fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
#     _overlay(axes[0], out1["rise"], rta.fs, rta.t, rta.v1, "PD1 rising (overlay)")
#     _overlay(axes[1], out1["fall"], rta.fs, rta.t, rta.v1, "PD1 falling (overlay)")
#     fig.suptitle("Edges overlay (PD1) with median t10/t90", y=1.02)
#     plt.tight_layout(); plt.show()
#
#     # === CSV per-edge ===
#     import pandas as pd
#     def _dump_csv(out, fname_csv):
#         rows = []
#         for m in out["rise"]:
#             rows.append(dict(
#                 channel=out["channel"], kind="rise",
#                 t_center_s=m["t_center"], t10_s=m["t10"], t50_s=m["t50"], t90_s=m["t90"],
#                 width_10_90_ms=m.get("width_10_90_ms", np.nan),
#                 sigma_width_ms=m.get("sigma_width_ms", np.nan),
#                 tau_ms=m.get("tau_ms", np.nan),
#                 width_exp_ms=m.get("width_exp_ms", np.nan),
#                 exp_r2=m.get("exp_r2", np.nan)
#             ))
#         for m in out["fall"]:
#             rows.append(dict(
#                 channel=out["channel"], kind="fall",
#                 t_center_s=m["t_center"], t10_s=m["t10"], t50_s=m["t50"], t90_s=m["t90"],
#                 width_10_90_ms=m.get("width_10_90_ms", np.nan),
#                 sigma_width_ms=m.get("sigma_width_ms", np.nan),
#                 tau_ms=m.get("tau_ms", np.nan),
#                 width_exp_ms=m.get("width_exp_ms", np.nan),
#                 exp_r2=m.get("exp_r2", np.nan)
#             ))
#         pd.DataFrame(rows).to_csv(fname_csv, index=False)
#         print(f"Saved per-edge report: {fname_csv}")
#
#     _dump_csv(out1, os.path.join(os.getcwd(), "slm_edge_timings_PD1_plico.csv"))
#     _dump_csv(out2, os.path.join(os.getcwd(), "slm_edge_timings_PD2_plico.csv"))
#
#     # === (NUOVO) TRIGGER SIMULATI da PD1: anticipo + jitter ===
#     sim_pd1 = simulate_triggers_from_events(
#         out_channel=out1,
#         advance_ms=8.67,   # ← cambia a 5.0 se vuoi
#         jitter_ms=1.3,    # ← cambia a 3.0 per jitter maggiore
#         seed=42,
#         rise_key="t90",
#         fall_key="t10"
#     )
#     print_sim_stats(sim_pd1, label="PD1")
#     plot_simulated_triggers_PD1(rta, sim_pd1, t0_first_cmd_s=t0_first_cmd_s, title_suffix=" (sim)")
#     rise_events_t90 = np.array([m["t90"] for m in out1["rise"]], float)
#     fall_events_t10 = np.array([m["t10"] for m in out1["fall"]], float)
#
#     lat_sim = compute_latencies_previous(
#         rise_events_t90=rise_events_t90,
#         fall_events_t10=fall_events_t10,
#         C_rise_cmd=sim_pd1["C_rise_sim"],   # comandi simulati low→high (per i rise)
#         C_fall_cmd=sim_pd1["C_fall_sim"],   # comandi simulati high→low (per i fall)
#     )
#     print_latency_report(lat_sim, prefix="[SIM/PREVIOUS]")
#
#     # (opzionale) plot con i trigger simulati al posto della griglia uniforme
#     plot_pd_with_triggers_and_events(
#         rta, out1,
#         C_rise=sim_pd1["C_rise_sim"],
#         C_fall=sim_pd1["C_fall_sim"],
#         channel_title="PD1",
#         t0=t0_first_cmd_s,
# )
#
# # if __name__ == "__main__":
# #     main_plico_data()
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from typing import Optional, List, Dict, Tuple

from bronte.mains.main241014_slm_time_resp_blink import (
    ResponseTimeAnalyzer,
    _get_rise_fall_times,
)

# ========= utilità robuste =========
def _se_median_no_boot(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2: return float('nan')
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad / np.sqrt(n)

def _periods_from_times(ts: np.ndarray) -> np.ndarray:
    ts = np.sort(np.array(ts, float))
    return np.diff(ts) if ts.size >= 2 else np.array([], float)

def _sdesc_ms(x):
    if x.size == 0: return "n=0"
    return f"n={x.size}, mean={np.mean(x)*1e3:.3f} ms, median={np.median(x)*1e3:.3f} ms, std={np.std(x, ddof=1)*1e3:.3f} ms"

# ========= assegnazione al comando PRECEDENTE =========
def _previous_offsets(events: np.ndarray, grid: np.ndarray):
    """
    Per ogni evento, prende il comando PRECEDENTE della stessa griglia (stesso tipo).
    offsets = evento - comando_precedente. Scarta eventi prima del primo comando.
    """
    events = np.asarray(events, float)
    grid   = np.asarray(grid, float)
    if events.size == 0 or grid.size == 0:
        return np.array([], float), np.array([], float)
    idx  = np.searchsorted(grid, events, side="right") - 1
    mask = (idx >= 0)
    ev   = events[mask]
    tr   = grid[idx[mask]]
    return (ev - tr, tr)

# ========= griglia uniforme (stampella) =========
def _build_uniform_grids_from_t0(t0: float, Td: float, tmin: float, tmax: float, first_kind: str):
    """
    Griglia uniforme alternata a partire da t0.
    first_kind in {'rise','fall'}.
    """
    assert first_kind in ("rise","fall")
    lo = tmin - 2*Td; hi = tmax + 2*Td
    C_rise, C_fall = [], []

    # avanti
    t = t0; kind = first_kind
    while t <= hi:
        (C_rise if kind=="rise" else C_fall).append(t)
        t += Td
        kind = "fall" if kind=="rise" else "rise"

    # indietro
    t = t0; kind = first_kind
    while t >= lo:
        (C_rise if kind=="rise" else C_fall).append(t)
        t -= Td
        kind = "fall" if kind=="rise" else "rise"

    C_rise = np.array([c for c in C_rise if lo <= c <= hi], float)
    C_fall = np.array([c for c in C_fall if lo <= c <= hi], float)
    C_rise = np.unique(np.round(C_rise, 12)); C_rise.sort()
    C_fall = np.unique(np.round(C_fall, 12)); C_fall.sort()
    return C_rise, C_fall

# ========= recovery (se il detector ha perso transiente) =========
def _scan_first_edge_in_window(rta: ResponseTimeAnalyzer, channel: str,
                               kind_wanted: str, t_start: float, t_end: float,
                               step_samples: int = 5, pre_ms: float = 5.0, post_ms: float = 25.0):
    assert kind_wanted in ("rising","falling")
    i0 = int(np.searchsorted(rta.t, max(t_start, rta.t[0])))
    i1 = int(np.searchsorted(rta.t, min(t_end,   rta.t[-1])))
    i0 = max(1,i0); i1 = min(len(rta.t)-2, i1)
    if i1 <= i0: return None
    best = None
    for j in range(i0, i1, max(1, step_samples)):
        m = rta.measure_edge_on_channel(j, channel, pre_ms=pre_ms, post_ms=post_ms)
        if m is not None and m["kind"] == kind_wanted:
            if (best is None) or (m["t_center"] < best["t_center"]):
                best = m
    return best

def _recover_missing_events_with_windows(
    rta: ResponseTimeAnalyzer,
    out_channel: dict,
    channel_name: str,
    t0: float,
    Td_hint: float,
    first_kind: str,
    want_rise: int,
    want_fall: int,
    search_after_cmd_ms: float = 80.0,
    safety_lead_ms: float = 0.5
):
    """
    Usa griglia UNIFORME come guida. Per ogni comando atteso:
      rise: cerca il primo 'rising' in [cmd+lead, cmd+search]
      fall: cerca il primo 'falling' in [cmd+lead, cmd+search]
    Aggiunge se manca.
    """
    pre_ms = 5.0; post_ms = 25.0
    r_times = np.array([m["t90"] for m in out_channel["rise"]], float) if len(out_channel["rise"]) else np.array([], float)
    f_times = np.array([m["t10"] for m in out_channel["fall"]], float) if len(out_channel["fall"]) else np.array([], float)

    tmin = float(rta.t[0]); tmax = float(rta.t[-1])
    C_rise_uni, C_fall_uni = _build_uniform_grids_from_t0(t0, Td_hint, tmin, tmax, first_kind)

    def _already_has_event(existing_times, t_new, tol_ms=8.0):
        return existing_times.size>0 and (np.min(np.abs(existing_times - t_new)) <= tol_ms/1e3)

    # RISE
    for c in C_rise_uni:
        if len(out_channel["rise"]) >= want_rise: break
        w_lo = c + safety_lead_ms/1e3
        w_hi = c + search_after_cmd_ms/1e3
        m = _scan_first_edge_in_window(rta, channel_name, "rising", w_lo, w_hi,
                                       step_samples=3, pre_ms=pre_ms, post_ms=post_ms)
        if (m is not None) and (not _already_has_event(r_times, m["t90"])):  # t90 per rise
            out_channel["rise"].append(m); r_times = np.append(r_times, m["t90"])

    # FALL
    for c in C_fall_uni:
        if len(out_channel["fall"]) >= want_fall: break
        w_lo = c + safety_lead_ms/1e3
        w_hi = c + search_after_cmd_ms/1e3
        m = _scan_first_edge_in_window(rta, channel_name, "falling", w_lo, w_hi,
                                       step_samples=3, pre_ms=pre_ms, post_ms=post_ms)
        if (m is not None) and (not _already_has_event(f_times, m["t10"])):  # t10 per fall
            out_channel["fall"].append(m); f_times = np.append(f_times, m["t10"])

    out_channel["rise"].sort(key=lambda m: m["t90"])
    out_channel["fall"].sort(key=lambda m: m["t10"])
    
def _compute_dwell_times_from_events(out_channel: dict,
                                     rise_key: str = "t90",
                                     fall_key: str = "t10"):
    """
    Dwell @HIGH  = tempo tra un RISE (t90) e il FALL successivo (t10)
    Dwell @LOW   = tempo tra un FALL (t10) e il RISE successivo (t90)
    Ritorna i due insiemi in secondi.
    """
    rises = [(m.get(rise_key, np.nan), "rise") for m in out_channel.get("rise", [])]
    falls = [(m.get(fall_key, np.nan), "fall") for m in out_channel.get("fall", [])]
    ev = [(t, k) for (t, k) in (rises + falls) if np.isfinite(t)]
    if not ev:
        return dict(high=np.array([], float), low=np.array([], float))

    ev.sort(key=lambda x: x[0])  # ordina per tempo

    dwell_high, dwell_low = [], []
    for (t0, k0), (t1, k1) in zip(ev[:-1], ev[1:]):
        if k0 == "rise" and k1 == "fall":
            dwell_high.append(t1 - t0)   # plateau alto fra rise→fall
        elif k0 == "fall" and k1 == "rise":
            dwell_low.append(t1 - t0)    # plateau basso fra fall→rise
        # negli altri casi (duplicati o eventi mancanti) non aggiungo nulla

    return dict(high=np.array(dwell_high, float), low=np.array(dwell_low, float))


def _print_dwell_report_pd1(dwell: dict, label="PD1"):
    def _summ(a):
        if a.size == 0: return "n=0"
        return f"n={a.size}, mean={np.mean(a)*1e3:.3f} ms, std={np.std(a, ddof=1)*1e3:.3f} ms, median={np.median(a)*1e3:.3f} ms"

    high_ms = dwell["high"] * 1e3
    low_ms  = dwell["low"]  * 1e3

    print(f"\n[{label}] Dwell times (marker: rise=t90, fall=t10)")
    if high_ms.size:
        print("  Per-event @HIGH:")
        for i, v in enumerate(high_ms, 1):
            print(f"    H{i:02d}: {v:.3f} ms")
    else:
        print("  Per-event @HIGH: n=0")

    if low_ms.size:
        print("  Per-event @LOW:")
        for i, v in enumerate(low_ms, 1):
            print(f"    L{i:02d}: {v:.3f} ms")
    else:
        print("  Per-event @LOW: n=0")

    print(f"  Summary @HIGH -> {_summ(dwell['high'])}")
    print(f"  Summary @LOW  -> {_summ(dwell['low'])}")


def latencies_previous_with_window(
    C_rise, C_fall,
    rise_t90, fall_t10,
    min_ms: float = 0.0,
    max_ms: float = 80.0,
):
    """L_r = c_rise - prev(fall_t10),  L_f = c_fall - prev(rise_t90)
       con vincolo min_ms <= latenza <= max_ms.
    """
    C_rise  = np.asarray(C_rise,  float)
    C_fall  = np.asarray(C_fall,  float)
    rise_t90 = np.asarray(rise_t90, float)
    fall_t10 = np.asarray(fall_t10, float)

    def _prev_deltas(trigs, evs):
        if trigs.size == 0 or evs.size == 0:
            return np.array([], float)
        idx = np.searchsorted(evs, trigs, side="right") - 1  # evento PRECEDENTE
        ok  = (idx >= 0)
        d   = np.full(trigs.size, np.nan, float)
        d[ok] = trigs[ok] - evs[idx[ok]]
        lo, hi = min_ms/1e3, max_ms/1e3
        mask = np.isfinite(d) & (d >= lo) & (d <= hi)
        return d[mask]

    L_r = _prev_deltas(C_rise, fall_t10)  # trigger rise vs FALL precedente
    L_f = _prev_deltas(C_fall, rise_t90)  # trigger fall vs RISE precedente
    return L_r, L_f


# ========= latenze con definizione RICHIESTA =========
def compute_latencies_previous(
    rise_events_t90: np.ndarray,
    fall_events_t10: np.ndarray,
    C_rise_cmd: np.ndarray,   # comandi low→high (che portano a rise)
    C_fall_cmd: np.ndarray,   # comandi high→low (che portano a fall)
):
    """
    L_r = t90_rise − cmd_low→high PRECEDENTE
    L_f = t10_fall − cmd_high→low PRECEDENTE

    Ritorna sia stime a MEDIANA (robuste) sia a MEDIA (classiche),
    con errori standard, più statistiche per-evento.
    Tutte le quantità sono in secondi (la stampa le converte in ms).
    """
    # offsets per-evento rispetto al comando PRECEDENTE
    off_r, _ = _previous_offsets(np.asarray(rise_events_t90, float),
                                 np.asarray(C_rise_cmd, float))
    off_f, _ = _previous_offsets(np.asarray(fall_events_t10, float),
                                 np.asarray(C_fall_cmd, float))

    # ---------- stime a MEDIANA ----------
    Lr_med = float(np.nanmedian(off_r)) if off_r.size else np.nan
    Lf_med = float(np.nanmedian(off_f)) if off_f.size else np.nan
    se_Lr_med = _se_median_no_boot(off_r)
    se_Lf_med = _se_median_no_boot(off_f)

    # ---------- stime a MEDIA ----------
    def _safe_mean(x): return float(np.mean(x)) if x.size else np.nan
    def _safe_std(x):  return float(np.std(x, ddof=1)) if x.size > 1 else (0.0 if x.size == 1 else np.nan)
    def _safe_min(x):  return float(np.min(x)) if x.size else np.nan
    def _safe_max(x):  return float(np.max(x)) if x.size else np.nan

    Lr_mean = _safe_mean(off_r)
    Lf_mean = _safe_mean(off_f)
    std_Lr  = _safe_std(off_r)
    std_Lf  = _safe_std(off_f)
    se_Lr_mean = (std_Lr / np.sqrt(off_r.size)) if off_r.size > 1 else (0.0 if off_r.size == 1 else np.nan)
    se_Lf_mean = (std_Lf / np.sqrt(off_f.size)) if off_f.size > 1 else (0.0 if off_f.size == 1 else np.nan)

    # ---------- range & diagnostica ----------
    min_Lr = _safe_min(off_r); max_Lr = _safe_max(off_r)
    min_Lf = _safe_min(off_f); max_Lf = _safe_max(off_f)
    rng_Lr = (max_Lr - min_Lr) if np.isfinite(min_Lr) and np.isfinite(max_Lr) else np.nan
    rng_Lf = (max_Lf - min_Lf) if np.isfinite(min_Lf) and np.isfinite(max_Lf) else np.nan

    # residui rispetto alle mediane (diagnostica combinata)
    res = np.concatenate([(off_r - Lr_med) if np.isfinite(Lr_med) else np.array([]),
                          (off_f - Lf_med) if np.isfinite(Lf_med) else np.array([])])
    mad = 1.4826 * np.median(np.abs(res - np.median(res))) if res.size else np.nan
    rms = float(np.sqrt(np.mean(res**2))) if res.size else np.nan

    return dict(
        # offset per-evento (se vuoi ispezionarli)
        off_r=off_r, off_f=off_f,
        n_r=int(off_r.size), n_f=int(off_f.size),

        # stime a MEDIANA (robuste)
        Lr=Lr_med, Lf=Lf_med,           # alias: Lr/Lf = mediana (per compatibilità retro)
        Lr_med=Lr_med, Lf_med=Lf_med,
        se_Lr=se_Lr_med, se_Lf=se_Lf_med,
        se_Lr_med=se_Lr_med, se_Lf_med=se_Lf_med,

        # stime a MEDIA (classiche)
        Lr_mean=Lr_mean, Lf_mean=Lf_mean,
        se_Lr_mean=se_Lr_mean, se_Lf_mean=se_Lf_mean,

        # dispersioni/estremi per-evento
        std_Lr=std_Lr, std_Lf=std_Lf,
        min_Lr=min_Lr, max_Lr=max_Lr, rng_Lr=rng_Lr,
        min_Lf=min_Lf, max_Lf=max_Lf, rng_Lf=rng_Lf,

        # diagnostiche globali sui residui vs mediana
        residual_mad=mad, residual_rms=rms,
    )


def print_latency_report(lat: dict, prefix: str = ""):
    """Stampa compatta (in ms) di mediana vs media, errori standard, std e range."""
    p = (prefix + " ").strip()
    ms = lambda x: (x * 1e3) if np.isfinite(x) else np.nan

    # --- MEDIANA (robusta) ---
    print(f"{p}MEDIAN  L_r ≈ {ms(lat['Lr_med']):.3f} ms  ± {ms(lat['se_Lr_med']):.3f} ms [SE(median)]")
    print(f"{p}MEDIAN  L_f ≈ {ms(lat['Lf_med']):.3f} ms  ± {ms(lat['se_Lf_med']):.3f} ms [SE(median)]")

    # --- MEDIA (classica) ---
    print(f"{p}MEAN    L_r ≈ {ms(lat['Lr_mean']):.3f} ms  ± {ms(lat['se_Lr_mean']):.3f} ms [SE(mean)]")
    print(f"{p}MEAN    L_f ≈ {ms(lat['Lf_mean']):.3f} ms  ± {ms(lat['se_Lf_mean']):.3f} ms [SE(mean)]")

    # --- dispersioni e range per-evento ---
    if 'n_r' in lat and 'n_f' in lat:
        print(f"{p}Per-event L_r: N={lat['n_r']}, std≈{ms(lat['std_Lr']):.3f} ms, "
              f"min={ms(lat['min_Lr']):.3f} ms, max={ms(lat['max_Lr']):.3f} ms, "
              f"range={ms(lat['rng_Lr']):.3f} ms")
        print(f"{p}Per-event L_f: N={lat['n_f']}, std≈{ms(lat['std_Lf']):.3f} ms, "
              f"min={ms(lat['min_Lf']):.3f} ms, max={ms(lat['max_Lf']):.3f} ms, "
              f"range={ms(lat['rng_Lf']):.3f} ms")

    # --- diagnostiche globali sui residui vs mediana ---
    if 'residual_mad' in lat and 'residual_rms' in lat:
        print(f"{p}Residuals (rise+fall vs median): MAD≈{ms(lat['residual_mad']):.3f} ms | RMS≈{ms(lat['residual_rms']):.3f} ms")


# ========= PLOT con marker come da tua specifica (FIX) =========
def plot_pd_with_triggers_and_events(rta, out_channel, C_rise, C_fall, channel_title, t0=None, send_req=None):
    """
    - Linee sottili: trigger rise/fall
    - Linee tratteggiate: tempi usati per le latenze (rise=t90, fall=t10)
    - Marker (come richiesto): 'v' = t10 dei FALL, '^' = t90 dei RISE
    """
    if channel_title.upper() == "PD1":
        t, v = rta.t, rta.v1
    else:
        t, v = rta.t, rta.v2

    rise_t90 = np.array([m["t90"] for m in out_channel["rise"]], float) if len(out_channel["rise"]) else np.array([], float)
    fall_t10 = np.array([m["t90"] for m in out_channel["fall"]], float) if len(out_channel["fall"]) else np.array([], float)
    
    plt.figure(figsize=(12,5))
    plt.plot(t, v, lw=0.9, alpha=0.75, label=f"{channel_title}")

    # if (t0 is not None) and (t[0] <= t0 <= t[-1]):
    #     plt.axvline(t0, linewidth=2.0, alpha=0.7, label="t0 (command)")
    
    send_delay_ms   = 79.3   # quanto PRIMA del trigger è stato inviato il comando
    send_jitter_ms  = 5.45    # jitter (dev std) sul tempo di invio
    _rng = np.random.default_rng(654)  # seed fisso per riproducibilità
    _delay_s  = send_delay_ms / 1e3
    _jitter_s = send_jitter_ms / 1e3
    
    #trigger
    for c in C_rise:
        if t[0] <= c <= t[-1]:
            plt.axvline(c, alpha=0.85,color ='r', linewidth=1.0, label="trigger rise")
            #s = c - _delay_s + _rng.normal(0.0, _jitter_s)
            #plt.axvline(s, alpha=0.85,color ='g', linewidth=1.0, label="send req(rise)")
    for c in C_fall:
        if t[0] <= c <= t[-1]: 
            plt.axvline(c, alpha=0.85,color='r', linewidth=1.0, label="trigger fall")
            #s = c - _delay_s + _rng.normal(0.0, _jitter_s)
            #plt.axvline(s, alpha=0.85, color = 'g', linewidth=1.0, label="send req(fall)")


    # linee tratteggiate = metrica latenze
    for tr in rise_t90:
        if t[0] <= tr <= t[-1]:
            plt.axvline(tr, linestyle="--", alpha=0.9, linewidth=1.3, label="rise t90 (lat)")
    for tf in fall_t10:
        if t[0] <= tf <= t[-1]: plt.axvline(tf, linestyle="--", alpha=0.9, linewidth=1.3, label="fall t10 (lat)")

    # marker: 'v' = t10 FALL, '^' = t90 RISE
    if fall_t10.size:
        plt.scatter(fall_t10, np.interp(fall_t10, t, v), s=36, marker="v", label=r"$0.1\Delta V$")
    if rise_t90.size:
        plt.scatter(rise_t90, np.interp(rise_t90, t, v), s=36, marker="^", label=r"$0.9\Delta V$")
    

    # legenda unica
    h, l = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(l, h))
    plt.legend(uniq.values(), uniq.keys(), loc="best")
    plt.title(f"PLICO control latency: Dwell = 50 ms")
    plt.xlabel("Time (s)"); plt.ylabel("Signal [V]")
    plt.xlim(2.9575,3.3022)
    plt.tight_layout(); plt.show()
    
    def _prev_deltas(trigs, evs):
        if (trigs is None) or (len(trigs) == 0) or (evs.size == 0):
            return np.array([], float)
        trigs = np.asarray(trigs, float)
        evs_sorted = np.asarray(evs, float)
        idx = np.searchsorted(evs_sorted, trigs, side="right") - 1
        valid = idx >= 0
        deltas = np.full(trigs.size, np.nan, float)
        deltas[valid] = trigs[valid] - evs_sorted[idx[valid]]
        return deltas[np.isfinite(deltas)]
    
    def _next_deltas(trigs, evs):
        """Per ogni trigger, usa il PRIMO evento evs >= trigger."""
        if (trigs is None) or (len(trigs) == 0) or (evs.size == 0):
            return np.array([], float)
        trigs = np.asarray(trigs, float)
        evs_sorted = np.asarray(evs, float)
        idx = np.searchsorted(evs_sorted, trigs, side="left")  # next/uguale
        valid = idx < evs_sorted.size
        deltas = np.full(trigs.size, np.nan, float)
        deltas[valid] = evs_sorted[idx[valid]] - trigs[valid]
        return deltas[np.isfinite(deltas)]


    L_r = _next_deltas(C_rise, fall_t10)
    L_f = _next_deltas(C_fall, rise_t90)
    

    # stampa elenco + statistiche in ms
    def _ms_stats(arr):
        if arr.size == 0: return "n=0"
        mu = np.mean(arr)*1e3
        sd = (np.std(arr, ddof=1)*1e3) if arr.size > 1 else 0.0
        med = np.median(arr)*1e3
        return f"n={arr.size}, mean={mu:.3f} ms, std={sd:.3f} ms, median={med:.3f} ms"

    print("\n[Latencies within plot_pd_with_triggers_and_events]")
    print("L_r (trigger rise − prev fall_t10) [ms]:", np.round(L_r*1e3, 3).tolist())
    print("   Summary:", _ms_stats(L_r))
    print("L_f (trigger fall − prev rise_t90) [ms]:", np.round(L_f*1e3, 3).tolist())
    print("   Summary:", _ms_stats(L_f))
    
    
    rise_t90 = np.array([m["t90"] for m in out_channel["rise"]], float)
    fall_t10 = np.array([m["t10"] for m in out_channel["fall"]], float)  # <-- FIX

    L_r, L_f = latencies_previous_with_window(
        C_rise=C_rise, C_fall=C_fall,
        rise_t90=rise_t90, fall_t10=fall_t10,
        min_ms=0.0, max_ms=80.0  # adatta max_ms al tuo search window
    )

    print("L_r [ms]:", np.round(L_r*1e3, 3).tolist())
    print("L_f [ms]:", np.round(L_f*1e3, 3).tolist())
    print(f"L_r mean±std = {np.mean(L_r)*1e3:.2f} ± {np.std(L_r, ddof=1)*1e3:.2f} ms")
    print(f"L_f mean±std = {np.mean(L_f)*1e3:.2f} ± {np.std(L_f, ddof=1)*1e3:.2f} ms")


def derive_send_request_triggers(sim: dict,
                                 transport_ms: float = 80.0,
                                 send_jitter_ms: float = 5.0,
                                 seed: int | None = 7):
    """
    Genera tempi 'send_request' (lato client) a partire dai trigger simulati lato dispositivo
    contenuti in `sim` (output di simulate_triggers_from_events).

    Modello:
      t_send_request = t_device_trigger - transport_ms + N(0, send_jitter_ms)

    Parametri
    ---------
    sim : dict
        Output di `simulate_triggers_from_events`, con chiavi:
        - "C_rise_sim"  (triggers dispositivo per i RISE, in secondi)
        - "C_fall_sim"  (triggers dispositivo per i FALL, in secondi)
    transport_ms : float
        Ritardo medio software/trasporto (ms) tra invio e ricezione lato dispositivo.
    send_jitter_ms : float
        Jitter gaussiano (ms) da sommare al modello.
    seed : int | None
        Seed per RNG.

    Ritorna
    -------
    dict con:
      - device_rise, device_fall          (eco dei trigger dispositivo)
      - send_request_rise, send_request_fall
      - delta_r, delta_f                  = device - send_request (attesi ~ transport_ms)
      - transport_ms, send_jitter_ms
    """
    rng = np.random.default_rng(seed)

    dev_r = np.array(sim.get("C_rise_sim", []), float)
    dev_f = np.array(sim.get("C_fall_sim", []), float)

    dt = transport_ms / 1e3
    sj = send_jitter_ms / 1e3

    send_r = dev_r - dt + (rng.normal(0.0, sj, size=dev_r.size) if dev_r.size else np.array([], float))
    send_f = dev_f - dt + (rng.normal(0.0, sj, size=dev_f.size) if dev_f.size else np.array([], float))

    # Diagnostica: quanto vale (device - send_request)? atteso ~ transport_ms
    delta_r = dev_r - send_r
    delta_f = dev_f - send_f

    return dict(
        device_rise=dev_r,
        device_fall=dev_f,
        send_request_rise=send_r,
        send_request_fall=send_f,
        delta_r=delta_r,
        delta_f=delta_f,
        transport_ms=float(transport_ms),
        send_jitter_ms=float(send_jitter_ms),
    )

def print_send_request_stats(sr: dict, label: str = "send_request"):
    def _sdesc(x):
        x_ms = np.asarray(x) * 1e3
        if x_ms.size == 0:
            return "n=0"
        mu = float(np.mean(x_ms))
        sd = float(np.std(x_ms, ddof=1) if x_ms.size > 1 else 0.0)
        med = float(np.median(x_ms))
        return f"n={x_ms.size}, mean={mu:.3f} ms, median={med:.3f} ms, std={sd:.3f} ms"

    print(f"\n--- {label} diagnostics ---")
    print(f"device - send_request (RISE): {_sdesc(sr['delta_r'])}  (target ≈ {sr['transport_ms']:.1f} ms)")
    print(f"device - send_request (FALL): {_sdesc(sr['delta_f'])}  (target ≈ {sr['transport_ms']:.1f} ms)")



# ========= (NUOVO) — SIMULAZIONE TRIGGER dai tuoi eventi =========
def simulate_triggers_from_events(out_channel: dict,
                                  advance_ms: float = 6.5,   # usa 5.0 se vuoi il set "classico"
                                  jitter_ms: float  = 0.3,   # usa 3.0 per jitter più grande
                                  seed: int | None  = 42,
                                  rise_key: str = "t90",
                                  fall_key: str = "t10"):
    """
    Crea tempi di trigger sintetici per ogni evento:
      - per i RISE: trigger = t90 − advance_ms + N(0, jitter_ms)
      - per i FALL: trigger = t10 − advance_ms + N(0, jitter_ms)
    Restituisce anche gli offset (evento − trigger), attesi ≈ +advance_ms.
    """
    rng = np.random.default_rng(seed)
    r_times = np.array([m[rise_key] for m in out_channel["rise"]], float) if len(out_channel["rise"]) else np.array([], float)
    f_times = np.array([m[fall_key] for m in out_channel["fall"]], float) if len(out_channel["fall"]) else np.array([], float)

    adv_s    = advance_ms / 1e3
    jitter_s = jitter_ms  / 1e3

    # genera i trigger simulati
    C_rise_sim = r_times - adv_s + rng.normal(0.0, jitter_s, size=r_times.size) if r_times.size else np.array([], float)
    C_fall_sim = f_times - adv_s + rng.normal(0.0, jitter_s, size=f_times.size) if f_times.size else np.array([], float)
    
    # --- vincolo: i trigger non devono superare i rispettivi eventi ---
    guard_ms = 0.06  # margine di sicurezza per evitare uguaglianze numeriche
    guard_s  = guard_ms / 1e3

    if r_times.size:
        # forza C_rise_sim <= r_times - guard_s
        C_rise_sim = np.minimum(C_rise_sim, r_times - guard_s)
    
    if f_times.size:
        # forza C_fall_sim <= f_times - guard_s
        C_fall_sim = np.minimum(C_fall_sim, f_times - guard_s)

    off_r = r_times - C_rise_sim
    off_f = f_times - C_fall_sim

    return dict(
        rise_times=r_times, fall_times=f_times,
        C_rise_sim=C_rise_sim, C_fall_sim=C_fall_sim,
        off_r=off_r, off_f=off_f,
        advance_ms=advance_ms, jitter_ms=jitter_ms
    )

def plot_simulated_triggers_PD1(rta, sim: dict, send_req: dict, t0_first_cmd_s: float | None = None, title_suffix: str = ""):
    """Mostra segnale PD1, eventi (t90/t10) e trigger simulati con anticipo e jitter."""
    t = rta.t; v = rta.v1
    plt.figure(figsize=(12,5))
    plt.plot(t, v, lw=0.9, alpha=0.75, label="PD1 (V)")

    if t0_first_cmd_s is not None and (t[0] <= t0_first_cmd_s <= t[-1]):
        plt.axvline(t0_first_cmd_s, linewidth=2.0, alpha=0.6, label="t0 (command)")

    # eventi (tratteggiati)
    for tr in sim["rise_times"]:
        if t[0] <= tr <= t[-1]:
            plt.axvline(tr, linestyle="--", alpha=0.85, linewidth=1.3, label="rise t90")
    for tf in sim["fall_times"]:
        if t[0] <= tf <= t[-1]:
            plt.axvline(tf, linestyle="--", alpha=0.85, linewidth=1.3, label="fall t10")

    # trigger simulati (sottili)
    for c in sim["C_rise_sim"]:
        if t[0] <= c <= t[-1]:
            plt.axvline(c, alpha=0.35, linewidth=1.0, label="trigger (sim) rise")
    for c in sim["C_fall_sim"]:
        if t[0] <= c <= t[-1]:
            plt.axvline(c, alpha=0.35, linewidth=1.0, label="trigger (sim) fall")
            
    # >>> AGGIUNTA: verticali dei tempi di "send request"
    if send_req is not None:
        for s in np.asarray(send_req.get("send_request_rise", []), float):
            if t[0] <= s <= t[-1]:
                plt.axvline(s, linestyle=":", linewidth=1.2, alpha=0.9, label="send request (rise)")
        for s in np.asarray(send_req.get("send_request_fall", []), float):
            if t[0] <= s <= t[-1]:
                plt.axvline(s, linestyle=":", linewidth=1.2, alpha=0.9, label="send request (fall)")

    # marker: '^' = rise t90, 'v' = fall t10
    if sim["rise_times"].size:
        plt.scatter(sim["rise_times"], np.interp(sim["rise_times"], t, v), s=28, marker="^", label="marker t90 (rise)")
    if sim["fall_times"].size:
        plt.scatter(sim["fall_times"], np.interp(sim["fall_times"], t, v), s=28, marker="v", label="marker t10 (fall)")

    h, l = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(l, h))
    plt.legend(uniq.values(), uniq.keys(), loc="best")

    plt.title(f"PD1: eventi & trigger simulati (−{sim['advance_ms']:.1f} ms, σ≈{sim['jitter_ms']:.1f} ms){title_suffix}")
    plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)")
    plt.tight_layout(); plt.show()



def print_sim_stats(sim: dict, label: str = "PD1"):
    def _sdesc(x):
        x_ms = np.asarray(x)*1e3
        if x_ms.size == 0:
            return "n=0"
        mu = float(np.mean(x_ms)); sd = float(np.std(x_ms, ddof=1) if x_ms.size > 1 else 0.0)
        med = float(np.median(x_ms))
        return f"n={x_ms.size}, mean={mu:.3f} ms, median={med:.3f} ms, std={sd:.3f} ms"
    print(f"\n--- Simulated trigger offsets (evento − trigger) [{label}] ---")
    print(f"Rise: {_sdesc(sim['off_r'])}  (target ≈ +{sim['advance_ms']:.1f} ms, std ≈ {sim['jitter_ms']:.1f} ms)")
    print(f"Fall: {_sdesc(sim['off_f'])}  (target ≈ +{sim['advance_ms']:.1f} ms, std ≈ {sim['jitter_ms']:.1f} ms)")

# ====== (1) STIMA 10–90% PER TRANSIENTE (interp, sempre rispetto a v0) ======
def _edge_widths_ms(out_channel: dict):
    """10–90% width per transiente (interp), SEMPRE riferito a v0.
    rise:  t90 - t10
    fall:  t10 - t90  (per avere una durata positiva)
    """
    rise_w, fall_w = [], []
    for m in out_channel.get("rise", []):
        t10 = m.get("t10", np.nan); t90 = m.get("t90", np.nan)
        if np.isfinite(t10) and np.isfinite(t90):
            w = (t90 - t10) * 1e3
            if w > 0: rise_w.append(w)
    for m in out_channel.get("fall", []):
        t10 = m.get("t10", np.nan); t90 = m.get("t90", np.nan)
        if np.isfinite(t10) and np.isfinite(t90):
            w = (t10 - t90) * 1e3
            if w > 0: fall_w.append(w)
    return np.array(rise_w, float), np.array(fall_w, float)


# ====== (2) PRINT RIEPILOGO 10–90% ======
def _print_edge_widths_summary(name: str, out_channel: dict):
    r_ms, f_ms = _edge_widths_ms(out_channel)
    def _fmt(x):
        if x.size == 0: return "n=0"
        return f"n={x.size}, mean={np.mean(x):.3f} ms, median={np.median(x):.3f} ms, std={np.std(x, ddof=1):.3f} ms"
    print(f"\n[{name}] 10–90% widths (interp, vs v0)")
    print(f"  Rise: { _fmt(r_ms) }")
    print(f"  Fall: { _fmt(f_ms) }")

# ========= MAIN =========
def main_plico_data():
    """
    - Detection su PD1/PD2
    - Recovery per centrare i conteggi (PD1: 9 rise, 10 fall)
    - Griglia UNIFORME da t0 (Td ≈ Tsum/2) per assegnazione PRECEDENTE
    - L_r, L_f coerenti con le definizioni
    - Plot con marker FIX: 'v' = t10 fall, '^' = t90 rise
    - (NUOVO) Simulazione trigger dai tuoi eventi con anticipo + jitter
    - (AGGIUNTA) Stima e print 10–90% (interp, vs v0)
    """
    # === file ===
    FDIR = "D:\\phd_slm_edo\\old_data\\slm_time_response\\photodiode\\"
    fname = FDIR + "20240906_1232_1\\Analog - 9-6-2024 12-32-43.52892 PM.csv"

    dwell_time_in_s_hint = 117e-3
    t0_first_cmd_s = 3.0404      # primo comando = FALL
    FIRST_KIND = "fall"          # sequenza alternata da t0

    if not os.path.exists(fname):
        alt = os.path.join(os.getcwd(), "Analog - 9-6-2024 12-32-43.52892 PM.csv")
        if os.path.exists(alt): fname = alt
        
        
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
    rta = ResponseTimeAnalyzer(fname)

    out1 = rta.analyze_one_channel_with_dwell(
        "PD1", dwell_time_in_s_hint,
        grad_threshold_k=6.0, min_separation_ms=70.0,
        pre_ms=5.0, post_ms=25.0, search_ms=40.0,
        expected_rise=9, expected_fall=10
    )
    out2 = rta.analyze_one_channel_with_dwell(
        "PD2", dwell_time_in_s_hint,
        grad_threshold_k=6.0, min_separation_ms=70.0,
        pre_ms=5.0, post_ms=25.0, search_ms=40.0,
        expected_rise=9, expected_fall=10
    )

    print("\n=== SLM Response Time Analysis — PLICO (marker fix + latencies previous + trigger sim) ===")
    print(f"File: {fname}")

    # >>> (AGGIUNTA) RIEPILOGO 10–90% PRIMA DEL RECOVERY
    _print_edge_widths_summary("PD1 (pre-recovery)", out1)
    _print_edge_widths_summary("PD2 (pre-recovery)", out2)
    _print_edge_widths_summary("PD1 (post-recovery)", out1)
    _print_edge_widths_summary("PD2 (post-recovery)", out2)

    # Tsum hint da PD1
    rise_t90_pd1, fall_t10_pd1 = _get_rise_fall_times(out1, rise_key="t90", fall_key="t10")
    per_r = _periods_from_times(rise_t90_pd1)
    per_f = _periods_from_times(fall_t10_pd1)
    Tsum_hint = np.nanmedian(np.concatenate([per_r, per_f])) if (per_r.size + per_f.size) else (2*dwell_time_in_s_hint)
    Td_hint_uniform = Tsum_hint/2.0
    print("\n--- Tsum hint (PD1) ---")
    print(f"rise→rise: {_sdesc_ms(per_r)}")
    print(f"fall→fall: {_sdesc_ms(per_f)}")
    print(f"=> Tsum_hint ≈ {Tsum_hint*1e3:.3f} ms  ⇒ Td_hint_uniform ≈ {Td_hint_uniform*1e3:.3f} ms")

    # Recovery PD1 per arrivare a 9 rise, 10 fall
    _recover_missing_events_with_windows(
        rta=rta, out_channel=out1, channel_name="PD1",
        t0=t0_first_cmd_s, Td_hint=Td_hint_uniform, first_kind=FIRST_KIND,
        want_rise=9, want_fall=10,
        search_after_cmd_ms=80.0, safety_lead_ms=0.5
    )
    print(f"[CHECK] PD1 after recovery: rise={len(out1['rise'])} (atteso 9), fall={len(out1['fall'])} (atteso 10)")

    # >>> (AGGIUNTA) RIEPILOGO 10–90% DOPO IL RECOVERY (PD1)
    _print_edge_widths_summary("PD1 (post-recovery)", out1)

    # Griglia UNIFORME da t0 per assegnazione PRECEDENTE
    tmin = float(rta.t[0]); tmax = float(rta.t[-1])
    C_rise_uni, C_fall_uni = _build_uniform_grids_from_t0(t0_first_cmd_s, Td_hint_uniform, tmin, tmax, FIRST_KIND)

    # L_r / L_f con definizione richiesta (PREVIOUS)
    rise_events_t90 = np.array([m["t90"] for m in out1["rise"]], float)
    fall_events_t10 = np.array([m["t10"] for m in out1["fall"]], float)
    lat = compute_latencies_previous(
        rise_events_t90=rise_events_t90,
        fall_events_t10=fall_events_t10,
        C_rise_cmd=C_rise_uni,
        C_fall_cmd=C_fall_uni,
    )
    print_latency_report(lat, prefix="[UNIFORM/PREVIOUS]")

    # === PLOT 1: con marker (v = t10 fall, ^ = t90 rise) ===
    plot_pd_with_triggers_and_events(rta, out1, C_rise_uni, C_fall_uni, "PD1", t0=t0_first_cmd_s)

    # === PLOT 2: panoramica PD1/PD2 ===
    plt.figure()
    plt.plot(rta.t, rta.v1, label="PD1 (V)")
    plt.plot(rta.t, rta.v2, label="PD2 (V)")
    plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)"); plt.title("Photodiode signals (PD1 & PD2)")
    plt.legend(); plt.show()

    # === PLOT 3: solo marker su PD1 (v = t10 fall, ^ = t90 rise) ===
    plt.figure()
    plt.plot(rta.t, rta.v1, label="PD1 (V)")
    fall_t10 = [m["t10"] for m in out1["fall"]]
    rise_t90 = [m["t90"] for m in out1["rise"]]
    if len(fall_t10): plt.scatter(fall_t10, np.interp(fall_t10, rta.t, rta.v1), marker="v", s=38, label="fall t10 (marker)")
    if len(rise_t90): plt.scatter(rise_t90, np.interp(rise_t90, rta.t, rta.v1), marker="^", s=38, label="rise t90 (marker)")
    plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)")
    plt.title("PD1 — markers: 'v'=fall t10, '^'=rise t90")
    plt.legend(); plt.show()

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
    fig5.suptitle("Transient signals on PD1", y=0.98)
    fig5.tight_layout()
    plt.show()

    # === CSV per-edge ===
    import pandas as pd
    def _dump_csv(out, fname_csv):
        rows = []
        for m in out["rise"]:
            rows.append(dict(
                channel=out["channel"], kind="rise",
                t_center_s=m["t_center"], t10_s=m["t10"], t50_s=m["t50"], t90_s=m["t90"],
                width_10_90_ms=m.get("width_10_90_ms", np.nan),
                sigma_width_ms=m.get("sigma_width_ms", np.nan),
                tau_ms=m.get("tau_ms", np.nan),
                width_exp_ms=m.get("width_exp_ms", np.nan),
                exp_r2=m.get("exp_r2", np.nan)
            ))
        for m in out["fall"]:
            rows.append(dict(
                channel=out["channel"], kind="fall",
                t_center_s=m["t_center"], t10_s=m["t10"], t50_s=m["t50"], t90_s=m["t90"],
                width_10_90_ms=m.get("width_10_90_ms", np.nan),
                sigma_width_ms=m.get("sigma_width_ms", np.nan),
                tau_ms=m.get("tau_ms", np.nan),
                width_exp_ms=m.get("width_exp_ms", np.nan),
                exp_r2=m.get("exp_r2", np.nan)
            ))
        pd.DataFrame(rows).to_csv(fname_csv, index=False)
        print(f"Saved per-edge report: {fname_csv}")

    _dump_csv(out1, os.path.join(os.getcwd(), "slm_edge_timings_PD1_plico.csv"))
    _dump_csv(out2, os.path.join(os.getcwd(), "slm_edge_timings_PD2_plico.csv"))

    # === (NUOVO) TRIGGER SIMULATI da PD1: anticipo + jitter ===
    sim_pd1 = simulate_triggers_from_events(
        out_channel=out1,
        advance_ms=0.65,   # ← cambia a 5.0 se vuoi
        jitter_ms=0.02,    # ← cambia a 3.0 per jitter maggiore
        seed=42,
        rise_key="t10",
        fall_key="t10"
    )
    print_sim_stats(sim_pd1, label="PD1")
    plot_simulated_triggers_PD1(rta, sim_pd1,None, t0_first_cmd_s=t0_first_cmd_s, title_suffix=" (sim)")
    rise_events_t90 = np.array([m["t90"] for m in out1["rise"]], float)
    fall_events_t10 = np.array([m["t10"] for m in out1["fall"]], float)
    
    lat_sim = compute_latencies_previous(
        rise_events_t90=rise_events_t90,
        fall_events_t10=fall_events_t10,
        C_rise_cmd=sim_pd1["C_rise_sim"],   # comandi simulati low→high (per i rise)
        C_fall_cmd=sim_pd1["C_fall_sim"],   # comandi simulati high→low (per i fall)
    )
    print_latency_report(lat_sim, prefix="[SIM/PREVIOUS]")
    
    # (opzionale) plot con i trigger simulati al posto della griglia uniforme
    sim_pd1 = simulate_triggers_from_events(out1, advance_ms=3.67, jitter_ms=0.3, seed=42)
    sr_pd1 = derive_send_request_triggers(sim_pd1, transport_ms=80.3, send_jitter_ms=6.45, seed=123)
    plot_pd_with_triggers_and_events(
        rta, out1,
        C_rise=sim_pd1["C_rise_sim"],
        C_fall=sim_pd1["C_fall_sim"],
        channel_title="PD1",
        t0=t0_first_cmd_s,
        send_req=sr_pd1
        )
    
        # --- Dwell time per PD1 (usando rise=t90, fall=t10) ---
    dwell_pd1 = _compute_dwell_times_from_events(out1, rise_key="t90", fall_key="t10")
    _print_dwell_report_pd1(dwell_pd1, label="PD1")
    
    
    print_send_request_stats(sr_pd1, label="PD1 send_request")

def main_trise_tfall_plico(t_offset = 0):
    FDIR = "D:\\phd_slm_edo\\old_data\\slm_time_response\\photodiode\\"
    fname = FDIR + "20240906_1232_1\\Analog - 9-6-2024 12-32-43.52892 PM.csv"

    dwell_time_in_s = 117e-3
    t0_first_cmd_s = 3.0404      # primo comando = FALL
    FIRST_KIND = "fall"          # sequenza alternata da t0

    if not os.path.exists(fname):
        alt = os.path.join(os.getcwd(), "Analog - 9-6-2024 12-32-43.52892 PM.csv")
        if os.path.exists(alt): fname = alt
        
        
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
    
    rta = ResponseTimeAnalyzer(fname)



    # PD1 e PD2, stessa pipeline
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
    
    
    # ---- hint Tsum e griglia uniforme ----
    rise_t90_pd1, fall_t10_pd1 = _get_rise_fall_times(out1, rise_key="t90", fall_key="t10")
    per_r = _periods_from_times(rise_t90_pd1)
    per_f = _periods_from_times(fall_t10_pd1)
    Tsum_hint = np.nanmedian(np.concatenate([per_r, per_f])) if (per_r.size + per_f.size) else (2*dwell_time_in_s)
    Td_hint = Tsum_hint/2.0
    
    _recover_missing_events_with_windows(
        rta=rta, out_channel=out1, channel_name="PD1",
        t0=t0_first_cmd_s, Td_hint=Td_hint, first_kind=FIRST_KIND,
        want_rise=9, want_fall=10, search_after_cmd_ms=80.0, safety_lead_ms=0.5
    )
    _recover_missing_events_with_windows(
        rta=rta, out_channel=out2, channel_name="PD2",
        t0=t0_first_cmd_s, Td_hint=Td_hint, first_kind=FIRST_KIND,
        want_rise=9, want_fall=10, search_after_cmd_ms=80.0, safety_lead_ms=0.5
    )
    def _filter_edges(edges, width_ms_bounds=(0.3, 2.0), min_step_frac=1):
        """Filtra outlier per larghezza 10–90% con fallback robusto.
           Se l'ampiezza del passo non è disponibile/affidabile, NON la usa."""
        if not edges:
            return []
    
        # --- larghezze 10–90 in ms ---
        w = np.array([e.get("width_10_90_ms", np.nan) for e in edges], float)
        finite_w = np.isfinite(w)
        if finite_w.sum() == 0:
            return edges  # non so stimare le larghezze: non filtro nulla
    
        # limiti robusti via IQR (oltre a bounds assoluti)
        q1, q3 = np.nanpercentile(w[finite_w], [25, 75])
        iqr = q3 - q1
        lo_iqr, hi_iqr = (q1 - 2.2*iqr, q3 + 2.2*iqr)
        lo = max(width_ms_bounds[0], lo_iqr)
        hi = min(width_ms_bounds[1], hi_iqr)
    
        # --- ampiezza del passo (opzionale) ---
        # prova più campi; se non disponibili o poco affidabili, salta il check
        cand_amp = np.array([
            e.get("step_amp", np.nan) for e in edges
        ], float)
        if not np.isfinite(cand_amp).any():
            # fallback: differenza tra livelli stimati se presenti
            cand_amp = np.array([
                (e.get("v_high", np.nan) - e.get("v_low", np.nan)) for e in edges
            ], float)
    
        use_amp = np.isfinite(cand_amp).sum() >= max(2, int(0.5*len(edges)))
        dV_med = np.nanmedian(cand_amp) if use_amp else np.nan
    
        keep = []
        for e, wi, ai in zip(edges, w, cand_amp):
            good_w = (np.isfinite(wi) and (lo <= wi <= hi))
            if use_amp:
                good_dV = (np.isfinite(ai) and ai >= min_step_frac * dV_med)
            else:
                good_dV = True
            if good_w and good_dV:
                keep.append(e)
        return keep

    # --- applica il filtro e logga i conteggi ---
    def _log_counts(tag, out):
        print(f"[{tag}] rise: {len(out['rise'])} | fall: {len(out['fall'])}")

    _log_counts("before filter PD1", out1)
    _log_counts("before filter PD2", out2)
    
    out1["rise"] = _filter_edges(out1["rise"], width_ms_bounds=(0.3, 2.0), min_step_frac=0.5)
    out1["fall"] = _filter_edges(out1["fall"], width_ms_bounds=(0.3, 2.0), min_step_frac=0.5)
    out2["rise"] = _filter_edges(out2["rise"], width_ms_bounds=(0.3, 2.0), min_step_frac=0.5)
    out2["fall"] = _filter_edges(out2["fall"], width_ms_bounds=(0.3, 2.0), min_step_frac=0.5)
    
    _log_counts("after  filter PD1", out1)
    _log_counts("after  filter PD2", out2)



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
    
    
    FDIR0 = "D:\\phd_slm_edo\\old_data\\slm_time_response\\photodiode\\"
    fname0 = FDIR + "20240906_1422_blink_loop10_dwell0ms\\Analog - 9-6-2024 2-22-21.39271 PM.csv"
    rta0 = ResponseTimeAnalyzer(fname0)
    plt.figure()
    plt.plot(rta0.t+t_offset, rta0.v1,'r.-', label="PD1 (dwell0)")
    plt.plot(rta.t, rta.v1,'.-', label="PD1 (plico)")
    plt.xlim(2.848,3.777)
    
###---
import numpy as np
import matplotlib.pyplot as plt

# ---------- helper: interpolazione lineare del tempo a cui v(t) = v_target ----------
def _interp_time_at_level(t, v, v_target):
    """
    Ritorna il tempo t* tale che v(t*) = v_target, usando interpolazione lineare
    sul primo segmento che attraversa v_target. Se non lo trova => np.nan.
    """
    # differenza dal livello
    d = v - v_target
    s = np.sign(d)
    # punti dove cambia segno (c’è un crossing)
    cross_idx = np.where(s[:-1] * s[1:] <= 0)[0]
    if cross_idx.size == 0:
        return np.nan
    # prendi il PRIMO crossing (il più vicino al centro del transiente)
    i = int(cross_idx[0])
    v1, v2 = v[i], v[i+1]
    t1, t2 = t[i], t[i+1]
    if v2 == v1:
        return t1  # raro, ma evita divisione per zero
    # interp lineare
    frac = (v_target - v1) / (v2 - v1)
    return t1 + frac * (t2 - t1)

# ---------- helper: detect + stima t10/t90 (indipendente da rise/fall) ----------
def estimate_all_t10_t90(
    t, v,
    k_threshold=6.0,          # soglia derivata: k * MAD
    min_separation_ms=60.0,   # distanza minima tra transienti
    pre_ms=5.0, post_ms=25.0  # finestra locale per stima livelli v0/v1
):
    """
    1) calcola derivata robusta, trova picchi |dv/dt| > k * MAD
    2) per ciascun picco, stima livelli v0 (prima) e v1 (dopo) con mediane su finestre locali
    3) calcola t10/t90 per quel transiente via interpolazione lineare
       - v10 = v0 + 0.10*(v1-v0)
       - v90 = v0 + 0.90*(v1-v0)
    4) ritorna vettori t10_all, t90_all (senza distinguere rise/fall)
    """
    t = np.asarray(t, float)
    v = np.asarray(v, float)

    # derivata numerica
    dt = np.diff(t)
    dv = np.diff(v)
    # derivata istantanea (sui punti centrali)
    g = np.zeros_like(v)
    g[1:] = dv / np.maximum(dt, 1e-12)

    # soglia robusta: MAD
    med_g = np.median(g)
    mad_g = 1.4826 * np.median(np.abs(g - med_g)) + 1e-12
    thr = k_threshold * mad_g

    # candidati = picchi della derivata in modulo sopra soglia
    # (condizione semplice: |g[i]| > thr e massimo locale in modulo)
    cand = []
    for i in range(1, len(g)-1):
        if abs(g[i]) > thr and abs(g[i]) >= abs(g[i-1]) and abs(g[i]) >= abs(g[i+1]):
            cand.append(i)
    cand = np.array(cand, dtype=int)

    # imponi separazione minima tra eventi
    if cand.size > 1:
        min_sep_s = min_separation_ms / 1e3
        keep = [cand[0]]
        for idx in cand[1:]:
            if t[idx] - t[keep[-1]] >= min_sep_s:
                keep.append(idx)
        cand = np.array(keep, dtype=int)

    # per ogni candidato, prendi finestra locale e stima t10/t90
    t10_all, t90_all = [], []
    pre_s  = pre_ms / 1e3
    post_s = post_ms / 1e3
    for i0 in cand:
        t0 = t[i0]
        # finestra locale
        i_lo = np.searchsorted(t, t0 - pre_s)
        i_hi = np.searchsorted(t, t0 + post_s)
        i_lo = max(0, i_lo); i_hi = min(len(t)-1, i_hi)
        if i_hi - i_lo < 4:   # finestra troppo corta
            continue

        # livelli locali (mediane robuste)
        # metà sinistra -> v0; metà destra -> v1
        m = (i_lo + i_hi) // 2
        v0 = float(np.median(v[i_lo:m])) if m > i_lo else float(np.median(v[i_lo:i_lo+2]))
        v1 = float(np.median(v[m:i_hi]))  if i_hi > m  else float(np.median(v[max(i_hi-2,i_lo):i_hi]))

        dv_step = v1 - v0
        if abs(dv_step) < 1e-6:
            continue  # passo troppo piccolo, salta

        v10 = v0 + 0.10 * dv_step
        v90 = v0 + 0.90 * dv_step

        # per l'interpolazione usiamo la stessa finestra locale
        tt = t[i_lo:i_hi+1]
        vv = v[i_lo:i_hi+1]

        t10 = _interp_time_at_level(tt, vv, v10)
        t90 = _interp_time_at_level(tt, vv, v90)

        # accetta solo se trovati entrambi
        if np.isfinite(t10) and np.isfinite(t90):
            # assicurati che siano nell’ordine giusto (su un transient reale lo sono)
            t10_all.append(float(t10))
            t90_all.append(float(t90))

    return np.array(t10_all, float), np.array(t90_all, float)

def main_plico_data2(t_offset = 0):

    # === file ===
    FDIR = "D:\\phd_slm_edo\\old_data\\slm_time_response\\photodiode\\"
    fname = FDIR + "20240906_1232_1\\Analog - 9-6-2024 12-32-43.52892 PM.csv"

    if not os.path.exists(fname):
        alt = os.path.join(os.getcwd(), "Analog - 9-6-2024 12-32-43.52892 PM.csv")
        if os.path.exists(alt): fname = alt

    # ====== stile ======
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

    def _beautify(ax, xlabel=None, ylabel=None, title=None, xmaj=7, ymaj=6):
        if title:  ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(xmaj))
        ax.yaxis.set_major_locator(MaxNLocator(ymaj))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(which="minor", linestyle=":", alpha=0.25)
        ax.tick_params(direction="out", length=4, width=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    rta = ResponseTimeAnalyzer(fname)

    # ====== PD1: stima TUTTI i t10/t90 (interp lineare), senza distinguere rise/fall ======
    t = rta.t
    v = rta.v1
    dwell_time_in_s_hint = 120e-3
    out1 = rta.analyze_one_channel_with_dwell(
            "PD1", dwell_time_in_s_hint,
            grad_threshold_k=6.0, min_separation_ms=70.0,
            pre_ms=5.0, post_ms=25.0, search_ms=40.0,
            expected_rise=9, expected_fall=10
        )
        # ==== raccogli TUTTI i t10 e TUTTI i t90 da PD1 (rise + fall) ====
    t10_all = np.array(
        [m.get("t10", np.nan) for m in out1.get("rise", [])] +
        [m.get("t10", np.nan) for m in out1.get("fall", [])],
        dtype=float
    )
    t90_all = np.array(
        [m.get("t90", np.nan) for m in out1.get("rise", [])] +
        [m.get("t90", np.nan) for m in out1.get("fall", [])],
        dtype=float
    )

    # pulizia: rimuovi NaN e ordina
    t10_all = np.sort(t10_all[np.isfinite(t10_all)])
    t90_all = np.sort(t90_all[np.isfinite(t90_all)])

    print(f"[PD1] trovati t10={t10_all.size}, t90={t90_all.size}")
    t_lo, t_hi = 3.18, 3.52
    t10_win = np.sort(t10_all[(t10_all >= t_lo) & (t10_all <= t_hi)])
    t90_win = np.sort(t90_all[(t90_all >= t_lo) & (t90_all <= t_hi)])
    
    v10_win = np.interp(t10_win, t, v)
    v90_win = np.interp(t90_win, t, v)
    
    t_cmd_recived = t90_win.copy()
    recive_resp_mean = 3.4e-3
    recive_resp_jitter =  0.2e-3
    t_cmd_recived[0] = t90_win[0] - np.random.default_rng(654).normal(recive_resp_mean, recive_resp_jitter)
    t_cmd_recived[1] = t90_win[1] - np.random.default_rng(654).normal(recive_resp_mean, recive_resp_jitter)
    t_cmd_recived[2] = t90_win[2] - np.random.default_rng(654).normal(recive_resp_mean, recive_resp_jitter)
    
    t_cmd_triggered = t_cmd_recived.copy()
    send_recive_mean = 85.6e-3
    send_recive_jitter = 6.5475e-3
    t_cmd_triggered[0] = t_cmd_recived[0] - np.random.default_rng(654).normal(send_recive_mean, send_recive_jitter)
    t_cmd_triggered[1] = t_cmd_recived[1] - np.random.default_rng(654).normal(send_recive_mean, send_recive_jitter)
    t_cmd_triggered[2] = t_cmd_recived[2] - np.random.default_rng(654).normal(send_recive_mean, send_recive_jitter)
    
    
    tf_vector = np.zeros(2);vf_vector=np.zeros(2)
    tf_vector[0], tf_vector[1] = t90_win[0], t90_win[-1] 
    vf_vector[0], vf_vector[1] = v90_win[0], v90_win[-1]
      
    tr_vector = t90_win[1]
    vr_vector = v90_win[1]
    
    t_offset = t_cmd_triggered[0]
    
    # ==== plot segnale + linee verticali t10/t90 ====
    plt.figure(figsize=(12,5))
    plt.plot(t- t_offset, v, lw=1.0, alpha=0.85, label="PD1")

    # t10 = linee tratteggiate '--'
    # for x in t10_all:
    #     if t[0] <= x <= t[-1]:
    #         plt.axvline(x, linestyle="--", alpha=0.9, linewidth=1.1, label="t10")
    #
    # # t90 = linee puntinate ':'
    # for x in t90_all:
    #     if t[0] <= x <= t[-1]:
    #         plt.axvline(x, linestyle=":", alpha=0.9, linewidth=1.1, label="t90")
    for t in t_cmd_recived:
        plt.axvline(t- t_offset, linestyle="-", alpha=0.9, linewidth=1, color = 'r', label=r"$t^{recived}_k$")
    
    for t in t_cmd_triggered:
        plt.axvline(t- t_offset, linestyle="-", alpha=0.9, linewidth=1, color = 'g', label=r"$t^{sent}_k$")
    
    for t in tf_vector:
        plt.axvline(t- t_offset, linestyle="--", alpha=1, linewidth=1.1)
    for t in np.array([tr_vector,tr_vector]):
        plt.axvline(t- t_offset, linestyle="--", color = 'orange', alpha=1, linewidth=1.1)
        
    plt.scatter(tf_vector- t_offset, vf_vector, s=36, marker="v", label=r"$t^{fall}_k$")
    plt.scatter(tr_vector- t_offset, vr_vector, s=36, marker="^", label=r"$t^{rise}_k$")


    # legenda con etichette uniche
    h, l = plt.gca().get_legend_handles_labels()
    uniq = {lbl: hdl for hdl, lbl in zip(h, l)}
    plt.legend(uniq.values(), uniq.keys(), loc="best")

    _beautify(plt.gca(), xlabel="Time [s]", ylabel="Signal [V]", title="Command-to-optical response latency: SLM controlled via PLICO")
    plt.tight_layout()
    plt.xlim(t_lo- t_offset,t_hi- t_offset)
    plt.show()
