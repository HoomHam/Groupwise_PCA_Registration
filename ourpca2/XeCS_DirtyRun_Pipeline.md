# Dirty Run — registration → functional maps → report figures (no scrutiny)

**Status: LIVE recipe (2026-09-03).** Hooman's name for it: *"a complete dirty run
without my scrutiny."* Built 2026-08-31 → 09-03 for the RT pre/post progress
report (002ZS / 003PM / 004DS). Kept as the reusable pipeline for any session
where results are needed fast (grant, talk) — Hooman's judgement: the
un-scrutinized answer is already close to what the scrutinized answer will be;
future corrections likely move numbers a few percent, not the story.

Input: Steve's **Tyger** dynamic recon (`<study>/d/recon.mat`, 16-bin
`gas_phase` 100³) — historical cloud outputs live in
`/Volumes/HoomHamExt/AIkill_Dynamic/<date>_<subject>/`.

## Steps (all from `workspace/`, `.venv` unless noted)

| # | Step | Script | Notes |
|---|------|--------|-------|
| 1 | Groupwise registration | `.venv/bin/python helpers/dirtyrun_register.py <study> …` | ourpca2 = native D_PCA2 (L=None), mesh (5,7,9,17), shrink (4,2,1,1), iters (150,120,100,80), max_disp 12, lattice λ 1e-3. Union mask ≥4 % max in ≥5/16 bins, largest CC. ~5–6 min/study. → `outputs/physreg/<study>/{stack,mask_union,u_ourpca2,registered_ourpca2}.npy` |
| 2 | Functional maps | `.venv/bin/python helpers/dirtyrun_maps.py <study> …` | `gas_dynamics.shoot` per voxel on the registered cine, gauge mean v=1. γ = plateau log-ratio (h = `diaphragm_pos`), dt = breath period/16 (nav peaks), **S_inf = p99.5·k with k bisected in [1,20] until trimmed-mean FV = 0.25**. Maps TV=max−min g, FRC=min g, FV=Σ(Δg)⁺, TTP=argmax g. Check: ΣTV·0.0429 mL ∈ 250–800 mL. → `ledger_aikill_v2.npz` |
| 3 | Two-page figures | `.venv/bin/python helpers/dirtyrun_fig.py <SUBJ> …` | Edit `SUBJECTS` (pre/post dirs + Hooman's 10 y-indices, post shifted by the xcorr offset). 8 paired rows × 10 cols, display `fliplr(rot90(slice.T,k=-1))`, **one shared crop box per subject**, TTP in seconds (hot_r, slow = dark red), FV jet. Page 2 = per-slice hists. |
| 4 | Narration | (chat) | Global / FRC / TV-FV / TTP / one-liner / caveats. Medians over the intersection slab; quote mask volume, period, ΣTV, γ. |

Optional: `helpers/dirtyrun_mrd_dump.py <tyger_run_dir> <outdir>` (**base
python3**, needs `mrd`) — Tyger `input.mrd` → recon_io_dyn dump for our CS
pipelines (`pipeline/cine_4d.py`), no `.dat` hunt.

## Results on file (RT cohort)

| Subject | pre → post | mask L | period s | γ 1/s | k | ΣTV mL | headline |
|---|---|---|---|---|---|---|---|
| 003PM | 2023-12-06 → 2024-06-18 | 2.84→2.07 | 5.07→3.13 | 0.29→0.43 | 2.67/2.32 | 653→503 | diffuse FRC ↑52 %, TV redistributed, later-phase TTP |
| 002ZS | 2023-11-06 → 2024-05-13 | 2.94→2.59 | 4.17→2.46 | 0.36→0.43 | 3.08/2.80 | 551→468 | one lung leaves the field; surviving lung per-voxel unchanged; rate +70 % |
| 004DS | 2024-01-16 → 2024-07-29 | 3.11→2.25 | 3.45→3.13 | 0.21→0.19 | 1.75/1.45 | 796→734 | −28 % field, ΣTV kept, focal confluent FRC ↑ |

Figures: `outputs/physreg/{002ZS,003PM,004DS}_maps_page{1,2}.png`
(003PM also as `pm_maps_v2_page{1,2}.png`), videos
`outputs/physreg/<study>/<study>_ourpca2_2x2.mp4` (003PM).

## EBV cohort (2026-09-09) — additions to the recipe
Report: `outputs/physreg/EBV_DirtyRun_Report_2026-09-09.md`. Pairs 001JM / 003KH / 008CR /
009JT (005DS post has no Tyger run). New standing steps, all carded helpers:

| # | Step | Script | Why |
|---|------|--------|-----|
| 0a | Raw inventory | by hand → `notes/EBV_Acquisitions_2026-09-09.md` | Hooman: some sessions ran the dynamic twice minutes apart; the breath-hold static shares the sequence name — classify by FID count, say which file the recon used |
| 0b | Breathing QC | `helpers/dirtyrun_breathing.py --tag <tag> <studies>` | plots nav + pneumo + k0 + per-bin signal; picks the period (pneumo if it agrees with k0/nav, else **k0 signal** — Hooman's rule when the nav fails; report which); flags double-humped per-bin signal = binning suspect |
| 0c | Constants | `outputs/physreg/<tag>_constants.json` (+ `_period_only`) | period + T1,eff per study. T1 rule: hold_T1 (GOOD/USABLE) > stat_T1 > group mean stat_T1 |
| 1b | Registration check | `helpers/dirtyrun_regcheck.py --tag <tag> <studies>` | temporal-CV ratio, |u| stats, det J, support excursion, 3-slice figure |
| 2' | Maps | `dirtyrun_maps.py --constants <json> --tag <ledger> --ee-roll` | γ = 1/T1,eff (plateau γ stored alongside), TTP from the signal-minimum bin (bin 0 ≠ EE for 3/8 sessions). Run twice: `ebv_t1` (T1 rule) and `ebv_plat` (period-only constants → plateau γ) |
| 3' | Figures | `dirtyrun_fig.py --ledger ledger_<x>.npz --suffix _<x> --auto SUBJ:pre:post` | auto slice table (10 slices over the pre mask, post lag by mask-profile xcorr) |
| 5 | Summary | `helpers/dirtyrun_summary.py --tag <tag> --ledger <x> SUBJ:pre:post …` | paired markers figure + md table |

**LTX cohort (2026-09-09, same day)**: `outputs/physreg/LTX_DirtyRun_Report_2026-09-09.md` —
043AS/039CP/036RL/044DY (13 sessions, longitudinal). Added `helpers/dirtyrun_constants.py`
(period + T1 rule → JSON), `helpers/dirtyrun_fig_long.py` (N-session pages, one per map),
`helpers/dirtyrun_process.sh` (regcheck → T1 ledger → plateau ledger), summary generalised to
N timepoints (x = months). Verdict: 1 stable, 1 improving, 2 with shrinking ventilated field.
Plateau γ = ×1.8–2.4 the T1,eff γ on all 13 sessions (the k0 excess-decay again).

**LTX two-session pairs (2026-09-09)**: `outputs/physreg/LTX2_DirtyRun_Report_2026-09-09.md` —
037GD/038RL/040RP/041WF + **047SS repeatability pair (1 week): ~10 % field, ~10–20 % ΣTV, ≤5 %
per-voxel medians = the dirty-run noise floor.** 040RP t2 (56 s, 32/min) unreadable.

Lessons: **FRC level flips sign with γ** (2/4 subjects) — quote FRC uniformity (p90/p10) not
level; hold_T1 can jump 3× pre→post in one subject (009JT) → plateau γ is the fairer pre/post
comparison there; the 4 %-of-max mask under-masks when an airway hot-spot sets the max
(008CR pre); registration ×0.66–0.82 temporal CV, no folding, on all 8.

## Caveats to attach every time
- Volumes are gauge units (mean v = 1) scaled by the S_inf knob — comparable
  as distributions, **not mL**; ΣTV mL uses the 3.5 mm voxel assumption.
- γ from 3–8 plateau transitions per study (T1,eff 2.3–5.3 s incl. RF).
- Convergence 80–99 %; non-converged voxels excluded; edge pileups at 0 and the
  τ cap are solver guards.
- "Absent" lung = below the 4 % mask threshold, not proof of zero ventilation.
- Never use the static scan's signal for the dynamic; each session self-calibrates.
- Model = cycle-closure ledger (V2-equivalent). Dynamic_Model V3–V5 exchange
  refinements NOT applied (`2026_Dynamic_Model/workspace/codes/`).

## Superseded scratch (kept, uncarded)
`_aikill_ledger_maps.py` (v1, p99.5 S_inf uncalibrated), `_aikill_ledger_maps2.py`
/ `_aikill_ledger_fig2.py` (003PM-only), `_aikill_ledger_fig.py` (v1 layout),
`_aikill_video.py`, `_jj_spatial_light.py`, `_jj_steve_vs_cs_fig.py`.
