"""Task 2: T(w)/T_H - 1 on a grid of (profile, Lam, w), for three T definitions.
Usage: python3 scan_c.py PROFILE_KEY [PROFILE_KEY ...]   (keys in PROFILES below)
Writes results/scan_<key>.json."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hawking_bvp import Tanh, LinCore, FlatCore, ShiftTanh, SumTanh, solve, temps
from predict_c import PROFILES as _PRED, BLIND2 as _B2

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
PROFILES = {
    'tanh05': lambda: Tanh(0.5),
    'tanh10': lambda: Tanh(1.0),
    'lin_a05_w04': lambda: LinCore(0.5, 0.4),
    'lin_a08_w02': lambda: LinCore(0.8, 0.2),
    'lin_a03_w06': lambda: LinCore(0.3, 0.6),
    'flat_m1': lambda: FlatCore(1, 1.0),
    'flat_m2': lambda: FlatCore(2, 1.0),
    'flat_m3': lambda: FlatCore(3, 1.0),
    'flat_m4': lambda: FlatCore(4, 1.0),
    'sh_q2_tp0125': lambda: ShiftTanh(2.0, 0.125),
    'sh_q2_tm0125': lambda: ShiftTanh(2.0, -0.125),
    'sh_q2_tp025': lambda: ShiftTanh(2.0, 0.25),
    'sh_q2_tm025': lambda: ShiftTanh(2.0, -0.25),
    'sh_q4_tp0125': lambda: ShiftTanh(4.0, 0.125),
    'sh_q4_tm0125': lambda: ShiftTanh(4.0, -0.125),
}
for _k in ('sum_asym1', 'sum_asym2', 'sum_mix3', 'sum_mix4', 'sum_mix5'):
    PROFILES[_k] = (lambda c: (lambda: SumTanh(c)))(_PRED[_k])
for _k in _B2:
    PROFILES[_k] = (lambda c: (lambda: SumTanh(c)))(_B2[_k])
LAMS = [10.0, 20.0, 40.0, 80.0]
WS = [0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]
SUFFIX = ''
if os.environ.get('HIGH_LAM'):  # extension run: Lam = 160, 320 on w <= 0.6
    LAMS, WS, SUFFIX = [160.0, 320.0], [0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6], '_hiLam'
    CHECK_HI = {(160.0, 0.1), (160.0, 0.4)}  # a finer-mesh check at Lam=320 exceeded the 1 GB cap
CHECK = {(80.0, 0.1), (80.0, 0.4), (10.0, 0.8)}

for key in sys.argv[1:]:
    prof = PROFILES[key]()
    L = prof.default_L()
    rec = dict(key=key, name=prof.name, kap=1.0, vpp0=prof.vpp0(),
               v_minusL=float(prof.v(-L)), v_plusL=float(prof.v(L)), L=L,
               mesh=dict(N=28, gamma=0.55, hmax=0.25), points=[])
    t0 = time.time()
    for Lam in LAMS:
        for w in WS:
            res = solve(prof, w, Lam)
            if res is None:
                rec['points'].append(dict(Lam=Lam, w=w, status='w>w_max'))
                continue
            T = temps(res, w)
            p = dict(Lam=Lam, w=w, status='ok', **{k: float(v) for k, v in T.items()},
                     unit_err=float(res['unit_err']), flux_err=float(res['flux_err']), nunk=res['nunk'])
            if (Lam, w) in CHECK or (SUFFIX and (Lam, w) in CHECK_HI):
                r2 = solve(prof, w, Lam, N=36, gamma=0.4, hmax=0.15, L=1.2 * L, check_flux=False)
                T2 = temps(r2, w)
                p['fine_mesh_dT_row'] = float(T2['T_row'] - T['T_row'])
                p['fine_mesh_dT_col'] = float(T2['T_col'] - T['T_col'])
            rec['points'].append(p)
            print(key, Lam, w, f"{T['T_row'] - 1:+.10e} {T['T_col'] - 1:+.10e} {T['T_n'] - 1:+.10e} u{res['unit_err']:.0e} f{res['flux_err']:.0e}", flush=True)
    rec['elapsed_s'] = time.time() - t0
    with open(os.path.join(OUT, f'scan_{key}{SUFFIX}.json'), 'w') as f:
        json.dump(rec, f, indent=1)
