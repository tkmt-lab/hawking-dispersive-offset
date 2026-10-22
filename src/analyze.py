"""Task 2 analysis: from results/scan_*.json, fit
    Delta(w) = T(w)/T_H - 1 = a0 + a2 w^2 + a4 w^4 [+ a6 w^6]   (kap = 1)
at each Lam, and report c(Lam) = a2 * Lam^2 (the coefficient of w^2/Lam^2),
for the three T definitions.  Then extrapolate c(Lam) to Lam -> infinity with
c(Lam) = c_inf + b/Lam^2 (+ d/Lam^4).  Compare with DPLS eq. (133):
    c_DPLS = 1/4 - 3/8 * v''(0)/kap^2."""
import glob, json, os, sys
import numpy as np

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


def fit(ws, ds, order, wmax):
    m = ws <= wmax + 1e-12
    x, y = ws[m], ds[m]
    cols = [x ** (2 * j) for j in range(order + 1)]
    A = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.max(np.abs(A @ coef - y))) if len(y) > order + 1 else 0.0
    return coef, resid, int(m.sum())


summary = {}
lines = ['# c = (coefficient of w^2 in T/T_H - 1) * Lam^2 ; kap = 1, w = Killing (lab) frequency',
         '# fits: A = (a0,a2,a4) on w<=0.4 ; B = (a0,a2,a4,a6) on w<=0.6 ; C = (a0,a2) on w<=0.15',
         '# key                v\'\'(0)  c_DPLS   def   Lam |  a0(=Delta at w->0)     c_A          c_B          c_C       maxres_A']
recs = {}
for fn in sorted(glob.glob(os.path.join(OUT, 'scan_*.json'))):
    r = json.load(open(fn))
    if r['key'] in recs:  # merge the high-Lam extension
        recs[r['key']]['points'] += r['points']
    else:
        recs[r['key']] = r
pred = {}
for pf in ('c_predictions_preregistered.json', 'c_predictions_blind_stage2.json'):
    if os.path.exists(os.path.join(OUT, pf)):
        pred.update(json.load(open(os.path.join(OUT, pf))))
for key, rec in sorted(recs.items()):
    cD = 0.25 - 0.375 * rec['vpp0']
    S = dict(name=rec['name'], vpp0=rec['vpp0'], c_DPLS=cD, v_minusL=rec['v_minusL'], v_plusL=rec['v_plusL'],
             by_def={})
    for dname in ('T_row', 'T_col', 'T_n'):
        per = []
        for Lam in sorted(set(p['Lam'] for p in rec['points'])):
            pts = [p for p in rec['points'] if p['Lam'] == Lam and p['status'] == 'ok']
            ws = np.array([p['w'] for p in pts])
            ds = np.array([p[dname] - 1 for p in pts])
            cA, rA, nA = fit(ws, ds, 2, 0.4)
            cB, rB, nB = fit(ws, ds, 3, 0.6)
            cC, rC, nC = fit(ws, ds, 1, 0.15)
            per.append(dict(Lam=Lam, a0=float(cA[0]), c_A=float(cA[1] * Lam ** 2), c_B=float(cB[1] * Lam ** 2),
                            c_C=float(cC[1] * Lam ** 2), a4Lam2=float(cA[2] * Lam ** 2), maxres_A=rA, nA=nA, nB=nB, nC=nC,
                            w_list=ws.tolist(), Delta=ds.tolist()))
            lines.append(f"{key:18s} {rec['vpp0']:+.3f} {cD:+.4f} {dname:5s} {Lam:5.0f} | {cA[0]:+.6e} {cA[1] * Lam**2:+.6e} {cB[1] * Lam**2:+.6e} {cC[1] * Lam**2:+.6e} {rA:.1e}")
        # extrapolation in Lam of c_A
        Ls = np.array([q['Lam'] for q in per])
        cs = np.array([q['c_A'] for q in per])
        ext = {}
        for lab, sel, nterm in (('20-80,1/L^2', Ls >= 20, 2), ('10-80,1/L^2+1/L^4', Ls >= 10, 3)):
            if sel.sum() >= nterm:
                A = np.vstack([Ls[sel] ** (-2 * j) for j in range(nterm)]).T
                co, *_ = np.linalg.lstsq(A, cs[sel], rcond=None)
                ext[lab] = dict(c_inf=float(co[0]), b=float(co[1]))
        # local scaling exponent of a2 between successive Lam: a2 ~ Lam^-p
        expo = []
        for i in range(len(per) - 1):
            a2a = per[i]['c_A'] / per[i]['Lam'] ** 2
            a2b = per[i + 1]['c_A'] / per[i + 1]['Lam'] ** 2
            if a2a * a2b > 0:
                expo.append(float(np.log(abs(a2a / a2b)) / np.log(per[i + 1]['Lam'] / per[i]['Lam'])))
            else:
                expo.append(None)
        b0s = [q['a0'] * q['Lam'] ** 2 for q in per]
        S['by_def'][dname] = dict(per_Lam=per, extrap=ext, a2_scaling_exponents=expo, a0_times_Lam2=b0s)
        lines.append(f"{key:18s}  {dname}: extrap c_inf {ext} ; a2 ~ Lam^-p, p = {expo}")
    if key in pred:
        S['c_formula_pred'] = pred[key].get('c_formula')
        S['b0_guess_pred'] = pred[key].get('b0_guess')
    lines.append(f"{key:18s}  PRED c_formula={S.get('c_formula_pred')} b0_guess={S.get('b0_guess_pred')} ; measured a0*Lam^2 (T_row) = {[round(b, 5) for b in S['by_def']['T_row']['a0_times_Lam2']]}")
    summary[key] = S
json.dump(summary, open(os.path.join(OUT, 'c_summary.json'), 'w'), indent=1)
open(os.path.join(OUT, 'c_summary.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(l for l in lines if ' T_row ' in l or 'PRED' in l or 'T_row: extrap' in l or l.startswith('#')))

# ---- final table: Richardson-type extrapolation on the three largest Lam (c = c_inf + b/Lam^2 + d/Lam^4)
fin = ['# key | v\'\'(0)/kap^2 | c_DPLS | c_formula (local inverse-series) | c(T_row) at largest Lam | c_inf (3 largest Lam) | '
       'b0 = Lam^2 (T/T_H-1)|_{w->0} at largest Lam | b0_inf | b0_guess',
       '# kap = 1.  lin_* rows: kinked profiles (v\'\'\' jumps), c does not settle for Lam <= 80; excluded from the conclusion.']
FIN = {}
for key, S in sorted(summary.items()):
    per = S['by_def']['T_row']['per_Lam']
    Ls = np.array([q['Lam'] for q in per])[-3:]
    cs = np.array([q['c_A'] for q in per])[-3:]
    bs = np.array([q['a0'] * q['Lam'] ** 2 for q in per])[-3:]
    A = np.vstack([np.ones(3), Ls ** -2.0, Ls ** -4.0]).T
    cinf = float(np.linalg.solve(A, cs)[0])
    binf = float(np.linalg.solve(A, bs)[0])
    cf = S.get('c_formula_pred')
    if cf is None and abs(S['vpp0']) < 1e-14 and key.startswith(('tanh', 'flat', 'lin')):
        cf = 0.0
    FIN[key] = dict(vpp0=S['vpp0'], c_DPLS=S['c_DPLS'], c_formula=cf, Lam_max=float(Ls[-1]), c_at_Lam_max=float(cs[-1]),
                    c_inf=cinf, b0_at_Lam_max=float(bs[-1]), b0_inf=binf, b0_guess=S.get('b0_guess_pred'),
                    Lams_used=Ls.tolist())
    fs = lambda x: 'n/a' if x is None else f'{x:+.6f}'
    fin.append(f"{key:14s} {S['vpp0']:+.3f} {S['c_DPLS']:+.4f} {fs(cf):>10s} | {cs[-1]:+.6f} (Lam={Ls[-1]:.0f}) {cinf:+.6f} | {bs[-1]:+.6f} {binf:+.6f} {fs(S.get('b0_guess_pred')):>10s}")
json.dump(FIN, open(os.path.join(OUT, 'c_final.json'), 'w'), indent=1)
open(os.path.join(OUT, 'c_final_table.txt'), 'w').write('\n'.join(fin) + '\n')
print('\n'.join(fin))
