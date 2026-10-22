#!/usr/bin/env python3
"""Blind numerical extrapolation, with window/basis sensitivity retained."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
FREQ_SPECS=[
    ('even_quadratic_all',2,2,0),
    ('even_linear_all',2,1,0),
    ('even_linear_low3',2,1,1),
    ('even_quadratic_low3',2,2,1),
    ('ordinary_linear_low3',1,1,1),
    ('ordinary_quadratic_all',1,2,0),
]
LAMBDA_SPECS=[
    ('quadratic_20_160',2,20),
    ('linear_40_160',1,40),
    ('linear_80_160',1,80),
    ('quadratic_40_160',2,40),
    ('cubic_10_160',3,10),
    ('quadratic_10_160',2,10),
    ('linear_20_160',1,20),
]


def fit(x,y,degree):
    # Scale abscissa to avoid artificial conditioning from small omega/Lambda^-1.
    x=np.asarray(x); y=np.asarray(y)
    scale=max(abs(x))
    a=np.polynomial.polynomial.polyvander(x/scale,degree)
    coef,_,rank,_=np.linalg.lstsq(a,y,rcond=None)
    if rank!=degree+1: raise ValueError('rank deficient extrapolation')
    return {'intercept':float(coef[0]),'coefficients_scaled':coef.tolist(),
            'x_scale':float(scale),'max_residual':float(max(abs(a@coef-y))),
            'condition_number':float(np.linalg.cond(a)),'points':len(x)}


def extrapolate(rows):
    answer={}
    for name in ['T05','B4','N1','N2']:
        lamvals=sorted({r['Lambda'] for r in rows if r['profile']==name})
        freq={}; t0=[]
        for label,power,degree,skip in FREQ_SPECS:
            freq[label]={}
            for lam in lamvals:
                rr=sorted([r for r in rows if r['profile']==name and r['Lambda']==lam],key=lambda r:-r['omega'])[skip:]
                fitrow=fit([r['omega']**power for r in rr],[r['T_over_TH_minus1'] for r in rr],degree)
                fitrow['omega_window']=[r['omega'] for r in rr]
                freq[label][str(lam)]=fitrow
        for lam in lamvals:
            d=freq['even_quadratic_all'][str(lam)]['intercept']
            vals=[freq[s[0]][str(lam)]['intercept'] for s in FREQ_SPECS]
            evenvals=[freq[s[0]][str(lam)]['intercept'] for s in FREQ_SPECS if s[1]==2]
            t0.append({'Lambda':lam,'T0_over_TH_minus1':d,'Lambda2_delta':lam*lam*d,
                       'all_frequency_basis_minmax':[min(vals),max(vals)],
                       'even_frequency_basis_minmax':[min(evenvals),max(evenvals)]})
        extrap=[]
        for flabel,_,_,_ in FREQ_SPECS:
            for llabel,degree,minlam in LAMBDA_SPECS:
                ll=[l for l in lamvals if l>=minlam]
                row=fit([1/l**2 for l in ll],[l*l*freq[flabel][str(l)]['intercept'] for l in ll],degree)
                row.update(frequency_basis=flabel,Lambda_basis=llabel,Lambda_window=ll)
                extrap.append(row)
        central=next(r for r in extrap if r['frequency_basis']=='even_quadratic_all' and r['Lambda_basis']=='quadratic_20_160')['intercept']
        centralomega=[r['intercept'] for r in extrap if r['frequency_basis']=='even_quadratic_all']
        centrallambda=[r['intercept'] for r in extrap if r['Lambda_basis']=='quadratic_20_160']
        allvals=[r['intercept'] for r in extrap]
        answer[name]={'b0_num':central,'Lambda_window_basis_minmax':[min(centralomega),max(centralomega)],
                      'frequency_window_basis_minmax':[min(centrallambda),max(centrallambda)],
                      'combined_minmax':[min(allvals),max(allvals)],'T0_rows':t0,'frequency_fits':freq,
                      'all_extrapolations':extrap}
    return answer


def compare(ref,other):
    key=lambda r:(r['profile'],r['Lambda'],r['omega'])
    bykey={key(r):r for r in ref}
    diffs=[]
    for r in other:
        b=bykey[key(r)]
        diffs.append({'profile':r['profile'],'Lambda':r['Lambda'],'omega':r['omega'],
                      'delta_difference':r['T_over_TH_minus1']-b['T_over_TH_minus1'],
                      'Lambda2_delta_difference':r['Lambda']**2*(r['T_over_TH_minus1']-b['T_over_TH_minus1'])})
    return {'points':len(diffs),'max_abs_delta_difference':max(abs(r['delta_difference']) for r in diffs),
            'max_abs_Lambda2_delta_difference':max(abs(r['Lambda2_delta_difference']) for r in diffs),'rows':diffs}


def main():
    names=['baseline','refined','dop853','bvp-confirmation']
    sets={name:json.loads((ROOT/(name+'.json')).read_text()) for name in names}
    assert len(sets['refined'])==80
    assert all(r['success'] for r in sets['bvp-confirmation'])
    out={'central_dataset':'refined.json','central_frequency_basis':'quadratic in omega^2, all four omega',
         'central_Lambda_basis':'quadratic in Lambda^-2, Lambda20..160',
         'range_meaning':'unweighted deterministic fit sensitivity, not a confidence interval or rigorous error bound',
         'profiles':extrapolate(sets['refined']),
         'integrator_b0':{name:{n:r['b0_num'] for n,r in extrapolate(sets[name]).items()} for name in ['baseline','dop853']},
         'comparisons':{name:compare(sets['refined'],sets[name]) for name in names if name!='refined'},
         'diagnostics':{},'input_sha256':{name:hashlib.sha256((ROOT/(name+'.json')).read_bytes()).hexdigest() for name in names}}
    for name,rows in sets.items():
        fields=['pseudo_unitarity_max']
        if name!='bvp-confirmation':
            fields+=['u_row_identity_error','flux_drift_max','plane_wave_flux_error','linear_residual_max','boundary_forbidden_amplitude']
        out['diagnostics'][name]={f:max(r[f] for r in rows) for f in fields}
    (ROOT/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    for n,r in out['profiles'].items():
        print(n,'b0',r['b0_num'],'Lambda range',r['Lambda_window_basis_minmax'],'omega range',r['frequency_window_basis_minmax'],'combined',r['combined_minmax'])
    for n,r in out['comparisons'].items():
        print(n,r['max_abs_delta_difference'],r['max_abs_Lambda2_delta_difference'])
    print('diagnostics',out['diagnostics'])


if __name__=='__main__':main()
