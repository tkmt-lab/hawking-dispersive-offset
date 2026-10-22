# Independent numerical check of b0

This directory holds a second scattering code, written from the problem definition below only, its stored solves and its analysis, and the predictions it was compared with. It shares no code with `src/` or `b0/`. Its author was instructed not to open the main code, its data, the article or the predictions, and records that it did not. The predictions (`predictions-sealed.json`, computed by `predict_sealed.py` from Eq. (19) of the article) were recorded before the independent computation and were not given to it.

## Problem definition given to the independent implementation

For a velocity field v(x) (smooth, monotone, exponentially approaching constants v_- < -1 and -1 < v_+ < 1 as x -> -inf, +inf, v(0) = -1, v'(0) = kappa = 1):

- wave equation (d_t + d_x v)(d_t + v d_x) phi = d_x^2 phi - Lambda^-2 d_x^4 phi, the first factor acting as d_t + d_x(v .); with phi = e^{-i omega t} f(x): Lambda^-2 f'''' = (omega^2 + i omega v') f + (2 i omega v - 2 v v') f' + (1 - v^2) f'';
- at constant velocity (omega - v k)^2 = k^2 + k^4/Lambda^2, Omega_c = omega - v k;
- left (supersonic): four real roots below the dispersive cutoff; right (subsonic): two propagating, one decaying and one growing root; the growing solution on the right is excluded;
- incoming channels: right co-propagating, left high-wave-number positive- and negative-norm; outgoing: right counter-propagating u and two on the left; norm sign sgn Omega_c; direction from the laboratory group velocity v_g = (v Omega_c + k + 2k^3/Lambda^2)/Omega_c;
- conserved current, with D = -i omega + v d_x: J(f1,f2) = v[f1* D f2 - f2 (D f1)*] - (f1* f2' - f2 f1*') + Lambda^-2 [f1* f2''' - f1*' f2'' + f1*'' f2' - f1*''' f2]; for f = a e^{ikx}, (i/2) J(f,f) = Omega_c v_g |a|^2;
- amplitudes normalized by sqrt|Omega_c v_g|, so that S^dagger eta S = eta, eta = diag(1,1,-1) in norm-sign order; on the outgoing u row |alpha|^2 = |S_{u,+}|^2, |beta|^2 = |S_{u,-}|^2, G = |S_{u,v}|^2;
- observable T_ratio(omega) = omega / log(|alpha|^2/|beta|^2), T_H = kappa/(2 pi);
- flows: T05 v = -1 + 0.5 tanh(2x); B4 v = -1 + 0.6 tanh(x/0.6 - 0.7x^2 + 0.5x^3); N1 v = -1 + 0.5 tanh(2x + 0.4x^2 + 0.3x^3); N2 v = -1 + A[erf((x+0.3)/0.7) - erf(0.3/0.7)], A = 0.7 sqrt(pi)/2 exp((0.3/0.7)^2);
- task: Lambda = 10, 20, 40, 80 (and 160), small omega (0.02, 0.01, 0.005, 0.0025), extrapolate T_ratio to omega -> 0, then Lambda^2 [T(0)/T_H - 1] in Lambda^-2 to b0_num, with the spread over extrapolation bases and windows; record the result before learning the predictions.

## Method

State y = (f, f'/Lambda, f''/Lambda^2, p) with p = [Lambda^-2 f''' + (v^2 - 1) f' - i omega v f]/(i omega), which gives the first-order system f' = Lambda y1, y1' = Lambda y2, y2' = i omega v f + Lambda (1 - v^2) y1 + i omega p, p' = -i omega f + Lambda v y1. On each short interval the 4x4 fundamental matrix is integrated with a six-stage Gauss–Legendre implicit Runge–Kutta method (order 12); the interval matching and both boundary conditions are solved together as one sparse linear system, so no long transfer-matrix product is formed. The four roots of the dispersion relation are classified by sign of Omega_c and v_g; propagating basis vectors are scaled by 1/sqrt|Omega_c v_g|; the growing root on the right has coefficient zero.

Runs (all 4 flows x Lambda = 10, 20, 40, 80, 160 x omega = 0.02, 0.01, 0.005, 0.0025, 80 points each):

- `refined.json` (central): domain [-9, 9], h Lambda <= 0.5, Gauss six-stage;
- `baseline.json`: [-7, 7], h Lambda <= 1;
- `dop853.json`: [-9, 9], h Lambda <= 1, DOP853 with local rtol 2e-13 (different integrator, same state and sparse boundary solve);
- `bvp-confirmation.json`: the original fourth-order equation with SciPy's adaptive cubic collocation (`check_bvp.py`), 4 flows at Lambda = 10, omega = 0.01, tol 1e-7 (flow and root routines shared with `scatter.py`).

| Check | points | max abs difference of delta from refined | max Lambda^2 x difference |
|---|---:|---:|---:|
| baseline | 80 | 8.765e-13 | 2.244e-08 |
| dop853 | 80 | 1.044e-12 | 2.192e-08 |
| bvp-confirmation | 4 | 2.097e-10 | 2.097e-08 |

Pseudo-unitarity residual max |S^dagger eta S - eta|: baseline 4.645e-11, refined 6.131e-11, dop853 3.586e-11, bvp-confirmation 4.866e-08. `check_definitions.py` checks the state transformation, the current and constant-medium propagators (relative differences below 2.3e-16; `definition-checks.json`); `check_extrapolation.py` repeats the central extrapolation with 60-digit decimal normal equations (`extrapolation-check.json`).

## Extrapolation and result

Central value: delta(omega) = T_ratio/T_H - 1 fitted quadratically in omega^2 on all four frequencies, intercept delta(0); then Lambda^2 delta(0) fitted quadratically in Lambda^-2 at Lambda = 20, 40, 80, 160. All fits are unweighted least squares with the abscissa scaled by its maximum (`analyze.py`, output `analysis.json`). The sensitivity range combines 6 frequency fits (omega^2 quadratic/linear on all four or the lowest three points; omega linear on the lowest three; omega quadratic on all four) with 7 cutoff fits (quadratic 20–160 [central], linear 40–160, linear 80–160, quadratic 40–160, cubic 10–160, quadratic 10–160, linear 20–160): 42 combinations. **The ranges are the sensitivity of the chosen extrapolation formulas, not confidence intervals or error bounds.**

| Flow | predicted b0 (Eq. (19)) | b0_num (central) | difference | range over 42 combinations | contains prediction |
|---|---:|---:|---:|---:|---|
| T05 | 0 | -4.103593092e-08 | -4.1e-08 | [-1.235234539e-06, -5.652460886e-09] | no, misses 0 by 5.7e-09 |
| B4 | 1.1336017778 (= 637651/562500) | 1.133601772 | -5.8e-09 | [1.133554247, 1.133602166] | yes |
| N1 | -0.6373333333 (= -239/375) | -0.6373333022 | +3.1e-08 | [-0.6373333541, -0.637283084] | yes |
| N2 | -0.4164931279 | -0.4164932353 | -1.1e-07 | [-0.4166146727, -0.4164854915] | yes |

The largest change of the central b0 between the solver variants above was 7.9e-9 (T05), 5.1e-9 (B4), 1.24e-8 (N1) and 6.8e-9 (N2). For T05 neither "exactly zero" nor "negative" is concluded from the computation itself. The agreement is numerical evidence for Eq. (19) on four flows; it does not prove the assumptions A1–A3 of the article.

Re-running the 80 central solves with other library versions changes T/T_H - 1 by about 1e-12 and the extrapolated b0 by up to a few times 1e-8 (see `reproduce.py`, step 6); the differences from the predictions then remain of order 1e-7.

## Reproduce

```sh
python3 -B scatter.py --profiles T05 B4 N1 N2 --lambdas 10 20 40 80 160 \
  --omegas .02 .01 .005 .0025 --step .5 --extent 9 --output replay-refined.json
```

(a few seconds; write to a new file so as not to overwrite the stored `refined.json`). `--method dop853 --step 1 --extent 9` gives the DOP853 variant; `check_bvp.py --profiles T05 B4 N1 N2 --tol 1e-7 --output <file>` the collocation check. `analyze.py`, `check_definitions.py` and `check_extrapolation.py` rewrite their output files next to themselves; run them in a copy. The original computation used Python 3.14, NumPy 2.5.4 and SciPy 1.18.1, single-threaded BLAS, a 3000 MB / 1800 s cap per run.

Two pilot attempts of the collocation check with complex unknowns or tol 1e-9 did not converge within the node limit and were not used; only the tol 1e-7 run with real and imaginary parts separated is reported above.
