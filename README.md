# Low-frequency offset of a dispersive analogue Hawking temperature

Release 1.0.0 of https://github.com/tkmt-lab/hawking-dispersive-offset (tag `v1.0.0`). The article is [paper-hawking-b0-v1.2-en.pdf](paper-hawking-b0-v1.2-en.pdf) (version 1.2, 15 pages).

For a stationary one-dimensional flow v(x) with sound speed 1, quartic superluminal dispersion ω² = k² + k⁴/Λ² in the comoving frame, and a single horizon v(0) = −1 with κ = v′(0) > 0, the article studies the temperature T_ratio = ω / log(|α|²/|β|²) from full three-channel scattering. Taking ω → 0 first at fixed Λ, it obtains

    T(0)/T_H − 1 = κ g‴(0) / (12 Λ²) + O(Λ⁻⁴),
    b0 = g‴(0)/(12κ) = [v⁗/κ⁴ − 4 v″v‴/κ⁵ + 3 (v″)³/κ⁶] / 12,

where g(u) = v′(x(u)) with u = 1 + v, and the derivatives are taken at the horizon. The O(Λ⁻⁴) term is the next order of a formal expansion; its coefficient b1 is also given.

**Scope.** The formula is the result of a formal asymptotic calculation under three stated assumptions (A1 low-frequency regularity and matching, A2 Abel treatment of oscillatory tails, A3 local evaluation of global integrals and of the travelling-wave normalization). A1–A3 are not proved, and there is no remainder bound. The calculation is supported by numerics: the zero-frequency reduction agrees with extrapolated ω → 0 scattering limits within 1.3×10⁻¹⁰ in 28 cases; predictions recorded before any calculation for two new profiles agree within 1.1×10⁻⁷; and a separately written scattering code reproduces the predicted b0 for four profiles, two of them new, within 1.1×10⁻⁷. The article also compares the frequency-curvature coefficient with a known phase-integral formula over 19 smooth profiles. There is no Lean proof and no human expert review.

## Check in a minute

Use Python 3.9+ with the packages in requirements.txt (`python3 -m pip install -r requirements.txt` in your own environment). From the repository root:

```sh
python3 -B reproduce.py
```

It works in a temporary copy and takes about 35 seconds on a laptop. It re-runs the exact rational expansion (b0 and b1 must come out as the stored rational coefficients), the zero-frequency formula for the seven profiles of Table 1 at every stored cutoff, two full scattering limits, the analyses of the stored scans, every table and figure of the article, and the independent check (its comparison with the recorded predictions, and a replay of its 80 central solves with your libraries). The last line is

```text
{"status": "PASS", "seconds": ..., "steps": ["1-exact-local-expansion", "2-zero-frequency-formula", "3-scattering-spot-check", "4-analyses-on-stored-scans", "5-tables-and-figures", "6-independent-check"]}
```

To see the formula at work on one profile, B4 of the article, v = −1 + 0.6 tanh(x/0.6 − 0.7x² + 0.5x³), solve the zero-frequency problem at three cutoffs and compare Λ²[T(0)/T_H − 1] with b0 + b1/Λ² from the local expansion (well under a second each):

```sh
python3 -B -c "
import json, sys; sys.path[:0] = ['b0', 'src']
import check_steps as cs
from zero_mode import zero_mode
p = json.load(open('b0/blind_predictions.json'))['blind_tcub']
for L in (20.0, 40.0, 80.0):
    z = zero_mode(cs.PROFILES['blind_tcub'](), L)
    print(f\"{L:4.0f}  {L**2*(z['T0_over_TH']-1):.7f}  {p['b0'] + p['b1']/L**2:.7f}\")
"
```

Expected output (b0 = 637651/562500 = 1.1336018 for this profile):

```text
  20  1.1350693  1.1350356
  40  1.1339624  1.1339602
  80  1.1336915  1.1336914
```

The two columns approach each other as Λ grows, as the next omitted term is of order Λ⁻⁴.

## Check the evidence

- Read [AGENTS.md](AGENTS.md), [claims.json](claims.json) and [VERIFICATION-RECORD.md](VERIFICATION-RECORD.md).
- `python3 -B check_packet.py` checks the file list and hashes, the claim scopes, the article's references and the absence of private paths.
- [b0/](b0) holds the zero-frequency solver, the exact local expansion and the stored stage-wise checks; [src/](src) the boundary-value scattering solver and the curvature scans; [results/](results) the stored scans; [exhibits/](exhibits) the tables, figures and their builder; [independent/](independent) the separately written scattering code, its stored solves and analysis, and the predictions it was compared with.
- New scattering scans are not needed for any table or figure. They can be re-run with `src/scan_c.py` (hours on one core) and `b0/check_steps.py`.

## Research and verification

The author is Tatsuhiko Takimoto, Independent researcher. AI systems assisted with the numerical implementation, the asymptotic calculation and its exact algebra, an independent check of that calculation, literature comparison and drafting; the author is responsible for the content and its stated scope. Process details (what was fixed before which result, how the independent code was kept separate, AI readings of the derivation and the article) are in [VERIFICATION-RECORD.md](VERIFICATION-RECORD.md).

## Built on prior work

- Jean Macher and Renaud Parentani's three-channel scattering description and the separation of a low-frequency temperature from spectral curvature ([Phys. Rev. D 79, 124008 (2009)](https://arxiv.org/abs/0903.2224v3)).
- Antonin Coutant and Silke Weinfurtner's zero-frequency matching procedure ([Korteweg–de Vries](https://arxiv.org/abs/1707.09651v2) and [Bogoliubov–de Gennes](https://arxiv.org/abs/1707.09664v2), Phys. Rev. D 97, 025005 and 025006 (2018)), which the low-frequency calculation follows.
- Ulf Leonhardt and Scott Robertson's phase-integral formula and inverse-flow method ([New J. Phys. 14, 053003 (2012)](https://doi.org/10.1088/1367-2630/14/5/053003)).
- Antonin Coutant, Renaud Parentani and Stefano Finazzi's analytical S-matrix methods ([Phys. Rev. D 85, 024021 (2012)](https://arxiv.org/abs/1108.1821v3)); Yoshiaki Himemoto and Takahiro Tanaka's analysis of higher-derivative corrections ([Phys. Rev. D 61, 064004 (2000)](https://arxiv.org/abs/gr-qc/9904076v3)); Alhan Moreno-Ruiz and David Bermudez's analytical and numerical comparison ([arXiv:1908.02368](https://arxiv.org/abs/1908.02368v1)).
- Francesco Del Porro, Stefano Liberati and Marc Schneider's tunneling method ([C. R. Phys. 25, S2 (2024; online 2025)](https://doi.org/10.5802/crphys.239)) and its connection with the Bogoliubov description ([arXiv:2604.24861](https://arxiv.org/abs/2604.24861v1)), whose effective-horizon coefficient is the comparison point for the frequency curvature.

I am grateful to them, and I hope this work adds something new to the field they opened. These acknowledgements imply no endorsement.

Code: Apache-2.0. Article, original text and generated data: CC BY 4.0. Contact: GitHub Issues at the repository.
