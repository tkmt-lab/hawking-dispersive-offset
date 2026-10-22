# Entry point for a computational reader

Start with README.md, claims.json and VERIFICATION-RECORD.md. The main result is the formula for b0 in Section 7 of the article, paper-hawking-b0-v1.2-en.tex. Its status is a formal asymptotic calculation conditional on the assumptions A1–A3 of Section 7.1. Do not describe it as a theorem, a proof or a rigorous error bound, and do not extend it beyond the hypotheses of Section 7.1 (single simple horizon, monotone flow with constant asymptotic velocities v_- < -1 < v_+ < 1, Lambda-independent flow, omega -> 0 taken first at fixed Lambda).

Run `python3 -B check_packet.py` for the file inventory, claim scopes and privacy checks, and `python3 -B reproduce.py` (about 35 seconds) to re-run the exact local expansion, the zero-frequency formula at every stored cutoff, two scattering limits, the stored-scan analyses, every table and figure, and the comparison with the independent check. Both work without network access. Full scattering scans (`src/scan_c.py`, `b0/check_steps.py`) take minutes to hours and are not needed for any check.

Evidence types are kept apart in claims.json: formal derivation under A1–A3; numerical (same code base); independent numerical (separately written code in independent/, which shares no code with src/ or b0/); and AI review (AI readers of the derivation and of the article, not human peer review). The independent check is numerical agreement on four profiles. Its 42-condition ranges are extrapolation sensitivities, not confidence intervals; for one profile (T05, exact b0 = 0) the range misses 0 by 5.7e-9.

Profile keys in the data map to article labels as follows: tanh05 = T_{0.5}, tanh10 = T_1, flat_m* = F_*, sh_q*_t* = S_{q,t}, sum_asym1/2 = A_+/A_-, sum_mix3/4/5 = M_3/M_4/M_5, blind_a/b = B_1/B_2 (held out for the curvature comparison), blind_erf = B_3, blind_tcub = B_4 (prediction-first families); in independent/, T05 = T_{0.5}, B4 = B_4, N1 and N2 as in Eq. (12). The full table is exhibits/all-profiles.md.

The curvature comparison with Eq. (133) of Del Porro, Liberati and Schneider is a statement about different coefficients on these backgrounds, not a claim that their analysis is wrong.
